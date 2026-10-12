"""Benchmark por lotes con pipeline HF, métricas y latencia con sincronización CUDA."""
import argparse
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime,timezone
from pathlib import Path
os.environ['TRANSFORMERS_NO_TF']='1'
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODELS={
    'distilbert':'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    'bert':'textattack/bert-base-uncased-SST-2',
    'roberta':'textattack/roberta-base-SST-2'
}

def to_id(name):
    value=str(name).strip().upper()
    if value in ('NEGATIVE','LABEL_0','0'):return 0
    if value in ('POSITIVE','LABEL_1','1'):return 1
    raise ValueError(f'Etiqueta sin mapeo explícito: {name!r}. Verifique config.id2label.')

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def measure_single(pipe,texts,n=80):
    sample=texts[:min(n,len(texts))]
    for s in sample[:5]:pipe(s,truncation=True,max_length=128)
    times=[]
    for s in sample:
        sync();t=time.perf_counter();pipe(s,truncation=True,max_length=128);sync()
        times.append((time.perf_counter()-t)*1000)
    return float(np.percentile(times,50)),float(np.percentile(times,95))

def run(csv_path='data/sst2_validation.csv',out_dir='results',allow_cpu=False,batch_size=32,model_revisions=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from huggingface_hub import HfApi
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('Requiere GPU CUDA. En Colab: Entorno de ejecución > Cambiar tipo > T4 GPU.')
    df=pd.read_csv(csv_path)
    assert set(['idx','sentence','label']).issubset(df.columns)
    assert df['idx'].is_unique and df['label'].isin([0,1]).all()
    assert df['sentence'].notna().all()
    texts=df['sentence'].astype(str).tolist();y=df['label'].astype(int).tolist()
    device=0 if torch.cuda.is_available() else -1
    path=Path(out_dir);path.mkdir(parents=True,exist_ok=True)
    revisions=model_revisions or {}
    summary=[];pred_tables=[]
    for name,mid in MODELS.items():
        print('Evaluando',name,mid)
        gc.collect()
        if device==0:torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        revision=revisions.get(name) or HfApi().model_info(mid).sha
        tok=AutoTokenizer.from_pretrained(mid,revision=revision)
        model=AutoModelForSequenceClassification.from_pretrained(mid,revision=revision)
        assert model.config.num_labels==2
        mapping={int(k):to_id(v) for k,v in model.config.id2label.items()}
        assert sorted(mapping.values())==[0,1],mapping
        clf=pipeline('text-classification',model=model,tokenizer=tok,device=device,framework='pt')
        for _ in range(3):clf(texts[:min(batch_size,len(texts))],batch_size=batch_size,truncation=True,max_length=128)
        sync();t0=time.perf_counter()
        outputs=clf(texts,batch_size=batch_size,truncation=True,max_length=128)
        sync();elapsed=time.perf_counter()-t0
        ypred=[to_id(x['label']) for x in outputs]
        # Identidad semántica comprobada en id2label (LABEL_0 => negativo en SST-2).
        acc=float(accuracy_score(y,ypred))
        p,r,f,_=precision_recall_fscore_support(y,ypred,average='macro',zero_division=0)
        pp,rr,ff,_=precision_recall_fscore_support(y,ypred,average='binary',pos_label=1,zero_division=0)
        cm=confusion_matrix(y,ypred,labels=[0,1]).tolist()
        p50,p95=measure_single(clf,texts)
        peak=round(torch.cuda.max_memory_allocated()/1024**2,1) if device==0 else None
        row={'modelo':name,'hf_id':mid,'revision':revision,'n':len(y),
             'accuracy':acc,'precision_macro':float(p),'recall_macro':float(r),'f1_macro':float(f),
             'precision_positiva':float(pp),'recall_positiva':float(rr),'f1_positiva':float(ff),
             'latencia_media_lote_ms_por_texto':1000*elapsed/len(y),
             'throughput_textos_s':len(y)/elapsed,'latencia_p50_individual_ms':p50,
             'latencia_p95_individual_ms':p95,'pico_vram_mib':peak,
             'parametros_millones':round(clf.model.num_parameters()/1e6,2),
             'matriz_confusion':json.dumps(cm)}
        summary.append(row)
        pred_tables.append(pd.DataFrame({'modelo':name,'idx':df['idx'],'label_real':y,
                                         'label_predicha':ypred,'confianza_predicha':[o['score'] for o in outputs]}))
        del clf,model,tok
        gc.collect()
        if device==0:torch.cuda.empty_cache()
    comp=pd.DataFrame(summary);comp.to_csv(path/'comparacion_modelos.csv',index=False)
    pd.concat(pred_tables,ignore_index=True).to_csv(path/'predicciones.csv',index=False)
    env={'utc':datetime.now(timezone.utc).isoformat(),'device':torch.cuda.get_device_name(0) if device==0 else 'CPU',
         'torch':torch.__version__,'transformers':__import__('transformers').__version__,
         'datasets_file':str(csv_path),'dataset_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
         'batch_size':batch_size,'max_length':128,'python':platform.python_version(),
         'models':{r['modelo']:r['revision'] for r in summary},
         'method':'end-to-end pipeline: tokenizacion + inferencia + postprocesamiento'}
    (path/'environment.json').write_text(json.dumps(env,indent=2,ensure_ascii=False),encoding='utf-8')
    print(comp[['modelo','accuracy','f1_macro','latencia_media_lote_ms_por_texto','latencia_p95_individual_ms','pico_vram_mib']].to_string(index=False))
    return comp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/sst2_validation.csv')
    p.add_argument('--out',default='results');p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();run(a.csv,a.out,a.allow_cpu,a.batch_size)
