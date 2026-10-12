"""Obtiene GLUE/SST-2 validation (872) desde HF; no incluye datos en el repositorio."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime,timezone

def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data/sst2_validation.csv')
    p.add_argument('--revision',default=None,help='SHA del dataset para reproducibilidad estricta')
    a=p.parse_args()
    revision=a.revision or HfApi().dataset_info('nyu-mll/glue').sha
    ds=load_dataset('nyu-mll/glue','sst2',split='validation',revision=revision)
    assert len(ds)==872,f'Cambio inesperado en el benchmark: {len(ds)}'
    df=ds.to_pandas()[['idx','sentence','label']]
    assert df['label'].isin([0,1]).all() and df['idx'].is_unique
    df['source']='nyu-mll/glue:sst2';df['split']='validation'
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False,encoding='utf-8')
    manifest={'dataset':'nyu-mll/glue','subset':'sst2','split':'validation','revision':revision,
              'n':len(df),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'download_utc':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(),'path':str(out)}
    out.with_suffix('.metadata.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
