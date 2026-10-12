"""Comprobación independiente de predicciones, métricas y huella SHA-256."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

def main(root=Path('.')):
    root=Path(root)
    dat=root/'data/sst2_validation.csv'
    meta=json.loads((root/'data/sst2_validation.metadata.json').read_text())
    assert hashlib.sha256(dat.read_bytes()).hexdigest()==meta['sha256']
    reference=pd.read_csv(dat).sort_values('idx').reset_index(drop=True)
    prediction=pd.read_csv(root/'results/predicciones.csv')
    summary=pd.read_csv(root/'results/comparacion_modelos.csv')
    assert len(summary)==3 and len(reference)==872
    for _,row in summary.iterrows():
        p=prediction[prediction.modelo==row.modelo].sort_values('idx').reset_index(drop=True)
        assert len(p)==872
        assert np.array_equal(p.idx.to_numpy(),reference.idx.to_numpy())
        assert np.array_equal(p.label_real.to_numpy(),reference.label.to_numpy())
        assert np.array_equal(p.sentence.to_numpy(),reference.sentence.to_numpy())
        a=accuracy_score(p.label_real,p.label_predicha)
        pr,rc,f1,_=precision_recall_fscore_support(p.label_real,p.label_predicha,average='macro',zero_division=0)
        for metric,value in [('accuracy',a),('precision_macro',pr),('recall_macro',rc),('f1_macro',f1)]:
            assert np.isclose(float(row[metric]),value),f'{row.modelo}: {metric}'
        cm=confusion_matrix(p.label_real,p.label_predicha,labels=[0,1]).tolist()
        assert cm==json.loads(row.matriz_confusion)
        print(f'{row.modelo}: validado, accuracy={a:.4%}, F1 macro={f1:.4%}')
    print('Integridad de dataset, predicciones y métricas: OK')

if __name__=='__main__':main()
