"""Criterios de aprobación: evita declarar ganador si no se satisfacen umbrales."""
import argparse
import pandas as pd

# Solo la ficha de DistilBERT declara licencia Apache-2.0 con claridad.
LICENSE_APPROVED={'distilbert':True,'bert':False,'roberta':False}

def select(df):
    needed={'modelo','f1_macro','accuracy','recall_positiva','latencia_p95_individual_ms','pico_vram_mib'}
    if not needed.issubset(df.columns):raise ValueError(f'Columnas faltantes: {needed-set(df.columns)}')
    elig=df[(df['modelo'].astype(str).str.strip().str.lower().map(LICENSE_APPROVED).fillna(False)) &
            (df['f1_macro']>=.89)&(df['accuracy']>=.89)&(df['recall_positiva']>=.88)&
            (df['latencia_p95_individual_ms']<=80)&(df['pico_vram_mib']<=4096)]
    if elig.empty:return None
    return elig.sort_values(['f1_macro','latencia_p95_individual_ms'],ascending=[False,True]).iloc[0]['modelo']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--file',default='results/comparacion_modelos.csv')
    x=pd.read_csv(p.parse_args().file)
    winner=select(x)
    print(f'Modelo aprobado: {winner}' if winner else 'NINGUNO APROBADO: revisar desempeño, recursos o licencia')
