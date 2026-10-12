"""Tests de calidad que no requieren descargar pesos ni GPU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import pandas as pd
from evaluate import to_id
from select_model import select

def test_mapping():
    assert [to_id(x) for x in ['LABEL_0','NEGATIVE','0','POSITIVE','LABEL_1','1']]==[0,0,0,1,1,1]

def test_gating():
    df=pd.DataFrame([{'modelo':'distilbert','accuracy':.92,'f1_macro':.91,'recall_positiva':.90,
                      'latencia_p95_individual_ms':20,'pico_vram_mib':800},
                     {'modelo':'bert','accuracy':.99,'f1_macro':.98,'recall_positiva':.98,
                      'latencia_p95_individual_ms':10,'pico_vram_mib':600}])
    assert select(df)=='distilbert'
    df.loc[df.modelo=='distilbert','f1_macro']=.8
    assert select(df) is None


def test_selection_colab_capitalization():
    df=pd.DataFrame([{'modelo':'DistilBERT','accuracy':.91055,'f1_macro':.91043,'recall_positiva':.93018,
                      'latencia_p95_individual_ms':14.0954,'pico_vram_mib':322.86},
                     {'modelo':'RoBERTa','accuracy':.94037,'f1_macro':.94034,'recall_positiva':.94595,
                      'latencia_p95_individual_ms':62.4660,'pico_vram_mib':553.25}])
    assert select(df)=='DistilBERT'
