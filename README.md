# Evaluación comparativa de sentimiento con Hugging Face

**Objetivo:** identificar un modelo preentrenado de clasificación binaria de opiniones en inglés, maximizando desempeño y controlando latencia, VRAM y riesgo de licencia en Google Colab con GPU.

**Rama:** procesamiento del lenguaje natural (NLP). **Caso:** determinar sentimiento (0 = negativo; 1 = positivo) de frases de reseñas cinematográficas, como prototipo transferible solamente tras evaluación específica en el dominio objetivo.

## Ecosistema Hugging Face
- **Models:** repositorio de pesos, tokenizadores, revisiones y fichas de DistilBERT, BERT y RoBERTa; consultar licencias antes del uso.
- **Datasets:** dataset `nyu-mll/glue` / `sst2` / `validation` (872 filas etiquetadas) mediante `datasets.load_dataset`.
- **Spaces:** despliegues demostrativos con Gradio/Streamlit. No se publica un Space ni datos sensibles para esta práctica.

## Modelos

| Alias | Repositorio | Arquitectura | Licencia en ficha |
|---|---|---|---|
| distilbert | [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | DistilBERT base (≈67M parámetros) | Apache-2.0 |
| bert | [BERT](https://huggingface.co/textattack/bert-base-uncased-SST-2) | BERT base (≈110M) | **No declarada**: revisar antes de desplegar |
| roberta | [RoBERTa](https://huggingface.co/textattack/roberta-base-SST-2) | RoBERTa base (≈125M) | **No declarada**: revisar antes de desplegar |

El experimento se ejecutó en Colab Tesla T4 y se guardaron resultados reales comprobados de la misma sesión. **No se han simulado métricas ni mediciones GPU**. La comparación técnica final se encuentra en `docs/reporte_analisis_comparativo.docx`.

## Entorno
- Google Colab, **Entorno de ejecución > Cambiar tipo de entorno > T4 GPU** (CUDA). Acelera inferencia sin ajustar pesos.
- Python 3.10+; PyTorch instalado con CUDA en Colab, recomendable `torch>=2.6` para modelos BERT/RoBERTa con `.bin`.
- Paquetes fijados en [`requirements.txt`](requirements.txt). Se necesita Internet para pesos y dataset.
- Acceso a modelos públicos sin token; para autenticación explícita use **Secrets** con nombre `HF_TOKEN`. Nunca guarde tokens en el repositorio.

## Ejecución en Colab (vía notebook autocontenido)
1. Abra [`notebooks/comparativa_sentimiento_hf_colab.ipynb`](notebooks/comparativa_sentimiento_hf_colab.ipynb) en Google Colab.
2. Active GPU T4 y ejecute **Entorno de ejecución > Ejecutar todas**; permita conexiones de red.
3. El notebook instala dependencias, confirma CUDA, ofrece autenticación segura al Hub y obtiene SST-2 `validation`.
4. Descarga secuencialmente los tres modelos, calcula accuracy, precision/recall/F1 macro, métricas positivas, latencias end-to-end (batch 32 y p50/p95 individual), throughput, memoria y matriz de confusión.
5. Exporte `results/comparacion_modelos.csv`, `results/predicciones.csv`, `results/environment.json`, `data/sst2_validation.csv` y manifiesto del dataset. Los resultados medidos ya se incorporaron al paquete de entrega; consulte las consideraciones de redistribución de textos antes de subirlos a GitHub.

## Ejecución por línea de comandos (opcional)
```bash
pip install -r requirements.txt
python src/generate_synthetic.py
python src/export_excel.py
python src/download_dataset.py
python src/evaluate.py --csv data/sst2_validation.csv --out results
python src/select_model.py --file results/comparacion_modelos.csv
python src/verify_results.py  # con el CSV y predicciones presentes
pytest -q tests
```
Se puede ejecutar `src/evaluate.py --csv data/synthetic_smoke_test.csv --allow-cpu` para depuración, **no** para reportar el benchmark. `src/evaluate.py` exige CUDA en modo normal.

## Estructura
```text
notebooks/     cuaderno Colab autocontenido
data/          muestra sintética CSV/Excel + esquema y procedencia
src/           generador, descargador, benchmark y selector
results/       métricas y evidencias de ejecución Colab
docs/          reporte técnico comparativo Word
tests/         controles unitarios offline
.github/        validación continua de esquema y tests
requirements.txt
```

## Resultados reales (ejecución Tesla T4, SST-2 validation, n=872)

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | Latencia p95 individual | VRAM pico | GPU s/1 000 textos |
|---|---:|---:|---:|---:|---:|---:|---:|
| DistilBERT | 91.06 % | 91.13 % | 91.02 % | 91.04 % | 14.10 ms | 322.86 MiB | 1.64 s |
| BERT | 92.43 % | 92.45 % | 92.41 % | 92.43 % | 42.67 ms | 485.87 MiB | 4.32 s |
| RoBERTa | **94.04 %** | **94.04 %** | **94.03 %** | **94.03 %** | 62.47 ms | 553.25 MiB | 2.99 s |

**RoBERTa lidera precisión y F1 macro.** **DistilBERT es el seleccionado para el prototipo** por superar los umbrales de aceptación, registrar menor p95 y VRAM y contar con licencia Apache-2.0 declarada. BERT y RoBERTa no fueron aprobados en el filtro legal automático porque sus checkpoints no declaran licencia; no significa que su uso esté prohibido, sino que debe aclararse.

Las mediciones corresponden a una ejecución única en una GPU de Colab, no una garantía de rendimiento, costos monetarios o robustez fuera de SST-2. El `results/environment.json` incluye las versiones y revisiones exactas.

Verificación independiente de métricas: `python src/verify_results.py`. Validado localmente con **2 616 predicciones** y SHA-256 de dataset.

**Precaución para repositorio público:** el paquete descargable incluye el CSV SST-2 y las predicciones completas para evaluación local. `.gitignore` excluye `data/sst2_validation.csv` y `results/predicciones.csv` de nuevas confirmaciones para no redistribuir textos originales automáticamente; la fuente y scripts permiten regenerarlos.

## Criterios y decisiones
- Un único benchmark: **872** registros SST-2 validados y mismas entradas para todos los checkpoints.
- Inferencia *zero-shot operativa* respecto al experimento: se reutilizan modelos ya entrenados en SST-2, **sin fine-tuning adicional**. Esto **no** implica zero-shot respecto al dataset o tarea original.
- Configuración común: `max_length=128`, truncamiento, lote=32, semilla=42 y CUDA con sincronización para latencia. Distintos tokenizadores originales de cada modelo.
- Elegible: ficha con licencia explícita compatible, F1 macro ≥0.89, accuracy ≥0.89, recall positivo ≥0.88, latencia p95 individual ≤80 ms y VRAM pico ≤4 GiB. Son objetivos **propuestos**, no SLAs validados.
- Desempate: mayor F1 macro; si empatan, menor p95. Si nadie pasa los umbrales: **ningún modelo aprobado**.
- **Decisión posterior al experimento:** RoBERTa obtuvo el F1 mayor, pero DistilBERT es el único candidato aprobado bajo todos los requisitos definidos, incluido el control de licencia; véase el reporte final.

## Limitaciones y riesgos
El corpus de cine en inglés no evalúa la generalización a otros sectores, español, sarcasmo, ironía ni textos extensos. Los modelos TextAttack carecen de licencia declarada en su ficha: no inferir permisos comerciales del modelo base. Colab ofrece GPU variable; medir y registrar hardware real y revisiones. La latencia amortizada por lote no sustituye p95 por solicitud. Revisar derechos de datos y evitar texto real de clientes sin aprobación y anonimización.

## Publicación de GitHub
Este ZIP **no es un enlace a GitHub ni significa que se haya publicado**. Cree un repositorio, suba esta carpeta (sin tokens ni archivos de datos restringidos). Ejemplo:
```bash
git init
git add .
git commit -m "Prototipo comparativo Hugging Face SST-2"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE_REPO.git
git push -u origin main
```

## Referencias
- https://huggingface.co/datasets/nyu-mll/glue
- https://huggingface.co/datasets/stanfordnlp/sst2
- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- https://huggingface.co/textattack/bert-base-uncased-SST-2
- https://huggingface.co/textattack/roberta-base-SST-2
- https://huggingface.co/docs/hub/spaces
- Wang et al. (2019), GLUE; Socher et al. (2013), Stanford Sentiment Treebank.
