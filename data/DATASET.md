# Datos y procedencia

## Datos experimentales principales (incluidos en el paquete local; excluidos de nuevos commits públicos)
- Identificador: `nyu-mll/glue`, configuración `sst2`, división `validation`.
- Fuente: https://huggingface.co/datasets/nyu-mll/glue ; documentación de SST-2: https://huggingface.co/datasets/stanfordnlp/sst2
- Filas esperadas: **872**. Las etiquetas de la división `test` son ocultas: no se usa para evaluación.
- `python src/download_dataset.py` descarga los 872 ejemplos y crea `data/sst2_validation.csv` y un manifiesto SHA-256 con revisión.
- Se incorpora al ZIP de entrega el CSV exportado de la ejecución Colab para auditoría local. Su publicación abierta requiere comprobar derechos y condiciones del corpus fuente (el Hub categoriza la licencia GLUE como «other»). Git ignora el archivo por defecto.

## Datos de prueba locales (incluidos)
- `synthetic_smoke_test.csv`: **80** reseñas cortas redactadas manualmente para probar lectura, esquema y pipeline sin usar SST-2.
- Origen: contenido sintético original de este proyecto; no son observaciones reales ni un test independiente de validez externa.
- Reproducir: `python src/generate_synthetic.py --out data/synthetic_smoke_test.csv`.
- **Nunca use las métricas del conjunto sintético para seleccionar el mejor modelo**: muestras no representativas.

## Estructura CSV
| Campo | Tipo | Descripción |
|---|---|---|
| `idx` | string/integer | Identificador único del registro. |
| `sentence` | string | Texto de opinión en inglés. |
| `label` | integer | Etiqueta real: 0 negativo, 1 positivo. |
| `source` | string | Procedencia del texto. |
| `split` | string | División (`validation` o `smoke_test`). |

El cuaderno utiliza SST-2 `validation` para las cifras reportables. Todos los modelos reciben el mismo archivo, texto, batch_size y longitud máxima. Controlar el sesgo por dominio: SST-2 proviene de reseñas cinematográficas y no demuestra rendimiento en banca, idioma español ni atención a clientes.
