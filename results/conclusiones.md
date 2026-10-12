# Conclusiones experimentales definitivas

Comparación de 3 modelos de sentimiento SST-2 (872 ejemplos), Colab Tesla T4, una sesión.

| Modelo | Accuracy | F1 macro | p95 individual | VRAM pico | GPU s/1000 |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 91.06% | 91.04% | 14.10 ms | 322.9 MiB | 1.64 s |
| BERT | 92.43% | 92.43% | 42.67 ms | 485.9 MiB | 4.32 s |
| RoBERTa | 94.04% | 94.03% | 62.47 ms | 553.2 MiB | 2.99 s |

RoBERTa logra la mejor clasificación (F1 macro 94.03%), mientras DistilBERT presenta latencia p95 más baja (14.10 ms) y menor consumo GPU.
**Selección condicionada: DistilBERT.** Satisface umbrales del proyecto y declara licencia Apache-2.0 en la ficha Hugging Face. BERT/RoBERTa (checkpoints TextAttack) requieren aclaración de licencia antes de su uso en producción; la omisión no demuestra prohibición de uso.
La diferencia RoBERTa–DistilBERT en F1 es de 2.99 puntos porcentuales, pero el p95 de RoBERTa es 4.43 veces el de DistilBERT. Esto se limita a un benchmark de un dominio (reseñas en inglés) y una sesión de GPU.
La memoria pico se registró como memoria asignada máxima por PyTorch en CUDA, no necesariamente VRAM total de la GPU.
GPU s/1000 es una extrapolación de tiempo por lote, NO un costo monetario ni SLA.
Se verificaron de manera independiente 2 616 predicciones (872 por modelo), métricas, matrices y hash del CSV.
Los resultados no indican rendimiento en español ni en contextos especializados y no se hizo fine-tuning adicional.

Archivos fuente: comparacion_modelos.csv, predicciones.csv, environment.json, tradeoff.png y metricas.png.
