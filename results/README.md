# Resultados medidos — Google Colab GPU

Se ejecutó el benchmark el **12 de octubre de 2026 a las 01:35 UTC** (instante registrado en `environment.json`; corresponde al 11 de octubre en México, zona UTC-6), sobre **Tesla T4**, 872 textos de SST-2 validation. Los tres checkpoints fueron evaluados secuencialmente y **sin fine-tuning adicional**.

Los resultados auditados se hallan en `comparacion_modelos.csv`. Las matrices de confusión, métricas positivas y revisiones SHA también están allí. `metricas.png` y `tradeoff.png` muestran la comparación gráfica. `environment.json` conserva parámetros, librerías, GPU y revisiones.

- **Mejor F1 macro:** RoBERTa: 94.03 %; p95 62.47 ms.
- **Menor p95:** DistilBERT: 14.10 ms; F1 macro 91.04 %.
- **Recomendación con los filtros del proyecto:** DistilBERT, por precisión suficiente, menor latencia, menor VRAM y licencia Apache-2.0 documentada. Las fichas TextAttack no declaran licencia; por tanto requieren revisión antes del despliegue.

**Interpretación:** latencia p95 = inferencia individual en subconjunto de hasta 80 textos, luego de calentamiento; throughput y GPU-segundos/1 000 textos = medición por lotes sobre 872 ejemplos. No son SLA, costos monetarios ni tiempos de descarga de modelos. Todos los valores son de una sola sesión de Colab.

**Verificación local (con `data/sst2_validation.csv` y `results/predicciones.csv` presentes):**
```bash
python src/verify_results.py
python src/select_model.py --file results/comparacion_modelos.csv
```

**Publicación:** `data/sst2_validation.csv` y `results/predicciones.csv` contienen textos originales de SST-2. Aunque se incluyen en el paquete de entrega para auditoría local, se excluyen mediante `.gitignore` de un repositorio público hasta comprobar los derechos de redistribución del corpus. El README documenta cómo regenerarlos.
