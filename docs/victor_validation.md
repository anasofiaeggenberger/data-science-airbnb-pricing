# Verificación de la entrega de Víctor

Fecha: 8 de noviembre de 2026.

- Entorno instalado desde uv.lock con Python 3.12; uv lock --check --offline correcto.
- 16 pruebas pasan, incluyendo las 8 originales de Sofi y 8 de carga, preprocesamiento, Ridge, búsqueda e inferencia.
- Notebook 03_ridge_victor: 5 celdas de código ejecutadas, sin errores guardados.
- Hashes originales de junio y septiembre coinciden con docs/data_manifest.json. Septiembre solo se descargó y empaquetó; no se leyeron sus filas para análisis o modelado.
- Se entrenó Ridge con 29,637 anuncios y 13,095 anfitriones de junio. Alpha seleccionado: 100 entre 0.1, 1, 10 y 100.
- Promedios de cinco folds: pinball p35 706.94 MXN; MAE p35 1,722.08 MXN; MAE p50 1,714.44 MXN; cobertura 80.20%; amplitud 3,661.32 MXN. Son resultados de selección en desarrollo.
- Compatibilidad comprobada con predict del baseline y evaluate_quantile_predictions de Pan en origin/feat/metrics-and-baseline (d17e9cc). El recomendador también aceptó el baseline de Pan. No se integraron ni reemplazaron sus archivos.
- Reproducción desde un clon local limpio de la rama, reutilizando el entorno instalado y copiando solamente el snapshot de junio: 16 pruebas pasan; entrenamiento, JSON de resultados y recomendación idénticos. El clon de verificación no contenía septiembre.
- git diff --check correcto. No se cambiaron dependencias ni los artefactos originales de preparación de Sofi.

Los scripts y notebook están versionados. Modelos, CSV y ZIP de entrega quedan fuera de Git y se reconstruyen con docs/reproducibility.md. El modelo final global y la evaluación del test siguen pendientes del trabajo de Pan.
