# Verificación del modelado y la inferencia

Fecha: 8 de noviembre de 2026.

- Entorno instalado desde uv.lock con Python 3.12; uv lock --check --offline correcto.
- 18 pruebas pasan, incluyendo las 8 pruebas originales de preparación y 10 de carga, preprocesamiento, Ridge, búsqueda e inferencia. La integración entrena con datos sintéticos, carga el artefacto guardado y ejecuta el recomendador por CLI en un proceso separado; cubre el modelo directo y el diccionario con metadatos.
- Notebook 03_ridge_recommender: 5 celdas de código ejecutadas, sin errores guardados.
- Hashes originales de junio y septiembre coinciden con docs/data_manifest.json. Septiembre solo se descargó y empaquetó; no se leyeron sus filas para análisis o modelado.
- Se entrenó Ridge con 29,637 anuncios y 13,095 anfitriones de junio. Alpha seleccionado: 100 entre 0.1, 1, 10 y 100.
- Promedios de cinco folds: pinball p35 706.94 MXN; MAE p35 1,722.08 MXN; MAE p50 1,714.44 MXN; cobertura 80.20%; amplitud 3,661.32 MXN. Son resultados provisionales de selección en desarrollo, pendientes de integrar las métricas comunes. La regeneración conserva esos valores y añade su procedencia mediante `metrics_status` y `metrics_implementation` al informe y al modelo.
- Compatibilidad comprobada con predict del baseline y evaluate_quantile_predictions de la rama origin/feat/metrics-and-baseline (d17e9cc). El recomendador también aceptó el baseline. La comprobación de interfaz no modificó los archivos de esa rama.
- En la verificación inicial se reprodujo la rama desde un clon local limpio, reutilizando el entorno instalado y copiando solamente el snapshot de junio: las 16 pruebas entonces disponibles pasaron; entrenamiento, JSON de resultados y recomendación idénticos. El clon de verificación no contenía septiembre.
- `joblib` se declara como dependencia directa en pyproject.toml y uv.lock. La actualización mantiene las versiones ya fijadas de los paquetes.
- git diff --check correcto. Los artefactos originales de preparación de datos permanecen intactos.

Los scripts y notebook están versionados. Modelos, CSV y ZIP de entrega quedan fuera de Git y se reconstruyen con docs/reproducibility.md. El modelo final global y la evaluación del test permanecen pendientes.
