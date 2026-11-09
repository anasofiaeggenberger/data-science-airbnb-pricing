# Verificación del modelado y la inferencia

Fecha: 8 de noviembre de 2026.

> **Nota de trazabilidad.** Las verificaciones de este documento fueron registradas en cortes de
> desarrollo anteriores al cierre experimental. Conservan evidencia histórica y no deben confundirse
> con el estado final: la solución quedó congelada después y la evaluación temporal se ejecutó una sola
> vez. Los archivos de referencia actuales son [`docs/frozen_solution.json`](frozen_solution.json) y
> [`docs/final_evaluation.json`](final_evaluation.json).

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

Los scripts y notebook están versionados. Modelos, CSV y ZIP de entrega quedan fuera de Git y se reconstruyen con docs/reproducibility.md.

## Verificación de la integración de experimentos

Fecha: 8 de noviembre de 2026. Rama `feat/lightgbm-experiments`, después de fusionar `main` con los PR #2 y #3 ya integrados.

- La fusión de `origin/main` no produjo conflictos. Se conservan las correcciones posteriores de Víctor: `joblib` declarado en pyproject.toml y presente en uv.lock, `tests/test_recommend_integration.py`, el README actualizado y el etiquetado de las métricas de Ridge como provisionales.
- 84 pruebas pasan, incluidas las de Víctor sin modificar. `uv lock --check` y `git diff --check` correctos.
- Las métricas comparativas salen únicamente de `src/metrics.py`. Una prueba contrasta su pinball contra `sklearn.metrics.mean_pinball_loss` en los cuatro cuantiles. `src/train_ridge.py` mantiene sus cálculos locales, etiquetados como provisionales, y no se usa para comparar familias.
- Baseline, Ridge y LightGBM se evalúan sobre las mismas cinco particiones de `make_folds()`. La corrida registró `fold_assignment_sha256` igual a `0bc17e53368075dfcc00702ad0e9b2061e04d670b93fb1004b062b7eb5a27508`, el mismo valor de docs/ridge_validation.json.
- La comparación regenerada tras la fusión reprodujo cifras idénticas a la corrida anterior, hasta el sexto decimal. Las correcciones de Víctor no tocaron `modeling_data.py`, `preprocessing.py` ni `ridge_model.py`.
- Selección por pinball p35; MAE y cobertura se reportan como secundarias. Hiperparámetros elegidos solo con junio: 4 alphas de Ridge y 12 configuraciones de LightGBM.
- Cruces de cuantiles de LightGBM reportados por separado: 4.98 % antes de ordenar, 0 % después.
- Serialización del ganador verificada sobre junio real en rutas temporales: entrenó con 29,637 filas, guardó el artefacto joblib con su metadata, y `src/recommend.py` lo cargó y devolvió los cuatro cuantiles en MXN, positivos y ordenados. También se probó con Ridge y LightGBM sobre datos sintéticos.
- En ese corte, el evaluador final se probó únicamente con snapshots sintéticos en directorios temporales.
  **La evaluación real todavía no se había ejecutado**, no existían los JSON oficiales de cierre y el
  snapshot de septiembre aún no formaba parte de esa verificación.
- En ese corte la solución todavía no estaba congelada; después se congeló LightGBM y se ejecutó una
  sola evaluación temporal. El detalle actual está en los dos JSON oficiales indicados arriba.

El test temporal de septiembre permanecía sin evaluar en el corte histórico de este documento. No se
debe repetir ahora: la evaluación oficial ya está registrada en `docs/final_evaluation.json`.
