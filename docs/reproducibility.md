# Reproducción del modelado y la inferencia

Desde la raíz del repositorio, con Python 3.12 y uv:

```bash
uv sync --frozen
uv run python -m src.fetch_development
uv run pytest -q
uv run python -m src.train_ridge
uv run python -m src.recommend --model models/ridge.joblib --listing examples/listing.json
```

La descarga comprueba el SHA-256 ya registrado y no cambia los manifiestos originales. Por defecto solo descarga junio. La búsqueda utiliza 4 alphas, 5 folds externos y 3 folds internos de calibración. Cada candidato ajusta nuevamente su preprocesamiento dentro de cada entrenamiento. El resultado es un candidato Ridge, no el ganador global.

Las métricas de `src/train_ridge.py` son provisionales: usa funciones de scikit-learn para pinball/MAE y cálculos locales de cobertura/amplitud. El informe y los metadatos del modelo guardan `metrics_status=provisional_pending_shared_metrics` y la implementación utilizada.

`src/metrics.py` ya está integrado en esta rama, y la comparación entre familias se calcula con él desde `src/train_models.py`. Esa es la comparación que declara un ganador. Las cifras provisionales de `train_ridge.py` quedan como evidencia del trabajo específico de Ridge y no se usan para comparar familias. Una prueba contrasta el pinball de `src/metrics.py` contra `sklearn.metrics.mean_pinball_loss`, así que las dos implementaciones coinciden.

`joblib` está declarado como dependencia directa para guardar y cargar modelos. La prueba `tests/test_recommend_integration.py` cubre entrenamiento sintético, serialización y ejecución del recomendador en otro proceso, tanto con el diccionario de modelo/metadatos como con el modelo directo.

Artefactos: `docs/ridge_validation.json` guarda resultados y procedencia; `data/processed/development_folds.csv` guarda posiciones por id/host; `data/processed/ridge_oof.csv` contiene predicciones de validación; `models/ridge.joblib` contiene el modelo y metadatos. Los CSV y modelos siguen ignorados por Git. No confundir `ridge_oof.csv` con test: todas sus filas son de junio.

Sensibilidad opcional, sin sobrescribir el candidato principal:

```bash
uv run python -m src.train_ridge --include-optional
```

Eso genera `ridge_optional.joblib`, `ridge_validation_optional.json` y `ridge_oof_optional.csv`. La especificación principal conserva todos los objetivos válidos sin recortarlos. Si se prueban topes, deben calcularse solo en entrenamiento y deben conservarse los precios de validación originales.

## Notebook de explicación

```bash
uv run python -m ipykernel install --prefix .venv --name airbnb --display-name 'Airbnb Python 3.12'
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=airbnb --ExecutePreprocessor.timeout=180 notebooks/03_ridge_recommender.ipynb
```

Ejecutar `src.train_ridge` antes del notebook. El notebook lee resultados guardados y demuestra inferencia; no repite la búsqueda completa y nunca abre septiembre.

## Paquete de entrega

```bash
uv run python -m src.fetch_development --include-test
uv run python -m src.package_data
```

`output/airbnb_datos_entrega.zip` contiene los dos CSV comprimidos originales y los manifiestos de procedencia/separación. Su preparación solo verifica bytes y hashes; no hace análisis ni evaluación del test. Entregar este archivo junto al repositorio y las diapositivas. El ZIP permanece ignorado por Git. El modelo se reconstruye con el comando de entrenamiento.

## Interfaz de integración

```python
from src.modeling_data import load_development, make_folds
from src.ridge_model import RidgeQuantileModel

X, y, metadata = load_development()
folds = make_folds(metadata)
for train, valid in folds:
    ridge = RidgeQuantileModel(alpha=100)
    ridge.fit(X.iloc[train], y.iloc[train], groups=metadata.host_id.iloc[train])
    predictions = ridge.predict(X.iloc[valid])
    # evaluate_quantile_predictions(y.iloc[valid], predictions), de src.metrics
```

Ridge también acepta `predict_quantiles`. Sin groups, su calibración interna usa KFold por filas para compatibilidad, pero para reproducir nuestros resultados hay que pasar groups. Los folds externos siempre deben ser los de make_folds. No hace falta pasar host_id como predictor.

X incluye las variables opcionales sin imputar para configurar el análisis de sensibilidad. Ridge principal solo selecciona las variables principales; LightGBM debe declarar explícitamente qué columnas usa. `feature_columns(False)` devuelve las listas principales numéricas/categóricas. Si LightGBM quiere categorías nativas, se deben aprender sus categorías solo en entrenamiento; el preprocesador OneHot es una alternativa ya disponible.

`src/train_ridge.py` implementa la búsqueda específica de Ridge: no sustituye `src/train_models.py`, las métricas comunes, LightGBM o la evaluación final del test. Las métricas comunes se mantienen en src.metrics, sin duplicar su implementación.

El recomendador acepta un joblib con el modelo directamente, o un diccionario `{'model': modelo, 'metadata': {...}}`. El modelo debe implementar predict(DataFrame) con las cuatro columnas en MXN y todas sus transformaciones ajustadas. Solo cargar joblib generados por el equipo y confiables.

## Comparación común, congelamiento y evaluación final

```bash
uv run python -m src.train_models --mode both              # comparación, búsqueda y sensibilidades
uv run python -m src.segments                              # tablas por segmento
uv run python -m src.evaluate_final freeze                 # congelar y serializar el ganador
```

El congelamiento registra en `docs/frozen_solution.json` la ruta del artefacto que acaba de generar,
junto con su SHA-256. La recomendación la lee de ahí en vez de escribirla a mano, así que el comando
funciona igual si el ganador resulta ser Ridge, LightGBM o el baseline:

```bash
MODELO=$(uv run python -c "import json; print(json.load(open('docs/frozen_solution.json'))['artifact_path'])")
uv run python -m src.recommend --model "$MODELO" --listing examples/listing.json
```

El nombre del archivo lleva la familia ganadora, pero no hay que adivinarlo: `artifact_path` es la
ruta real del artefacto serializado, y la metadata que viaja dentro dice qué modelo es.

En macOS, LightGBM necesita el runtime de OpenMP: `brew install libomp`. `uv sync` no lo instala porque es dependencia de sistema, y sin él falla el import de `lightgbm`.

`--mode both` compara baseline, Ridge y LightGBM sobre las particiones de `make_folds()`, ejecuta la búsqueda acotada (4 alphas de Ridge, 12 configuraciones de LightGBM) y luego las tres variantes de sensibilidad. El ejecutor consume el `RidgeQuantileModel` de este repositorio; no mantiene una segunda implementación. Cada registro guarda `fold_assignment_sha256`, comparable con el de `docs/ridge_validation.json`.

`evaluate_final freeze` hace tres cosas, en este orden: guarda la configuración ganadora en `docs/frozen_solution.json`, reentrena esa configuración con todo junio y serializa `{'model': ..., 'metadata': ...}` en `models/final_<modelo>.joblib`, en el mismo formato que consume el recomendador. Antes de guardar verifica sobre desarrollo que los cuantiles salgan positivos y ordenados. Con `--skip-artifact` solo guarda la configuración.

La evaluación final es un comando aparte y deliberadamente incómodo:

```bash
uv run python -m src.evaluate_final run --confirm-frozen
```

Se niega a ejecutarse sin `--confirm-frozen`, sin `docs/frozen_solution.json`, si el SHA-256 de junio no coincide con el registrado al congelar, si el contrato de cuantiles cambió, o si ya existe `docs/final_evaluation.json`. Ese último candado existe porque el test temporal es de un solo uso: repetirlo deja de ser una estimación independiente.

El candado no está solo en la CLI. `run_final_evaluation()` exige `confirm_frozen=True` en la propia llamada de Python y comprueba los requisitos del congelamiento antes de abrir ningún archivo, de modo que un script que la invoque sin autorización no alcanza a leer septiembre. Las pruebas lo verifican espiando que no se llame ni a `load_development` ni a `load_test_set`.

El artefacto serializado guarda los parámetros efectivos del modelo ajustado, no solo los que se escribieron al llamarlo: semilla, particiones internas de calibración, umbral de categorías raras, `n_jobs`, los hiperparámetros reales del estimador y las columnas predictoras, con `id` y `host_id` marcados como excluidos. Una prueba comprueba que esa metadata corresponde al modelo que viaja en el mismo archivo.

**Septiembre no ha sido evaluado.** Las pruebas del evaluador usan snapshots sintéticos generados en directorios temporales.

Artefactos de esta parte: `docs/model_comparison.json`, `docs/sensitivity_results.json`, `docs/segment_tables.md` y `docs/development_results.md` quedan en Git. `data/processed/validation_predictions.csv`, `data/processed/segment_tables.csv` y `models/final_*.joblib` quedan fuera, y se reconstruyen con los comandos de arriba.
