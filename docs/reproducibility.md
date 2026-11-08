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

Las métricas de `src/train_ridge.py` son provisionales: usa funciones de scikit-learn para pinball/MAE y cálculos locales de cobertura/amplitud. `src/metrics.py` está pendiente de integración desde la rama de métricas y baseline. El informe y los metadatos del modelo guardan `metrics_status=provisional_pending_shared_metrics` y la implementación utilizada. Tras integrar las métricas comunes, se deben conectar a `evaluate_quantile_predictions` y regenerar los resultados de desarrollo antes de compararlos entre familias.

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

El recomendador acepta un joblib con el modelo directamente, o un diccionario `{'model': modelo, 'metadata': {...}}`. El modelo debe implementar predict(DataFrame) con las cuatro columnas en MXN y todas sus transformaciones ajustadas. Conservar su estado de candidato hasta la selección común. Solo cargar joblib generados por el equipo y confiables.
