# Reproducir la parte de Víctor

Desde la raíz del repositorio, con Python 3.12 y uv:

```bash
uv sync --frozen
uv run python -m src.fetch_development
uv run pytest -q
uv run python -m src.train_ridge
uv run python -m src.recommend --model models/ridge.joblib --listing examples/listing.json
```

La descarga comprueba el SHA-256 ya registrado y no cambia los manifiestos de Sofi. Por defecto solo descarga junio. La búsqueda utiliza 4 alphas, 5 folds externos y 3 folds internos de calibración. Cada candidato ajusta nuevamente su preprocesamiento dentro de cada entrenamiento. El resultado es un candidato Ridge, no el ganador global.

Artefactos: `docs/ridge_validation.json` guarda resultados y procedencia; `data/processed/development_folds.csv` guarda posiciones por id/host; `data/processed/ridge_oof.csv` contiene predicciones de validación; `models/ridge.joblib` contiene el modelo y metadatos. Los CSV y modelos siguen ignorados por Git. No confundir `ridge_oof.csv` con test: todas sus filas son de junio.

Sensibilidad opcional, sin sobrescribir el candidato principal:

```bash
uv run python -m src.train_ridge --include-optional
```

Eso genera `ridge_optional.joblib`, `ridge_validation_optional.json` y `ridge_oof_optional.csv`. La especificación principal conserva todos los objetivos válidos sin recortarlos. Si Pan prueba topes, debe calcularlos solo en entrenamiento y mantener los precios de validación originales.

## Notebook de explicación

```bash
uv run python -m ipykernel install --prefix .venv --name airbnb --display-name 'Airbnb Python 3.12'
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=airbnb --ExecutePreprocessor.timeout=180 notebooks/03_ridge_victor.ipynb
```

Ejecutar `src.train_ridge` antes del notebook. El notebook lee resultados guardados y demuestra inferencia; no repite la búsqueda completa y nunca abre septiembre.

## Paquete de entrega

```bash
uv run python -m src.fetch_development --include-test
uv run python -m src.package_data
```

`output/airbnb_datos_entrega.zip` contiene los dos CSV comprimidos originales y los manifiestos de procedencia/separación. Su preparación solo verifica bytes y hashes; no hace análisis ni evaluación del test. Entregar este archivo junto al repositorio y las diapositivas. El ZIP permanece ignorado por Git. El modelo se reconstruye con el comando de entrenamiento.

## Integración con Pan

```python
from src.modeling_data import load_development, make_folds
from src.ridge_model import RidgeQuantileModel

X, y, metadata = load_development()
folds = make_folds(metadata)
for train, valid in folds:
    ridge = RidgeQuantileModel(alpha=100)
    ridge.fit(X.iloc[train], y.iloc[train], groups=metadata.host_id.iloc[train])
    predictions = ridge.predict(X.iloc[valid])
    # evaluate_quantile_predictions(y.iloc[valid], predictions), de src.metrics de Pan
```

Ridge también acepta `predict_quantiles`. Sin groups, su calibración interna usa KFold por filas para compatibilidad, pero para reproducir nuestros resultados hay que pasar groups. Los folds externos siempre deben ser los de make_folds. No hace falta pasar host_id como predictor.

X incluye las variables opcionales sin imputar para que Pan pueda configurar su sensibilidad. Ridge principal solo selecciona las variables principales; LightGBM debe declarar explícitamente qué columnas usa. `feature_columns(False)` devuelve las listas principales numéricas/categóricas. Si LightGBM quiere categorías nativas, Pan debe aprender sus categorías solo en entrenamiento; el preprocesador OneHot es una alternativa ya disponible.

`src/train_ridge.py` es la búsqueda específica de Víctor: no sustituye `src/train_models.py`, las métricas comunes, LightGBM o el test final de Pan. No importar un segundo metrics.py ni reemplazar el suyo.

El recomendador acepta un joblib con el modelo directamente, o un diccionario `{'model': modelo, 'metadata': {...}}`. El modelo debe implementar predict(DataFrame) con las cuatro columnas en MXN y todas sus transformaciones ajustadas. Conservar su estado de candidato hasta la selección común. Solo cargar joblib generados por el equipo y confiables.
