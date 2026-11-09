# Reproducción del modelado y la inferencia

Desde la raíz del repositorio, con Python 3.12 y uv, la reproducción ordinaria del desarrollo usa únicamente junio y deja sus resultados regenerables fuera de los archivos oficiales:

```bash
uv sync --frozen
uv run python -m src.fetch_development
repro_dir=$(mktemp -d)
uv run python -m src.train_models --mode both \
  --comparison "$repro_dir/model_comparison.json" \
  --predictions "$repro_dir/validation_predictions.csv" \
  --sensitivities "$repro_dir/sensitivity_results.json"
```

La descarga comprueba el SHA-256 ya registrado y, por defecto, solo verifica o descarga junio. El ejecutor compara baseline, Ridge y LightGBM con las mismas particiones y métricas. No es necesario volver a descargar septiembre ni ejecutar la evaluación temporal para reproducir el desarrollo.

Las métricas de `src/train_ridge.py` son provisionales: usa funciones de scikit-learn para pinball/MAE y cálculos locales de cobertura/amplitud. El informe y los metadatos del modelo guardan `metrics_status=provisional_pending_shared_metrics` y la implementación utilizada.

`src/metrics.py` ya está integrado en esta rama, y la comparación entre familias se calcula con él desde `src/train_models.py`. Esa es la comparación que declara un ganador. Las cifras provisionales de `train_ridge.py` quedan como evidencia del trabajo específico de Ridge y no se usan para comparar familias. Una prueba contrasta el pinball de `src/metrics.py` contra `sklearn.metrics.mean_pinball_loss`, así que las dos implementaciones coinciden.

`joblib` está declarado como dependencia directa para guardar y cargar modelos. La prueba `tests/test_recommend_integration.py` cubre entrenamiento sintético, serialización y ejecución del recomendador en otro proceso, tanto con el diccionario de modelo/metadatos como con el modelo directo.

Artefactos: `docs/ridge_validation.json` guarda resultados y procedencia del candidato Ridge; `data/processed/development_folds.csv` guarda posiciones por id/host; `data/processed/ridge_oof.csv` contiene predicciones de validación; `models/ridge.joblib` contiene el candidato Ridge. La solución final seleccionada queda registrada en `docs/frozen_solution.json`, y su evaluación única en `docs/final_evaluation.json`. Los CSV y modelos siguen ignorados por Git. No confundir los CSV de validación con el test: las predicciones de desarrollo son de junio.

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
# Solo para reconstruir el paquete histórico de entrega; no es parte de la reproducción ordinaria.
uv run python -m src.fetch_development --include-test
uv run python -m src.package_data
```

`output/airbnb_datos_entrega.zip` contiene los dos CSV comprimidos originales y los manifiestos de procedencia/separación. Su preparación solo verifica bytes y hashes; no hace análisis ni evaluación del test. Este paquete corresponde a la entrega histórica de datos y no forma parte de la reproducción ordinaria del modelo final. El ZIP permanece ignorado por Git.

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

La comparación y el congelamiento ya se realizaron. La selección se hizo únicamente con junio y el
modelo congelado fue LightGBM con `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31` y
`min_child_samples=50`.

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

`evaluate_final freeze` hizo tres cosas, en este orden: guardó la configuración ganadora en `docs/frozen_solution.json`, reentrenó esa configuración con todo junio y serializó `{'model': ..., 'metadata': ...}` en `models/final_<modelo>.joblib`, en el mismo formato que consume el recomendador. Antes de guardar verificó sobre desarrollo que los cuantiles salieran positivos y ordenados. La evaluación final no depende de volver a ejecutar este comando.

La evaluación final fue un comando aparte y deliberadamente incómodo:

```bash
uv run python -m src.evaluate_final run --confirm-frozen
```

Ese comando ya se ejecutó una sola vez y no debe volver a ejecutarse. La CLI se niega a continuar sin `--confirm-frozen`, sin `docs/frozen_solution.json`, si el SHA-256 de junio no coincide con el registrado al congelar, si el contrato de cuantiles cambió, o si ya existe `docs/final_evaluation.json`, salvo `--allow-rerun`, que no forma parte del procedimiento permitido.

El candado no está solo en la CLI. `run_final_evaluation()` exige `confirm_frozen=True` en la propia llamada de Python y comprueba los requisitos del congelamiento antes de abrir ningún archivo, de modo que un script que la invoque sin autorización no alcanza a leer septiembre. Las pruebas lo verifican espiando que no se llame ni a `load_development` ni a `load_test_set`.

El artefacto serializado guarda los parámetros efectivos del modelo ajustado, no solo los que se escribieron al llamarlo: semilla, particiones internas de calibración, umbral de categorías raras, `n_jobs`, los hiperparámetros reales del estimador y las columnas predictoras, con `id` y `host_id` marcados como excluidos. Una prueba comprueba que esa metadata corresponde al modelo que viaja en el mismo archivo.

**La evaluación de septiembre ya se realizó una sola vez.** Los resultados oficiales están en `docs/final_evaluation.json`. Las pruebas del evaluador usan snapshots sintéticos generados en directorios temporales y no sustituyen esa evaluación.

Artefactos de esta parte: `docs/model_comparison.json`, `docs/sensitivity_results.json`, `docs/segment_tables.md`, `docs/development_results.md`, `docs/frozen_solution.json` y `docs/final_evaluation.json` quedan en Git. `data/processed/validation_predictions.csv`, `data/processed/segment_tables.csv` y `models/final_*.joblib` quedan fuera por las reglas `.gitignore` `data/processed/*`, `models/*` y `*.joblib`; son regenerables.

## Resultados temporales finales

La evaluación se hizo sobre 2,604 anuncios con `id` ausente en junio y precio válido. Esta definición identifica anuncios nuevos respecto del snapshot, no anfitriones completamente nuevos: 1,493 anuncios pertenecían a anfitriones que ya tenían al menos un anuncio en junio. No se debe interpretar que esos anuncios fueron necesariamente publicados por primera vez en septiembre.

| Métrica | Resultado |
|---|---:|
| Pinball p35 | 327.8444 |
| MAE p35 | 786.0543 MXN |
| Cobertura p10–p90 | 79.6467 % |
| Amplitud media p10–p90 | 3,364.2481 MXN |
| Cruce de cuantiles después del ordenamiento | 0 % |

La cobertura global quedó ligeramente por debajo del 80 % nominal. Por segmentos, las habitaciones
privadas tuvieron 76.0793 % de cobertura (`n=857`) y Q1, el cuartil de precios más bajos, tuvo
64.6697 % (`n=651`). Q2 tuvo 80.0307 %, Q3 92.3547 % y Q4 81.4815 %. El segmento de hotel tuvo
un solo anuncio y no permite una conclusión estable.

La recomendación se ejecuta con el artefacto cuya ruta quedó registrada en el JSON congelado:

```bash
MODELO=$(uv run python -c "import json; print(json.load(open('docs/frozen_solution.json'))['artifact_path'])")
uv run python -m src.recommend --model "$MODELO" --listing examples/listing.json
```

Si el `.joblib` original no está disponible, generar una copia fuera del repositorio para no
sobrescribir los JSON oficiales:

```bash
repro_dir=$(mktemp -d)
uv run python -m src.evaluate_final freeze \
  --comparison docs/model_comparison.json \
  --output "$repro_dir/frozen_solution.json" \
  --june data/raw/mexico_city_2026-06-15_listings.csv.gz \
  --artifact "$repro_dir/final_lightgbm.joblib"
uv run python -m src.recommend \
  --model "$repro_dir/final_lightgbm.joblib" \
  --listing examples/listing.json
```

La copia nueva se entrena solo con junio y puede tener un SHA-256 distinto por diferencias de
entorno o serialización. Reproducir el entrenamiento no significa repetir la evaluación temporal.
