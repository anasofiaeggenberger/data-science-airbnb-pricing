# data-science-airbnb-pricing

Proyecto final de Ciencia de Datos — segundo semestre 2026, UFM.

Pregunta: ¿qué precio por noche puede usar como referencia un anfitrión nuevo de Airbnb en Ciudad de México? El objetivo es recomendar un precio en MXN y un rango razonable usando anuncios reales de Inside Airbnb. La propuesta contempla un baseline de mediana por alcaldía y tipo de alojamiento, regresión lineal regularizada sobre `log(precio)` y LightGBM con pérdida por cuantiles. La métrica principal es pinball loss, además de MAE y cobertura del intervalo.

El repositorio incluye adquisición, auditoría, preparación de datos, EDA, las métricas comunes, el baseline de cuantiles, Ridge con calibración de residuos, LightGBM cuantílico, el ejecutor que compara los tres sobre las mismas particiones, las sensibilidades acordadas y un recomendador que carga el modelo guardado. Las métricas locales de `src/train_ridge.py` siguen siendo provisionales y están etiquetadas como tales; la comparación común se calcula con `src/metrics.py`. La validación de junio seleccionó LightGBM y la evaluación temporal de septiembre se ejecutó una sola vez después del congelamiento; sus resultados oficiales están en `docs/final_evaluation.json` y no deben recalcularse.

## Fuente y snapshots

La fuente obligatoria es [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/). Se usará el archivo detallado `listings.csv.gz`, no el resumen `listings.csv`. La adquisición descubre los enlaces de Mexico City desde la página oficial y selecciona los dos snapshots cronológicos más recientes; si la página expone menos enlaces, reutiliza únicamente URLs ya registradas en el manifiesto. No construye URLs de archivo por suposición.

Al preparar esta fase se verificó el enlace actual publicado para el snapshot `23-09-2026`:

`https://data.insideairbnb.com/mexico/df/mexico-city/2026-09-23/data/listings.csv.gz`

El URL exacto del snapshot anterior `15-06-2026` se verificó con HTTP 200 y tipo `application/x-gzip`. Ambos archivos se descargaron en `data/raw` sin modificar sus bytes; tamaños y SHA-256 quedaron registrados en `docs/data_manifest.json`.

## Estructura

```text
data/raw/          snapshots originales descargados; ignorados por Git
data/processed/    datos derivados; ignorados por Git
docs/              auditoría y manifiesto de procedencia
models/            modelos; ignorados por Git
notebooks/         notebooks reproducibles
presentation/      materiales de presentación
reports/figures/   figuras generadas
src/               adquisición, preparación, entrenamiento e inferencia
tests/             pruebas pequeñas de funciones críticas
```

## Instalación con uv

Se requiere Python 3.12.

```bash
uv sync
```

Esto crea/usa `.venv` y respeta `uv.lock`. Si Python 3.12 no está instalado localmente, instalarlo con el mecanismo de Python disponible en el equipo y repetir `uv sync`.

## Descarga y auditoría

Para la reproducción ordinaria del desarrollo, sin volver a ejecutar la evaluación temporal:

```bash
uv run python -m src.fetch_development
```

`src.fetch_development` verifica y, si hace falta, descarga únicamente el snapshot de junio registrado en `docs/data_manifest.json`. La adquisición histórica de ambos snapshots se hizo con `src.download_data`; no es necesario repetirla para reproducir el entrenamiento ni la inferencia. La auditoría sobre ambos archivos solo debe repetirse si se está revisando la procedencia, no como parte de la evaluación final.

Para una adquisición controlada de snapshots concretos también existen los enlaces exactos:

```bash
uv run python -m src.download_data --url URL_SNAPSHOT_1 --url URL_SNAPSHOT_2
```

La auditoría verifica columnas, filas, `id`, `price`, marcadores de moneda e intersección de IDs. El [diccionario oficial de Inside Airbnb](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing) documenta `price` como precio diario en moneda local y el `$` como artefacto técnico del export; para Mexico City esto sustenta reportar MXN, aunque el CSV no contiene una columna explícita `currency`.

## Preparación y EDA de Fase 2

La preparación histórica de datos se ejecuta con:

```bash
uv run python -m src.preparation
```

Este comando usa junio para construir `data/processed/development_clean.csv`, `docs/cleaning_log.json` y `docs/feature_policy.md`, y lee el snapshot posterior únicamente para actualizar los conteos del manifiesto temporal `docs/split_manifest.json`; no guarda filas ni IDs de septiembre. Como la evaluación final ya se realizó, este comando no forma parte de la reproducción ordinaria del entrenamiento y no debe usarse para repetir la evaluación temporal.

`notebooks/01_eda.ipynb` carga únicamente el snapshot de junio, vuelve a ejecutar la preparación de desarrollo y guarda las figuras generadas en `reports/figures/`. Para reproducir el notebook completo:

```bash
uv run jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3 \
  notebooks/01_eda.ipynb
```

La notebook documenta el problema, los costos de sobrevalorar y subvalorar, la métrica futura, el diseño temporal, la limpieza y el EDA exclusivamente de junio. No entrena modelos ni calcula métricas del test.

La política de variables está en [`docs/feature_policy.md`](docs/feature_policy.md). Reseñas, ratings, disponibilidad observada, ingresos estimados, derivados del precio, identificadores y otras métricas posteriores a publicar quedan fuera. “Nuevo en el snapshot” no necesariamente significa “anuncio recién publicado”.

El caso de uso está pensado para un anfitrión nuevo que necesita fijar el precio inicial de un anuncio. Esa intención no debe confundirse con la población del test temporal: el test se definió por anuncios cuyos `id` estaban ausentes en junio, no por anfitriones sin historial. De los 2,604 anuncios evaluados, 1,493 pertenecían a anfitriones que ya tenían al menos un anuncio en junio. Por eso no se afirma que fueran anuncios publicados por primera vez en septiembre ni que representen exclusivamente anfitriones completamente nuevos.

Para reconstruir el desarrollo desde un clon limpio, verificar junio y guardar los resultados regenerables fuera del repositorio:

```bash
uv sync --frozen
uv run python -m src.fetch_development
repro_dir=$(mktemp -d)
uv run python -m src.train_models --mode both \
  --comparison "$repro_dir/model_comparison.json" \
  --predictions "$repro_dir/validation_predictions.csv" \
  --sensitivities "$repro_dir/sensitivity_results.json"
```

La comparación anterior usa únicamente junio y escribe sus resultados temporales fuera del repositorio. El notebook de EDA también trabaja únicamente con junio. No se debe agregar `--include-test`, ejecutar `src.evaluate_final run` ni usar `--allow-rerun` como parte de esta reproducción ordinaria.

## Regla para validación interna

Antes de cualquier validación cruzada o partición interna de junio, se debe separar el fold y ajustar dentro del fold de entrenamiento todas las reglas que aprendan de datos: imputaciones, agrupación de categorías raras, escalado y umbrales de extremos. Después se aplican esas reglas al fold de validación. Las estadísticas descriptivas de `docs/cleaning_log.json` y de la notebook son solo EDA. El precio objetivo de validación debe conservarse en su escala original para calcular pinball loss, MAE y cobertura. El snapshot de septiembre no participa en ninguna de estas decisiones.

## Reglas de disponibilidad y fuga

El caso de uso es un anfitrión nuevo. No deben usarse como predictores al publicar: reseñas, calificaciones, número de reseñas, superhost, tasas de respuesta/aceptación/cancelación ni cualquier métrica que solo exista después de recibir reservas o huéspedes. También se revisarán como posibles fugas `availability_30/60/90/365`, `availability_eoy`, `calendar_updated`, `calendar_last_scraped`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `price_quote_*`, variables de reviews, `calculated_host_listings_count*` y metadatos del scrape (`scrape_id`, `last_scraped`, `source`). La evaluación temporal no prueba el desempeño exclusivo para anfitriones completamente nuevos, porque su regla identifica anuncios ausentes en junio y no filtra por historial de `host_id`.

## Pruebas

```bash
uv run pytest
uv lock --check
git diff --check
```

Las pruebas cubren adquisición y limpieza, separación por anfitrión, preprocesamiento sin fugas, cuantiles de Ridge, métricas comunes contrastadas con valores calculados a mano, baseline con respaldos, LightGBM con categorías desconocidas y cruces, el ejecutor de experimentos, las tablas por segmento, la serialización del modelo ganador y los candados del evaluador final. Las pruebas de integración entrenan con datos sintéticos, guardan el modelo y ejecutan el recomendador desde la línea de comandos con el artefacto guardado, sin usar los snapshots reales.

En macOS, LightGBM necesita el runtime de OpenMP: `brew install libomp`. `uv sync` no lo instala porque es dependencia de sistema, y sin él las pruebas de LightGBM fallan al importar.

## Flujo colaborativo

1. Crear una rama descriptiva desde `main`, por ejemplo `feat/data-preparation-eda`.
2. Trabajar en cambios pequeños, ejecutar las pruebas y validaciones (`pytest`, `uv lock --check`, `git diff --check`) y documentar comandos/resultados.
3. Abrir un pull request hacia `main` con descripción, evidencia de auditoría, figuras y limitaciones.
4. Revisar al menos por otra persona del equipo; resolver comentarios antes de integrar.
5. No subir datos crudos, modelos ni figuras generadas: el código, la documentación y `uv.lock` sí se conservan en Git.

## Ridge y recomendador

La base para modelar, las particiones por anfitrión, Ridge con calibración de residuos y el recomendador están implementados. Véanse [contrato común](docs/modeling_contract.md), [comandos reproducibles](docs/reproducibility.md) y [notebook de Ridge](notebooks/03_ridge_recommender.ipynb).

La validación de Ridge utiliza exclusivamente junio. `src/train_ridge.py` calcula métricas locales provisionales; su procedencia y estado quedan registrados como `metrics_implementation` y `metrics_status` en el modelo y el informe. Esa búsqueda específica de Ridge no declara un ganador entre familias ni evalúa septiembre. Sus resultados están en `docs/ridge_validation.json`; la bitácora está en `docs/ai/modeling_log.md`.

La comparación entre familias sí declara un ganador, y se calcula aparte con `src/metrics.py` desde `src/train_models.py`.

## Comparación de modelos, congelamiento y evaluación final

La comparación común, el congelamiento y la evaluación temporal ya se completaron en ese orden.
Para regenerar resultados de desarrollo sin sobrescribir los oficiales, usar las rutas temporales
documentadas en [reproducibilidad](docs/reproducibility.md). No volver a ejecutar la evaluación
temporal.

El congelamiento registra en `docs/frozen_solution.json` la ruta del artefacto que acaba de generar,
junto con su SHA-256. La recomendación la lee de ahí en vez de escribirla a mano, así que el comando
funciona igual si el ganador resulta ser Ridge, LightGBM o el baseline:

```bash
MODELO=$(uv run python -c "import json; print(json.load(open('docs/frozen_solution.json'))['artifact_path'])")
uv run python -m src.recommend --model "$MODELO" --listing examples/listing.json
```

El nombre del archivo lleva la familia ganadora, pero no hay que adivinarlo: `artifact_path` es la
ruta real del artefacto serializado, y la metadata que viaja dentro dice qué modelo es.

El ejecutor compara baseline, Ridge y LightGBM sobre las mismas cinco particiones agrupadas por anfitrión y guarda `fold_assignment_sha256` en cada registro, para poder verificar que todos midieron sobre las mismas filas. La selección es por pinball loss del cuantil recomendado; MAE y cobertura son secundarias.

Resultados y limitaciones en [resultados de desarrollo](docs/development_results.md); el procedimiento, en [protocolo de experimentos](docs/experiment_protocol.md). Las reglas de cada modelo están en [contrato de métricas](docs/metrics_contract.md), [baseline](docs/baseline_rules.md) y [LightGBM](docs/lightgbm_rules.md).

**Esos números son validación de desarrollo usada para seleccionar configuración, no el test final.** Están optimistamente sesgados a favor del ganador, porque los hiperparámetros se eligieron mirando esas mismas particiones. La estimación independiente es el test temporal, una sola vez, después del congelamiento.

El congelamiento guardó la configuración en `docs/frozen_solution.json`, reentrenó a LightGBM con todo junio y registró el artefacto oficial `models/final_lightgbm.joblib` mediante su ruta y SHA-256. El archivo `.joblib` está excluido de Git. La evaluación final se ejecutó una sola vez con la solución congelada y quedó registrada en `docs/final_evaluation.json`; no debe volver a ejecutarse.

## Resultados finales

La selección se hizo exclusivamente con las cinco particiones de desarrollo de junio, antes de mirar septiembre. El modelo seleccionado fue LightGBM con `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31` y `min_child_samples=50`.

La evaluación temporal utilizó 2,604 anuncios con `id` ausente en junio y precio válido. Esta es una población de anuncios nuevos respecto del snapshot, no una población de anfitriones nuevos: 1,493 anuncios correspondieron a anfitriones que ya tenían al menos un anuncio en junio.

| Métrica | Resultado |
|---|---:|
| Pinball p35 | 327.8444 |
| MAE p35 | 786.0543 MXN |
| Cobertura p10–p90 | 79.6467 % |
| Amplitud media p10–p90 | 3,364.2481 MXN |
| Cruce de cuantiles después del ordenamiento | 0 % |

La cobertura global quedó ligeramente por debajo del 80 % nominal. También hubo diferencias entre segmentos: habitaciones privadas alcanzaron 76.0793 % de cobertura con 857 anuncios, mientras que el cuartil de precios más bajos (Q1) alcanzó 64.6697 % con 651 anuncios. Q2 quedó en 80.0307 %, Q3 en 92.3547 % y Q4 en 81.4815 %. Estos resultados describen este snapshot y no garantizan el comportamiento futuro; el segmento de habitaciones de hotel tuvo solo un anuncio y no permite una conclusión confiable.

El artefacto original y las predicciones CSV no están versionados: `.gitignore` excluye `models/*`, `data/processed/*` y `*.joblib`. Para usar el artefacto oficial, desde la raíz:

```bash
MODELO=$(uv run python -c "import json; print(json.load(open('docs/frozen_solution.json'))['artifact_path'])")
uv run python -m src.recommend --model "$MODELO" --listing examples/listing.json
```

Si el `.joblib` no está disponible, se puede generar una copia nueva sin sobrescribir los JSON oficiales ni repetir septiembre:

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

La copia nueva se entrena solo con junio y puede tener un SHA-256 distinto por diferencias de entorno o serialización. El resultado final oficial sigue siendo el registrado en `docs/final_evaluation.json`; reproducir el entrenamiento no equivale a repetir la evaluación temporal.
