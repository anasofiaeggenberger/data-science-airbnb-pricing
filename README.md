# data-science-airbnb-pricing

Proyecto final de Ciencia de Datos — segundo semestre 2026, UFM.

Pregunta: ¿qué precio por noche debe cobrar un anfitrión nuevo de Airbnb en Ciudad de México? El objetivo es recomendar un precio en MXN y un rango razonable usando anuncios reales de Inside Airbnb. La propuesta contempla un baseline de mediana por alcaldía y tipo de alojamiento, regresión lineal regularizada sobre `log(precio)` y LightGBM con pérdida por cuantiles. La métrica principal será pinball loss, además de MAE y cobertura del intervalo.

Esta entrega cubre las Fases 1 y 2: infraestructura, adquisición, auditoría inicial, preparación de datos y EDA descriptivo sobre junio. No incluye entrenamiento de modelos ni resultados inventados.

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
src/               adquisición y auditoría
tests/             pruebas pequeñas de funciones críticas
```

## Instalación con uv

Se requiere Python 3.12.

```bash
uv sync
```

Esto crea/usa `.venv` y respeta `uv.lock`. Si Python 3.12 no está instalado localmente, instalarlo con el mecanismo de Python disponible en el equipo y repetir `uv sync`.

## Descarga y auditoría

Con conectividad a la fuente oficial:

```bash
uv run python -m src.download_data
uv run python -m src.audit_data
```

La descarga conserva los bytes del `listings.csv.gz` original y registra URL, snapshot, fecha UTC, tamaño y SHA-256 en `docs/data_manifest.json`. Si la página expone menos de dos enlaces detallados, el script termina con error y no inventa una segunda URL. También se pueden pasar enlaces exactos ya verificados:

```bash
uv run python -m src.download_data --url URL_SNAPSHOT_1 --url URL_SNAPSHOT_2
```

La auditoría verifica columnas, filas, `id`, `price`, marcadores de moneda e intersección de IDs. El [diccionario oficial de Inside Airbnb](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing) documenta `price` como precio diario en moneda local y el `$` como artefacto técnico del export; para Mexico City esto sustenta reportar MXN, aunque el CSV no contiene una columna explícita `currency`.

## Preparación y EDA de Fase 2

La separación y limpieza reproducible se ejecutan con:

```bash
uv run python -m src.preparation
```

Este comando usa junio para construir `data/processed/development_clean.csv`, `docs/cleaning_log.json`, `docs/feature_policy.md` y `docs/split_manifest.json`. El archivo procesado y los IDs del test están aislados: el manifiesto guarda conteos de elegibilidad, pero no guarda filas ni IDs de septiembre.

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

Para reconstruir los artefactos ignorados desde un clon limpio, usar los URLs exactos ya verificados y luego ejecutar la preparación:

```bash
uv sync
uv run python -m src.download_data \
  --url https://data.insideairbnb.com/mexico/df/mexico-city/2026-06-15/data/listings.csv.gz \
  --url https://data.insideairbnb.com/mexico/df/mexico-city/2026-09-23/data/listings.csv.gz
uv run python -m src.audit_data
uv run python -m src.preparation
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3 notebooks/01_eda.ipynb
```

La notebook depende del snapshot de junio en `data/raw/`, que está ignorado por Git, y regenera `data/processed/development_clean.csv` y las figuras ignoradas. Los JSON y la política en `docs/` sí se conservan en Git. `docs/cleaning_log.json` registra estadísticas ajustadas sobre todo junio para EDA y desarrollo final; no se debe partir luego ese CSV para validación interna.

## Regla para validación interna futura

Antes de cualquier validación cruzada o partición interna de junio, se debe separar el fold y ajustar dentro del fold de entrenamiento todas las reglas que aprendan de datos: imputaciones, agrupación de categorías raras, escalado y umbrales de extremos. Después se aplican esas reglas al fold de validación. Las estadísticas descriptivas de `docs/cleaning_log.json` y de la notebook son solo EDA. El precio objetivo de validación debe conservarse en su escala original para calcular pinball loss, MAE y cobertura. El snapshot de septiembre no participa en ninguna de estas decisiones.

## Reglas de disponibilidad y fuga

El caso de uso es un anfitrión nuevo. No deben usarse como predictores al publicar: reseñas, calificaciones, número de reseñas, superhost, tasas de respuesta/aceptación/cancelación ni cualquier métrica que solo exista después de recibir reservas o huéspedes. También se revisarán como posibles fugas `availability_30/60/90/365`, `availability_eoy`, `calendar_updated`, `calendar_last_scraped`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `price_quote_*`, variables de reviews, `calculated_host_listings_count*` y metadatos del scrape (`scrape_id`, `last_scraped`, `source`).

## Pruebas

```bash
uv run pytest
uv lock --check
git diff --check
```

Las pruebas cubren parseo numérico de `price`, estados de validez, separación temporal, reglas de limpieza, extracción de enlaces detallados y auditoría de IDs/moneda en archivos pequeños temporales.

## Flujo colaborativo

1. Crear una rama descriptiva desde `main`, por ejemplo `feat/data-preparation-eda`.
2. Trabajar en cambios pequeños, ejecutar las pruebas y validaciones (`pytest`, `uv lock --check`, `git diff --check`) y documentar comandos/resultados.
3. Abrir un pull request hacia `main` con descripción, evidencia de auditoría, figuras y limitaciones.
4. Revisar al menos por otra persona del equipo; resolver comentarios antes de integrar.
5. No subir datos crudos, modelos ni figuras generadas: el código, la documentación y `uv.lock` sí se conservan en Git.

## Ridge y recomendador

La base para modelar, las particiones por anfitrión, Ridge con calibración de residuos y el recomendador están implementados. Véanse [contrato común](docs/modeling_contract.md), [comandos reproducibles](docs/reproducibility.md) y [notebook de Ridge](notebooks/03_ridge_recommender.ipynb).

La sección inicial de Fases 1 y 2 describe la entrega de datos anterior. Este avance añade validación de Ridge exclusivamente en junio. No declara un ganador entre familias ni evalúa septiembre. La comparación con baseline/LightGBM y la evaluación final del test permanecen pendientes. Los resultados de esta búsqueda están en `docs/ridge_validation.json`; la bitácora de esta sesión está en `docs/ai/modeling_log.md`.
