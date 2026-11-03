# data-science-airbnb-pricing

Proyecto final de Ciencia de Datos — segundo semestre 2026, UFM.

Pregunta: ¿qué precio por noche debe cobrar un anfitrión nuevo de Airbnb en Ciudad de México? El objetivo es recomendar un precio en MXN y un rango razonable usando anuncios reales de Inside Airbnb. La propuesta contempla un baseline de mediana por alcaldía y tipo de alojamiento, regresión lineal regularizada sobre `log(precio)` y LightGBM con pérdida por cuantiles. La métrica principal será pinball loss, además de MAE y cobertura del intervalo.

Integrantes: Sofía Eggenberger, Juan Pablo Madriz y Víctor Saravia.

Esta entrega cubre únicamente la Fase 1: infraestructura, adquisición y auditoría inicial. No incluye EDA profundo, modelos ni resultados inventados.

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

## Notebook inicial

`notebooks/01_eda.ipynb` contiene el framing, la carga inicial y reglas de disponibilidad temporal. No se ejecuta EDA profundo antes de separar el futuro test basado en anuncios que aparecen en el snapshot reciente y no estaban en el anterior.

## Reglas de disponibilidad y fuga

El caso de uso es un anfitrión nuevo. No deben usarse como predictores al publicar: reseñas, calificaciones, número de reseñas, superhost, tasas de respuesta/aceptación/cancelación ni cualquier métrica que solo exista después de recibir reservas o huéspedes. También se revisarán como posibles fugas `availability_30/60/90/365`, `availability_eoy`, `calendar_updated`, `calendar_last_scraped`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `price_quote_*`, variables de reviews, `calculated_host_listings_count*` y metadatos del scrape (`scrape_id`, `last_scraped`, `source`).

## Pruebas

```bash
uv run pytest
```

Las pruebas cubren parseo numérico de `price`, extracción de enlaces detallados y auditoría de IDs/moneda en archivos pequeños temporales.

## Flujo colaborativo

1. Crear una rama descriptiva desde `main`, por ejemplo `feat/data-audit` o `docs/framing`.
2. Trabajar en cambios pequeños, ejecutar `uv run pytest` y documentar comandos/resultados.
3. Abrir un pull request hacia `main` con descripción, evidencia de auditoría y limitaciones.
4. Revisar al menos por otra persona del equipo; resolver comentarios antes de integrar.
5. No subir datos crudos, modelos ni figuras generadas: el código, la documentación y `uv.lock` sí se conservan en Git.
