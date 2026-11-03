# Auditoría inicial de datos

Proyecto: `data-science-airbnb-pricing` — Ciudad de México.

Ejecución UTC: `2026-11-03T18:35:30.534237+00:00`.

## Alcance

Esta auditoría valida infraestructura y calidad básica del archivo detallado `listings.csv.gz`. No contiene EDA profundo, entrenamiento ni resultados de modelos.

## Procedencia

Fuente: [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/). Las URLs exactas, fechas de descarga, tamaños y SHA-256 están en `docs/data_manifest.json`.
El [diccionario oficial de datos](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing) define `price` como precio diario en moneda local y documenta el símbolo `$` como artefacto técnico del export.
La [Ley Monetaria de México](https://www.ordenjuridico.gob.mx/Documentos/Federal/html/wo14412.html) establece el peso como unidad del sistema monetario; por eso el objetivo monetario del proyecto se documenta como MXN.

## Estado de los archivos

| Archivo | Filas | Columnas | id nulos | id únicos | id duplicados | price nulos | price parseable | price inválidos | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `data/raw/mexico_city_2026-06-15_listings.csv.gz` | 31430 | 90 | 0 | 31430 | 0 | 1793 | 29637 | 0 | `bbe24eee4a96cb41a157e8b3eed9420225ec9fb94137233d9c2f4ce6d64eebfe` |
| `data/raw/mexico_city_2026-09-23_listings.csv.gz` | 30468 | 90 | 0 | 30468 | 0 | 2108 | 28360 | 0 | `76cd2d8186d8e2d6b62ebe769d83322ef7a2dcf059b7eede8492ef61849c1cee` |

### `data/raw/mexico_city_2026-06-15_listings.csv.gz`

- Snapshot: `2026-06-15`
- Tamaño: `18411663` bytes
- Columnas: `id, listing_url, scrape_id, last_scraped, source, name, description, neighborhood_overview, picture_url, host_id, host_url, host_profile_id, host_profile_url, host_name, host_since, hosts_time_as_user_years, hosts_time_as_user_months, hosts_time_as_host_years, hosts_time_as_host_months, host_location, host_about, host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, host_thumbnail_url, host_picture_url, host_neighbourhood, host_listings_count, host_total_listings_count, host_verifications, host_has_profile_pic, host_identity_verified, neighbourhood, neighbourhood_cleansed, neighbourhood_group_cleansed, latitude, longitude, property_type, room_type, accommodates, bathrooms, bathrooms_text, bedrooms, beds, amenities, price, price_quote_checkin_date, price_quote_checkout_date, price_quote_total_price, price_quote_price_per_night, price_quote_raw, minimum_nights, maximum_nights, minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights, maximum_maximum_nights, minimum_nights_avg_ntm, maximum_nights_avg_ntm, calendar_updated, has_availability, availability_30, availability_60, availability_90, availability_365, calendar_last_scraped, number_of_reviews, number_of_reviews_ltm, number_of_reviews_l30d, availability_eoy, number_of_reviews_ly, estimated_occupancy_l365d, estimated_revenue_l365d, first_review, last_review, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, license, instant_bookable, calculated_host_listings_count, calculated_host_listings_count_entire_homes, calculated_host_listings_count_private_rooms, calculated_host_listings_count_shared_rooms, reviews_per_month`
- Columna `currency` presente: `False`
- Formato de `price`: `{'parseable': 29637, 'missing': 1793}`
- Marcadores de moneda en `price`: `{'symbol:$ (ambiguous)': 29637}`
- Valores de columna `currency`: `{}`
- Moneda interpretada: `MXN`, por la definición oficial de moneda local y el alcance geográfico Mexico City; el código MXN no aparece explícito en el CSV.
- Ejemplos no interpretados como evidencia de moneda: `['$2,066.00', '$628.00', '$499.00', '$2,511.00', '$8,263.24']`
- Hallazgos: `['No existe columna currency; la moneda se documenta con la definición oficial de price.', 'El símbolo $ es un artefacto técnico del export según el diccionario oficial; no se usó como evidencia independiente.']`

### `data/raw/mexico_city_2026-09-23_listings.csv.gz`

- Snapshot: `2026-09-23`
- Tamaño: `17169955` bytes
- Columnas: `id, listing_url, scrape_id, last_scraped, source, name, description, neighborhood_overview, picture_url, host_id, host_url, host_profile_id, host_profile_url, host_name, host_since, hosts_time_as_user_years, hosts_time_as_user_months, hosts_time_as_host_years, hosts_time_as_host_months, host_location, host_about, host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, host_thumbnail_url, host_picture_url, host_neighbourhood, host_listings_count, host_total_listings_count, host_verifications, host_has_profile_pic, host_identity_verified, neighbourhood, neighbourhood_cleansed, neighbourhood_group_cleansed, latitude, longitude, property_type, room_type, accommodates, bathrooms, bathrooms_text, bedrooms, beds, amenities, price, price_quote_checkin_date, price_quote_checkout_date, price_quote_total_price, price_quote_price_per_night, price_quote_raw, minimum_nights, maximum_nights, minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights, maximum_maximum_nights, minimum_nights_avg_ntm, maximum_nights_avg_ntm, calendar_updated, has_availability, availability_30, availability_60, availability_90, availability_365, calendar_last_scraped, number_of_reviews, number_of_reviews_ltm, number_of_reviews_l30d, availability_eoy, number_of_reviews_ly, estimated_occupancy_l365d, estimated_revenue_l365d, first_review, last_review, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, license, instant_bookable, calculated_host_listings_count, calculated_host_listings_count_entire_homes, calculated_host_listings_count_private_rooms, calculated_host_listings_count_shared_rooms, reviews_per_month`
- Columna `currency` presente: `False`
- Formato de `price`: `{'parseable': 28360, 'missing': 2108}`
- Marcadores de moneda en `price`: `{'symbol:$ (ambiguous)': 28360}`
- Valores de columna `currency`: `{}`
- Moneda interpretada: `MXN`, por la definición oficial de moneda local y el alcance geográfico Mexico City; el código MXN no aparece explícito en el CSV.
- Ejemplos no interpretados como evidencia de moneda: `['$3,505.00', '$8,874.00', '$21,950.00', '$605.00', '$3,095.00']`
- Hallazgos: `['No existe columna currency; la moneda se documenta con la definición oficial de price.', 'El símbolo $ es un artefacto técnico del export según el diccionario oficial; no se usó como evidencia independiente.']`

## Comparación entre snapshots

- IDs en A: `31430`; IDs en B: `30468`.
- Intersección: `27804`.
- Solo A: `3626`; solo B: `2664`.
- Cobertura de A en B: `0.8846325167037862`; de B en A: `0.9125640015754234`.

## Decisiones de framing y fuga de información

El precio se recomendará para un anfitrión nuevo: no se deben usar reseñas, calificación, número de reseñas, superhost, tasa de respuesta, aceptación o cancelación si no están disponibles al publicar el anuncio. También se revisarán como potencial fuga variables calculadas después de publicar, como disponibilidad futura observada, precio histórico, métricas de ocupación, ingresos, reservas y cualquier señal derivada de reviews.

En estas columnas concretas habrá que excluir o justificar `availability_30/60/90/365`, `availability_eoy`, `calendar_updated`, `calendar_last_scraped`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `price_quote_*`, `number_of_reviews*`, `review_scores_*`, `first_review`, `last_review`, `reviews_per_month`, `host_is_superhost`, métricas de respuesta/aceptación, `calculated_host_listings_count*` y metadatos de scrape como `scrape_id`, `last_scraped` y `source`.

El diccionario oficial define `price` como moneda local y el símbolo `$` como artefacto técnico. Para estos snapshots de Mexico City, esa definición sustenta reportar MXN; el CSV no contiene una columna de código de moneda.

## Limitaciones de esta ejecución

La descarga y esta auditoría deben repetirse si se reemplazan los snapshots. Los bytes crudos no se transforman; cualquier derivado futuro debe quedar en `data/processed`.
