# Política de variables

Proyecto: `data-science-airbnb-pricing` — Fase 2.

La unidad de decisión es el precio que podría fijar un anfitrión nuevo antes de publicar. Por eso una variable solo puede ser predictor si existe o puede conocerse en ese momento. `id` no se usa como predictor. La política se auditó individualmente contra las columnas del snapshot de junio y no usa el snapshot de septiembre.

## Categorías

- `Permitida`: candidata disponible antes de publicar.
- `Permitida, opcional`: disponible o potencialmente disponible antes de publicar, pero su estado puede cambiar o depende de una decisión del anfitrión; se debe evaluar en una comparación de sensibilidad.
- `Permitida con revisión / diferida`: podría estar disponible, pero se excluye del pipeline actual por ambigüedad, texto o codificación pendiente.
- `Prohibida`: objetivo, identificador, métrica posterior o variable con fuga.
- `Ambigua: no usar todavía`: requiere una decisión adicional antes de modelar.

## Decisión por columna

| Columna | Decisión | Justificación |
|---|---|---|
| `id` | Prohibida | Identificador técnico; nunca debe ser predictor. |
| `listing_url` | Prohibida | Identificador/enlace del anuncio; no es una característica causal y puede filtrar identidad. |
| `scrape_id` | Prohibida | Metadato del momento de recolección; no está disponible al publicar. |
| `last_scraped` | Prohibida | Fecha de recolección; metadato temporal, no característica del anuncio. |
| `source` | Prohibida | Indica cómo se encontró el anuncio en el scrape; metadato de recolección. |
| `name` | Permitida con revisión / diferida | Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas. |
| `description` | Permitida con revisión / diferida | Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas. |
| `neighborhood_overview` | Permitida con revisión / diferida | Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas. |
| `picture_url` | Prohibida | Imagen del anuncio; se difiere visión por computador y no entra en el pipeline numérico. |
| `host_id` | Prohibida | Identificador técnico del anfitrión; puede memorizar el objetivo y no generaliza a un anfitrión nuevo. |
| `host_url` | Prohibida | Enlace identificador del anfitrión; no es predictor. |
| `host_profile_id` | Prohibida | Identificador técnico del perfil; no es predictor. |
| `host_profile_url` | Prohibida | Enlace identificador del perfil; no es predictor. |
| `host_name` | Prohibida | Nombre del anfitrión; no es una característica de publicación apropiada y puede contener información personal. |
| `host_since` | Permitida con revisión / diferida | Conocida por la plataforma, pero funciona como proxy de experiencia y no es estable para un anfitrión nuevo. |
| `hosts_time_as_user_years` | Ambigua: no usar todavía | No hay una justificación explícita de disponibilidad temporal; revisar antes de modelar. |
| `hosts_time_as_user_months` | Ambigua: no usar todavía | No hay una justificación explícita de disponibilidad temporal; revisar antes de modelar. |
| `hosts_time_as_host_years` | Ambigua: no usar todavía | No hay una justificación explícita de disponibilidad temporal; revisar antes de modelar. |
| `hosts_time_as_host_months` | Ambigua: no usar todavía | No hay una justificación explícita de disponibilidad temporal; revisar antes de modelar. |
| `host_location` | Prohibida | Ubicación autodeclarada del anfitrión, no del alojamiento; no es necesaria para la decisión. |
| `host_about` | Permitida con revisión / diferida | Texto del perfil; se difiere por posible información personal y por no ser necesario para el baseline. |
| `host_response_time` | Prohibida | Métrica de respuesta posterior a interacciones; fuga de información para publicar. |
| `host_response_rate` | Prohibida | Métrica de comportamiento posterior; fuga de información para publicar. |
| `host_acceptance_rate` | Prohibida | Métrica de solicitudes aceptadas; posterior a publicar y potencial fuga. |
| `host_is_superhost` | Prohibida | Estatus de reputación acumulada; un anfitrión nuevo no lo tiene al publicar. |
| `host_thumbnail_url` | Prohibida | Imagen/enlace de perfil; no se usará como predictor en esta fase. |
| `host_picture_url` | Prohibida | Imagen/enlace de perfil; no se usará como predictor en esta fase. |
| `host_neighbourhood` | Prohibida | Zona del anfitrión, no del anuncio; no representa la ubicación del alojamiento. |
| `host_listings_count` | Prohibida | Inventario actual del anfitrión; no representa a un anfitrión nuevo y puede reflejar actividad posterior. |
| `host_total_listings_count` | Prohibida | Inventario actual del anfitrión; no representa a un anfitrión nuevo y puede reflejar actividad posterior. |
| `host_verifications` | Permitida con revisión / diferida | Estado previo y disponible, pero requiere codificación de lista; se difiere para evitar decisiones prematuras. |
| `host_has_profile_pic` | Permitida, opcional | Puede conocerse antes de publicar si el anfitrión completa su perfil, pero el snapshot observa el estado actual; se conserva como candidata opcional y debe probarse sin asumir estabilidad histórica. |
| `host_identity_verified` | Permitida, opcional | Puede conocerse antes de publicar si el anfitrión completa la verificación, pero el snapshot observa el estado actual; se conserva como candidata opcional y debe probarse sin asumir estabilidad histórica. |
| `neighbourhood` | Permitida con revisión / diferida | Texto de ubicación declarado por el anfitrión; puede ser inconsistente, se prefiere neighbourhood_cleansed. |
| `neighbourhood_cleansed` | Permitida | Alcaldía geocodificada a partir de la ubicación; conocida antes de publicar. |
| `neighbourhood_group_cleansed` | Permitida | Contexto geográfico derivado de ubicación; se conserva si tiene cobertura. |
| `latitude` | Permitida | Ubicación aproximada; conocida antes de publicar, con la limitación de anonimización espacial. |
| `longitude` | Permitida | Ubicación aproximada; conocida antes de publicar, con la limitación de anonimización espacial. |
| `property_type` | Permitida | Tipo de propiedad declarado en el anuncio. |
| `room_type` | Permitida | Tipo de alojamiento declarado en el anuncio. |
| `accommodates` | Permitida | Capacidad máxima declarada. |
| `bathrooms` | Permitida | Cantidad de baños declarada; se prioriza sobre bathrooms_text para modelar. |
| `bathrooms_text` | Permitida con revisión / diferida | Información potencialmente útil, pero redundante con bathrooms y textual; no entra en la tabla limpia. |
| `bedrooms` | Permitida | Cantidad de dormitorios declarada. |
| `beds` | Permitida | Cantidad de camas declarada. |
| `amenities` | Permitida | Amenidades declaradas antes de publicar; se usará una representación simple en esta fase. |
| `price` | Prohibida | Variable objetivo; nunca debe entrar como predictor. |
| `price_quote_checkin_date` | Prohibida | Parte de una cotización observada; depende de una consulta posterior y no es predictor base. |
| `price_quote_checkout_date` | Prohibida | Parte de una cotización observada; depende de una consulta posterior y no es predictor base. |
| `price_quote_total_price` | Prohibida | Derivado del precio y de una cotización; fuga directa del objetivo. |
| `price_quote_price_per_night` | Prohibida | Derivado del precio; fuga directa del objetivo. |
| `price_quote_raw` | Prohibida | Respuesta cruda de una cotización; contiene información del objetivo. |
| `minimum_nights` | Permitida, opcional | Regla de estadía que el anfitrión puede definir antes de publicar; se conserva como candidata opcional porque depende de la estrategia que el anfitrión decida ofrecer. |
| `maximum_nights` | Permitida, opcional | Regla de estadía que el anfitrión puede definir antes de publicar; se conserva como candidata opcional porque depende de la estrategia que el anfitrión decida ofrecer. |
| `minimum_minimum_nights` | Prohibida | Resumen calculado mirando el calendario futuro; no disponible al publicar. |
| `maximum_minimum_nights` | Prohibida | Resumen calculado mirando el calendario futuro; no disponible al publicar. |
| `minimum_maximum_nights` | Prohibida | Resumen calculado mirando el calendario futuro; no disponible al publicar. |
| `maximum_maximum_nights` | Prohibida | Resumen calculado mirando el calendario futuro; no disponible al publicar. |
| `minimum_nights_avg_ntm` | Prohibida | Promedio calculado de reglas futuras; no disponible al publicar. |
| `maximum_nights_avg_ntm` | Prohibida | Promedio calculado de reglas futuras; no disponible al publicar. |
| `calendar_updated` | Prohibida | Fecha de actualización del calendario; puede ocurrir después de publicar. |
| `has_availability` | Prohibida | Estado observado en calendario; mezcla reservas y bloqueos posteriores. |
| `availability_30` | Prohibida | Disponibilidad observada a 30 días; variable futura y potencial fuga. |
| `availability_60` | Prohibida | Disponibilidad observada a 60 días; variable futura y potencial fuga. |
| `availability_90` | Prohibida | Disponibilidad observada a 90 días; variable futura y potencial fuga. |
| `availability_365` | Prohibida | Disponibilidad observada a 365 días; variable futura y potencial fuga. |
| `calendar_last_scraped` | Prohibida | Fecha de observación del calendario; metadato posterior. |
| `number_of_reviews` | Prohibida | Reseñas acumuladas; un anfitrión nuevo no las tiene. |
| `number_of_reviews_ltm` | Prohibida | Reseñas de los últimos 12 meses; posterior a publicar. |
| `number_of_reviews_l30d` | Prohibida | Reseñas de los últimos 30 días; posterior a publicar. |
| `availability_eoy` | Prohibida | Disponibilidad futura observada; potencial fuga. |
| `number_of_reviews_ly` | Prohibida | Reseñas del año anterior; posterior a publicar. |
| `estimated_occupancy_l365d` | Prohibida | Ocupación estimada con datos posteriores y reviews; fuga. |
| `estimated_revenue_l365d` | Prohibida | Ingresos estimados con datos posteriores y reviews; fuga. |
| `first_review` | Prohibida | Fecha de la primera reseña; posterior y ausente para anfitrión nuevo. |
| `last_review` | Prohibida | Fecha de la última reseña; posterior y ausente para anfitrión nuevo. |
| `review_scores_rating` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_accuracy` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_cleanliness` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_checkin` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_communication` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_location` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `review_scores_value` | Prohibida | Calificación acumulada; posterior y ausente para anfitrión nuevo. |
| `license` | Permitida con revisión / diferida | Dato regulatorio potencialmente previo, pero con alta ausencia y significado heterogéneo; se difiere. |
| `instant_bookable` | Permitida | Configuración del anuncio disponible antes de una reserva. |
| `calculated_host_listings_count` | Prohibida | Conteo calculado del inventario actual; puede reflejar actividad posterior. |
| `calculated_host_listings_count_entire_homes` | Prohibida | Conteo calculado del inventario actual; puede reflejar actividad posterior. |
| `calculated_host_listings_count_private_rooms` | Prohibida | Conteo calculado del inventario actual; puede reflejar actividad posterior. |
| `calculated_host_listings_count_shared_rooms` | Prohibida | Conteo calculado del inventario actual; puede reflejar actividad posterior. |
| `reviews_per_month` | Prohibida | Derivado de reviews y del tiempo de vida; fuga. |

## Regla de modelado

El pipeline de esta fase usa solo un subconjunto numérico y categórico de variables permitidas, más `amenities_count`; `price_mxn` es el objetivo y `id` no se exporta. `neighbourhood_group_cleansed` e `instant_bookable` son conceptualmente disponibles antes de publicar, pero se excluyen del subconjunto seleccionado porque en junio tienen cobertura nula completa y quedarían como constantes. `amenities` se resume en `amenities_count`. Las demás variables diferidas o prohibidas no entran al pipeline. La decisión se toma solo con desarrollo de junio; el test de septiembre no participa.
