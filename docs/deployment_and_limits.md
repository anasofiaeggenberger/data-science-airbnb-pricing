# Despliegue y limitaciones del recomendador de precios

## 1. Objetivo y usuario previsto

El proyecto busca apoyar la definición de un precio diario en MXN para un anuncio nuevo de Airbnb en Ciudad de México. El usuario previsto es una persona anfitriona o integrante de un equipo que necesita una referencia inicial para publicar un alojamiento.

La recomendación debe entenderse como apoyo para tomar una decisión, no como una tasación ni como una promesa de resultado comercial. El precio final también requiere criterio humano y conocimiento de la propiedad, la zona, la temporada y las condiciones del mercado.

El estado revisado del repositorio cubre la infraestructura de datos, la adquisición, la auditoría, la preparación y la exploración descriptiva de junio. También incluye el código de Ridge, la serialización del modelo y un recomendador local por línea de comandos. El artefacto `models/ridge.joblib` se genera localmente y no se versiona. En esta rama todavía no están integradas la comparación común con LightGBM ni la evaluación final temporal.

## 2. Flujo de uso implementado y previsto

Para el candidato Ridge, el flujo local ya está implementado. La comparación con otros modelos y la selección global siguen pendientes del PR #4.

1. La persona ingresa las características disponibles antes de publicar el anuncio. `src/recommend.py` exige actualmente `latitude`, `longitude`, `accommodates`, `neighbourhood_cleansed`, `property_type` y `room_type`. También puede recibir `bathrooms`, `bedrooms`, `beds`, `amenities_count`, `minimum_nights` y `maximum_nights`, que se validan si están presentes.
2. `validate_listing` comprueba campos obligatorios, tipos de texto y rangos numéricos razonables. `id` y `host_id` no forman parte de los predictores. Las categorías no observadas no se rechazan en la codificación: el `OneHotEncoder` del preprocesador usa `handle_unknown='infrequent_if_exist'`.
3. `prepare_features` convierte valores numéricos y categorías, calcula `amenities_count` cuando corresponde y selecciona solo las columnas permitidas. El `Pipeline` ajusta imputación, escalado, agrupación de categorías poco frecuentes y codificación dentro del entrenamiento. En validación, `src/train_ridge.py` vuelve a ajustar el pipeline dentro de cada fold.
4. `RidgeQuantileModel` ajusta Ridge sobre `log(precio)`. Cuando recibe `host_id`, calcula residuos logarítmicos fuera de muestra mediante folds agrupados para calibrar p10, p35, p50 y p90. Después transforma las predicciones nuevamente a MXN.
5. `src/recommend.py` carga un artefacto joblib confiable, valida que los cuatro cuantiles sean finitos, positivos y estén ordenados, y devuelve el precio p35, p10, p50 y p90 en MXN. La respuesta incluye una nota que aclara que son referencias de precios publicados y no garantías comerciales.

Este flujo local está implementado para un artefacto Ridge generado con junio. No implica que LightGBM sea el modelo elegido ni que exista todavía una interfaz web o un servicio productivo.

## 3. Arquitectura conceptual de despliegue

La arquitectura mínima prevista puede representarse así:

```text
Características del anuncio
          |
          v
Validación de entradas y contrato de variables
          |
          v
Preprocesamiento guardado con el modelo
          |
          v
Modelo de precios en la escala definida
          |
          v
Conversión y presentación de resultados en MXN
```

El repositorio permite ejecutar localmente la descarga, la auditoría, la preparación, el entrenamiento de Ridge y la inferencia del recomendador. No contiene una API, una interfaz web, un contenedor ni un servicio hospedado. Por eso, este documento describe una arquitectura conceptual y no afirma que exista un despliegue productivo.

La ejecución local del candidato Ridge se documenta con:

```bash
uv run python -m src.train_ridge
uv run python -m src.recommend --model models/ridge.joblib --listing examples/listing.json
```

El primer comando genera el artefacto ignorado por Git a partir de junio; el segundo lo carga y procesa el archivo de ejemplo. El modelo disponible por este flujo es un candidato Ridge, no necesariamente la solución final del proyecto.

Una futura implementación podría empezar como una herramienta de línea de comandos o un proceso local y, después de validar el contrato y el modelo, envolverse en una API o interfaz. Esa decisión todavía no está implementada ni debe darse por hecha.

## 4. Ejecución local y reproducibilidad

Para reproducir la parte actualmente disponible se requiere Python `3.12` y `uv`. Las dependencias declaradas están en `pyproject.toml` y las versiones resueltas en `uv.lock`. LightGBM figura como dependencia del proyecto, aunque su ejecutor y comparación todavía están pendientes del PR #4.

El flujo de datos actual puede ejecutarse con:

```bash
uv sync --frozen
uv run python -m src.download_data
uv run python -m src.audit_data
uv run python -m src.preparation
uv run pytest
uv lock --check
```

La adquisición debe usar los enlaces oficiales de Inside Airbnb y conservar la URL, la fecha de snapshot y el hash registrados en los manifiestos. La preparación actual separa conceptualmente junio como desarrollo y reserva el snapshot posterior para una evaluación final; esa separación no debe romperse para ajustar transformaciones o hiperparámetros.

La preparación existente genera artefactos de datos para la fase de desarrollo. Después, `uv run python -m src.train_ridge` entrena y serializa el candidato Ridge, y `src.recommend` ejecuta la inferencia local. El artefacto y los CSV derivados se reconstruyen porque están ignorados por Git.

Para que una futura ejecución sea reproducible se deben conservar, como mínimo, el snapshot exacto, su hash, la política de variables, la partición agrupada por `host_id`, la semilla, la configuración de hiperparámetros, el preprocesador y la versión del modelo. Los campos `id` y `host_id` no deben entrar como predictores.

## 5. Precio recomendado y cuantiles

La salida debe estar expresada en MXN y conservar claramente la diferencia entre una predicción y el precio que finalmente podría observarse en el mercado. Un precio recomendado sería un punto de referencia operacional obtenido a partir de las características del anuncio y del modelo elegido; no representa una garantía de demanda ni de venta.

El contrato implementado para Ridge define p10, p35, p50 y p90 como cuantiles de la distribución estimada del precio publicado, condicionada a las características del anuncio. No son intervalos de confianza sobre la media ni probabilidades de reserva. Su interpretación es:

- **p10:** una referencia baja dentro de la distribución estimada de precios para anuncios comparables.
- **p35:** el cuantil recomendado y un punto operativo relativamente conservador para el precio sugerido.
- **p50:** la mediana estimada; divide la distribución prevista en dos partes iguales.
- **p90:** una referencia alta dentro de la distribución estimada.

Estos cuantiles describen incertidumbre del modelo bajo los datos y supuestos utilizados. El intervalo p10–p90 tiene una cobertura nominal del 80 % en el contrato de métricas, pero esa cobertura no está garantizada para cada anuncio ni para el mercado futuro. No son probabilidades de conseguir una reserva, no son límites del mercado y no garantizan que el precio publicado sea pagado. Las comisiones, impuestos, descuentos, limpieza, estancias mínimas, disponibilidad, ocupación y condiciones de cada reserva pueden cambiar el resultado real.

El contrato común de `src/metrics.py` define pinball, MAE y cobertura para comparar modelos. En esta rama, `src/train_ridge.py` todavía calcula las métricas de Ridge con funciones locales y las marca como provisionales; falta regenerar sus resultados con las métricas comunes. El baseline está integrado, pero la comparación común con LightGBM y la selección global siguen pendientes del PR #4.

## 6. Limitaciones de los datos de Inside Airbnb

La fuente contiene información publicada de anuncios observados. El campo `price` se documenta como precio diario en la moneda local; para Ciudad de México el proyecto lo presenta en MXN, aunque el CSV no incluye una columna de moneda explícita. El símbolo `$` por sí solo no prueba la moneda.

Entre las limitaciones más importantes están las siguientes:

- El precio publicado no es necesariamente el precio efectivamente pagado. La fuente no permite inferir por sí sola reservas, descuentos negociados, impuestos, comisiones, cargos de limpieza, cancelaciones ni conversión de visitas en reservas.
- La muestra representa anuncios observados, no todo el inventario ni todos los posibles alojamientos. Puede haber sesgos por visibilidad, actividad, ubicación, tipo de anfitrión y disponibilidad para ser capturado.
- Un snapshot refleja un momento concreto. Cambios de temporada, eventos, regulación, competencia, inflación o demanda pueden hacer que un modelo entrenado con junio pierda vigencia.
- Hay datos faltantes, formatos inconsistentes y categorías que pueden aparecer o desaparecer entre snapshots. Las categorías desconocidas deben tener una conducta definida y observable en el preprocesamiento.
- Los precios extremos pueden ser errores, casos legítimos o una combinación de ambos. La preparación actual conserva los valores válidos y no demuestra que los extremos hayan sido eliminados; cualquier regla futura debe aprenderse solo con los datos de entrenamiento y documentarse.

Estas limitaciones impiden interpretar la predicción como una medición completa del valor económico del alojamiento.

## 7. Generalización y cambios del mercado

El desempeño puede variar entre alcaldías, colonias, zonas turísticas y zonas residenciales. También puede variar entre casas completas, habitaciones privadas y otros tipos de alojamiento. Una categoría con pocos ejemplos puede producir predicciones inestables o quedar fuera de la representación aprendida.

La ubicación disponible en la fuente tampoco equivale a una evaluación perfecta de la microzona. La calidad del alojamiento, el estado de la propiedad, la vista, el ruido, la seguridad percibida, la reputación del anfitrión y otros factores pueden no estar representados o pueden ser información posterior a la publicación que la política prohíbe utilizar.

Un cambio en la distribución de anuncios, en la forma de publicar precios o en el comportamiento de la demanda puede degradar el modelo aunque el código siga funcionando. Por ese motivo, un resultado válido para junio no debe extrapolarse automáticamente a otra fecha, alcaldía, tipo de alojamiento o condición de mercado.

## 8. Mantenimiento y actualización

Ante cada nueva actualización se recomienda:

1. Descargar únicamente desde la fuente oficial, conservar la URL exacta, la fecha y el hash, y repetir la auditoría de columnas, identificadores, precios y faltantes.
2. Revisar cambios de esquema, valores nuevos, categorías desconocidas y cobertura geográfica antes de volver a entrenar.
3. Mantener separado el snapshot reservado para evaluación final. No usarlo repetidamente para elegir variables, imputaciones, umbrales, hiperparámetros o calibración.
4. Reentrenar solo con un protocolo documentado y con transformaciones ajustadas dentro de cada partición de entrenamiento. Las particiones deben ser reproducibles y estar agrupadas por `host_id`.
5. Registrar versión de código, datos, modelo, semilla, configuración, métricas y fecha de entrenamiento.
6. Comparar el modelo nuevo con el baseline y revisar pinball, MAE y cobertura en la escala correspondiente. También conviene revisar errores por alcaldía y tipo de alojamiento, sin convertir esos resultados en garantías de desempeño futuro.
7. Definir una señal de alerta para deriva de datos o deterioro de errores y establecer cuándo se debe volver a auditar o entrenar.

## 9. Precauciones de uso

El recomendador debe presentarse como una herramienta informativa para apoyar la decisión de publicación. No debe afirmar ni garantizar:

- que habrá reservas, ocupación o un determinado número de noches vendidas;
- que se obtendrán ingresos, rentabilidad o retorno de inversión;
- que el precio sugerido será el precio de mercado o el precio finalmente pagado;
- que el modelo conoce las condiciones particulares de una propiedad que no estén en los datos.

También debe evitarse ingresar o utilizar información personal innecesaria, identificadores como `id` o `host_id`, métricas de desempeño posteriores a la publicación y variables expresamente prohibidas por la política de features. Cuando falte una característica relevante o el anuncio esté fuera de la cobertura del modelo, la aplicación futura debería informarlo en lugar de ocultar la incertidumbre.

## 10. Estado actual y pendientes de integración

### Implementado en el estado revisado

- Descarga desde enlaces oficiales de Inside Airbnb y conservación de metadatos de procedencia y hash.
- Auditoría básica de archivos, columnas, identificadores y precios.
- Preparación de un conjunto de desarrollo basado en junio y documentación de limpieza.
- Política de variables con exclusiones explícitas, incluyendo `id`, `host_id` y variables posteriores a la publicación.
- Manifiesto de separación temporal que reserva el snapshot posterior para una evaluación final.
- Notebook de exploración descriptiva de junio y pruebas automatizadas para las utilidades existentes.
- Contrato común de cuantiles y métricas en `src/metrics.py`, además del baseline por alcaldía y tipo de alojamiento.
- Ridge sobre `log(precio)`, con búsqueda de `alpha`, `GroupKFold` por `host_id`, preprocesamiento dentro de los folds y calibración interna de residuos fuera de muestra.
- Serialización del candidato Ridge con `joblib` y recomendador local en `src/recommend.py`, incluyendo validación de entradas y resultados p10, p35, p50 y p90 en MXN.
- Dependencias y procedimiento de reproducción local con `uv`.

### Pendiente o no disponible en esta rama

- Conectar la búsqueda de Ridge con el contrato común de métricas y regenerar sus resultados comparables; en esta rama todavía se marcan como provisionales las métricas de `src/train_ridge.py`.
- Integrar el ejecutor común y LightGBM del PR #4, con los mismos predictores, particiones y métricas para baseline, Ridge y LightGBM.
- Seleccionar el modelo final y conectar el recomendador con esa selección; actualmente el flujo local carga el candidato Ridge.
- Generar y revisar el análisis de errores y segmentos del experimento común.
- Ejecutar una única evaluación final sobre el snapshot reservado, después de congelar la solución y sin usar ese conjunto para tomar decisiones.
- API, interfaz web, contenedor, hospedaje y monitoreo productivo.

Las funciones pendientes deben integrarse y validarse con el contrato de datos antes de describirlas como disponibles. Esta rama ya contiene el trabajo de Ridge y recomendador integrado en el estado revisado; todavía no presenta LightGBM como ganador ni contiene resultados de la evaluación final. Este documento no sustituye esa integración ni presenta funcionalidades planificadas como resultados existentes.
