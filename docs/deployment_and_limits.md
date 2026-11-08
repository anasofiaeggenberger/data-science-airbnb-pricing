# Despliegue y limitaciones del recomendador de precios

## 1. Objetivo y usuario previsto

El proyecto busca apoyar la definición de un precio diario en MXN para un anuncio nuevo de Airbnb en Ciudad de México. El usuario previsto es una persona anfitriona o integrante de un equipo que necesita una referencia inicial para publicar un alojamiento.

La recomendación debe entenderse como apoyo para tomar una decisión, no como una tasación ni como una promesa de resultado comercial. El precio final también requiere criterio humano y conocimiento de la propiedad, la zona, la temporada y las condiciones del mercado.

El estado revisado del repositorio cubre la infraestructura de datos, la adquisición, la auditoría, la preparación y la exploración descriptiva de junio. El README describe modelos que se planean para una fase posterior, pero en esta rama todavía no hay un recomendador entrenado listo para usar.

## 2. Flujo de uso previsto

El flujo conceptual para una futura integración es el siguiente:

1. La persona ingresa las características disponibles antes de publicar el anuncio: ubicación, tipo de propiedad y habitación, capacidad, habitaciones, camas, baños y amenidades. La política de variables de `docs/feature_policy.md` es la referencia para decidir qué campos pueden utilizarse.
2. El sistema valida que las entradas tengan el formato esperado, que los valores numéricos sean razonables y que no se incluyan identificadores, métricas posteriores a la publicación ni otras variables prohibidas. También debe manejar valores faltantes y categorías que no se hayan visto durante el entrenamiento.
3. Se aplican las transformaciones aprendidas únicamente con los datos de entrenamiento: imputación, agrupación de categorías poco frecuentes, conteo de amenidades y, si corresponde al modelo, escalado. Las reglas aprendidas deben guardarse junto con el modelo y aplicarse de la misma manera al nuevo anuncio.
4. El modelo genera una predicción en la escala de `log(precio)` o en la escala que defina su contrato. La aplicación debe convertir el resultado a MXN antes de presentarlo.
5. La salida muestra un precio o rango sugerido, con una explicación breve de que se trata de una referencia basada en anuncios observados. Si el modelo final produce cuantiles, también muestra p10, p35, p50 y p90, siempre indicando que son estimaciones y no precios garantizados.

Este flujo es un diseño previsto. En el estado actual no existe el módulo de recomendación, un artefacto de modelo ni una interfaz que ejecute todos estos pasos.

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

El repositorio actual permite ejecutar localmente la descarga, la auditoría y la preparación de los datos. No contiene una API, una interfaz web, un contenedor ni un servicio hospedado para el recomendador. Por eso, este documento describe una arquitectura conceptual y no afirma que exista un despliegue productivo.

Una futura implementación podría empezar como una herramienta de línea de comandos o un proceso local y, después de validar el contrato y el modelo, envolverse en una API o interfaz. Esa decisión todavía no está implementada ni debe darse por hecha.

## 4. Ejecución local y reproducibilidad

Para reproducir la parte actualmente disponible se requiere Python `3.12` y `uv`. Las dependencias declaradas están en `pyproject.toml` y las versiones resueltas en `uv.lock`. El README documenta, entre otros, pandas, NumPy, scikit-learn, LightGBM, matplotlib, seaborn, Jupyter, ipykernel y pytest.

El flujo de datos actual puede ejecutarse con:

```bash
uv sync
uv run python -m src.download_data
uv run python -m src.audit_data
uv run python -m src.preparation
uv run pytest
uv lock --check
```

La adquisición debe usar los enlaces oficiales de Inside Airbnb y conservar la URL, la fecha de snapshot y el hash registrados en los manifiestos. La preparación actual separa conceptualmente junio como desarrollo y reserva el snapshot posterior para una evaluación final; esa separación no debe romperse para ajustar transformaciones o hiperparámetros.

La preparación existente genera artefactos de datos para la fase de desarrollo, pero no entrena ni carga un modelo de recomendación. Por lo tanto, los comandos anteriores no constituyen todavía una ejecución completa del recomendador.

Para que una futura ejecución sea reproducible se deben conservar, como mínimo, el snapshot exacto, su hash, la política de variables, la partición agrupada por `host_id`, la semilla, la configuración de hiperparámetros, el preprocesador y la versión del modelo. Los campos `id` y `host_id` no deben entrar como predictores.

## 5. Precio recomendado y cuantiles

La salida debe estar expresada en MXN y conservar claramente la diferencia entre una predicción y el precio que finalmente podría observarse en el mercado. Un precio recomendado sería un punto de referencia operacional obtenido a partir de las características del anuncio y del modelo elegido; no representa una garantía de demanda ni de venta.

Si la integración final define los cuantiles solicitados, los cuatro serán cuantiles de la distribución estimada del precio publicado, condicionada a las características del anuncio. No son intervalos de confianza sobre la media ni probabilidades de reserva. Su interpretación sería:

- **p10:** una referencia baja dentro de la distribución estimada de precios para anuncios comparables.
- **p35:** un punto operativo relativamente conservador que podría usarse como precio sugerido si el contrato final del equipo así lo establece.
- **p50:** la mediana estimada; divide la distribución prevista en dos partes iguales.
- **p90:** una referencia alta dentro de la distribución estimada.

Estos cuantiles describen incertidumbre del modelo bajo los datos y supuestos utilizados. No son probabilidades de conseguir una reserva, no son límites del mercado y no garantizan que el precio publicado sea pagado. Las comisiones, impuestos, descuentos, limpieza, estancias mínimas, disponibilidad, ocupación y condiciones de cada reserva pueden cambiar el resultado real.

El README propone comparar un baseline, una regresión regularizada y un modelo de cuantiles usando métricas como pinball, MAE y cobertura. En la rama revisada no hay código que produzca p10, p35, p50 o p90, ni una regla implementada que seleccione uno de ellos como precio recomendado.

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
- Dependencias y procedimiento de reproducción local con `uv`.

### Pendiente o no disponible en esta rama

- Entrenamiento integrado del baseline, Ridge sobre `log(precio)` y modelo LightGBM de cuantiles.
- Búsqueda reproducible de regularización y contrato común de predictores y particiones entre modelos.
- Calibración de residuos sin utilizar una validación externa.
- Artefacto serializado del modelo y del preprocesamiento.
- Módulo de recomendación que cargue el artefacto, valide entradas, maneje categorías desconocidas y devuelva p10, p35, p50 y p90 en MXN.
- Evaluación final sobre el snapshot reservado y comparación completa de modelos.
- API, interfaz web, contenedor, hospedaje y monitoreo productivo.

Las funciones pendientes deben integrarse y validarse con el contrato de datos antes de describirlas como disponibles. Si alguna de ellas corresponde a otro integrante, como Juan Pablo, el recomendador no puede considerarse completo hasta que sus componentes estén incorporados, probados y conectados con el flujo común. La sección de pendientes deberá actualizarse después de incorporar el PR de Ridge y recomendador de Víctor; mientras tanto, ese trabajo no debe describirse como integrado en `main` ni en esta rama. Este documento no sustituye esa integración ni presenta funcionalidades planificadas como resultados existentes.
