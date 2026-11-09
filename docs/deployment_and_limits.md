# Despliegue y limitaciones del recomendador de precios

## 1. Objetivo y usuario previsto

El proyecto busca apoyar la definición de un precio diario en MXN para un anuncio nuevo de Airbnb en Ciudad de México. El usuario previsto es una persona anfitriona o integrante de un equipo que necesita una referencia inicial para publicar un alojamiento.

La recomendación debe entenderse como apoyo para tomar una decisión, no como una tasación ni como una promesa de resultado comercial. El precio final también requiere criterio humano y conocimiento de la propiedad, la zona, la temporada y las condiciones del mercado.

El estado revisado del repositorio incluye la infraestructura de datos, la auditoría, la preparación y la exploración descriptiva de junio. También incluye el baseline, Ridge sobre `log(precio)`, LightGBM cuantílico, el ejecutor común de modelos, el análisis por segmentos, el notebook de comparación y el recomendador local por línea de comandos. La validación de junio seleccionó LightGBM, la solución quedó congelada y la evaluación temporal de septiembre se ejecutó una sola vez. Sus resultados oficiales están en `docs/final_evaluation.json` y no deben recalcularse.

## 2. Flujo de uso implementado y previsto

El flujo local consume el artefacto serializado cuya ruta está registrada en `docs/frozen_solution.json`. La selección se hizo únicamente con junio, antes de evaluar septiembre; la evaluación temporal final sirve para medir generalización y no para reajustar el modelo.

1. La persona ingresa las características disponibles antes de publicar el anuncio. `src/recommend.py` exige actualmente `latitude`, `longitude`, `accommodates`, `neighbourhood_cleansed`, `property_type` y `room_type`. También puede recibir `bathrooms`, `bedrooms`, `beds`, `amenities_count`, `minimum_nights` y `maximum_nights`, que se validan si están presentes.
2. `validate_listing` comprueba campos obligatorios, tipos de texto y rangos numéricos razonables. `id` y `host_id` no forman parte de los predictores. Las categorías no observadas reciben el tratamiento definido por el preprocesamiento, en lugar de utilizarse como identificadores.
3. El preprocesamiento transforma numéricos y categorías, calcula `amenities_count` cuando corresponde y selecciona solo las columnas permitidas. Durante el entrenamiento, la imputación, el escalado y la codificación se ajustan dentro de cada partición de entrenamiento. Los folds comunes se agrupan por `host_id` y no incluyen `id` ni `host_id` como variables predictoras.
4. El baseline estima cuantiles por combinación de alcaldía y tipo de alojamiento. Ridge ajusta un modelo sobre `log(precio)` y calibra cuantiles mediante residuos fuera de muestra. LightGBM ajusta modelos cuantílicos para los niveles definidos por el contrato común. Los tres se comparan con las mismas variables, folds y métricas de desarrollo.
5. `src/recommend.py` carga un artefacto joblib confiable, valida que los cuatro cuantiles sean finitos, positivos y estén ordenados, y devuelve el precio p35, p10, p50 y p90 en MXN. La respuesta incluye una nota que aclara que son referencias de precios publicados y no garantías comerciales.

El código del recomendador puede cargar un artefacto de Ridge, baseline o LightGBM si fue serializado con el contrato esperado. En la entrega final, el artefacto seleccionado corresponde a LightGBM con `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31` y `min_child_samples=50`.

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

El repositorio permite ejecutar localmente la descarga, la auditoría, la preparación, la comparación de modelos, el análisis por segmentos y la inferencia del recomendador. No contiene una API, una interfaz web, un contenedor ni un servicio hospedado. Por eso, este documento describe una arquitectura conceptual y no afirma que exista un despliegue productivo.

La preparación de resultados de desarrollo y de artefactos locales se documenta con:

```bash
uv run python -m src.train_models --mode both
uv run python -m src.segments
uv run python -m src.evaluate_final freeze
```

El primer comando compara baseline, Ridge y LightGBM y guarda resultados de desarrollo; también produce las predicciones de validación que consume el análisis por segmentos. El segundo genera las tablas de segmentos a partir de esas predicciones. El tercero fue el procedimiento usado para congelar la configuración seleccionada y, salvo que se use `--skip-artifact`, entrena y serializa el artefacto correspondiente. No es necesario repetirlo para preservar la evaluación final ya realizada.

Después del congelamiento, el recomendador debe localizar el artefacto mediante la ruta registrada en `docs/frozen_solution.json`:

```bash
MODELO=$(uv run python -c "import json; print(json.load(open('docs/frozen_solution.json'))['artifact_path'])")
uv run python -m src.recommend --model "$MODELO" --listing examples/listing.json
```

Si el `.joblib` original no está disponible, se puede generar una copia sin sobrescribir los JSON
oficiales y sin volver a evaluar septiembre:

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

La copia nueva se entrena únicamente con junio y puede tener un SHA-256 distinto por diferencias de
entorno o serialización. `.gitignore` excluye `models/*`, `data/processed/*` y `*.joblib`, por lo
que el artefacto original y las predicciones CSV no forman parte del commit.

El comando de evaluación temporal fue independiente y se ejecutó una sola vez. No debe volver a ejecutarse:

```bash
uv run python -m src.evaluate_final run --confirm-frozen
```

El resultado oficial quedó guardado en `docs/final_evaluation.json`; no se debe usar `--allow-rerun`.

Una futura implementación podría empezar como una herramienta de línea de comandos o un proceso local y, después de validar el contrato y el modelo, envolverse en una API o interfaz. Esa decisión todavía no está implementada ni debe darse por hecha.

## 4. Ejecución local y reproducibilidad

Para reproducir el proyecto se requiere Python `3.12` y `uv`. Las dependencias declaradas están en `pyproject.toml` y las versiones resueltas en `uv.lock`; `joblib` y LightGBM forman parte de las dependencias directas del proyecto. En macOS, la instalación de `libomp` puede ser necesaria para ejecutar LightGBM.

Para reproducir el desarrollo sin volver a leer ni evaluar el snapshot temporal, se puede ejecutar:

```bash
uv sync --frozen
uv run python -m src.fetch_development
uv run pytest
uv lock --check
```

`src.fetch_development` verifica o descarga únicamente junio usando el manifiesto. La preparación histórica del proyecto pudo leer el snapshot reservado para construir el manifiesto temporal, pero no forma parte de la reproducción ordinaria del modelo final. No se debe ejecutar `src.evaluate_final run`, usar `--allow-rerun` ni descargar septiembre para reproducir el entrenamiento.

La preparación existente genera artefactos de datos para la fase de desarrollo. Después, `uv run python -m src.train_models --mode both` puede regenerar la comparación y las predicciones de validación de junio; para no sobrescribir resultados oficiales se recomienda pasar rutas temporales. `src.recommend` ejecuta la inferencia local sobre el artefacto cuya ruta está en el JSON congelado. El `.joblib` y los CSV derivados se reconstruyen porque están ignorados por Git.

Para que una ejecución futura sea reproducible se deben conservar, como mínimo, el snapshot exacto, su hash, la política de variables, la partición agrupada por `host_id`, la semilla, la configuración efectiva de hiperparámetros, el preprocesador, la calibración y la versión del modelo. Los campos `id` y `host_id` no deben entrar como predictores.

La suite automatizada integrada reportó 98 pruebas aprobadas en el estado informado del proyecto. Volver a ejecutar pruebas no implica volver a ejecutar la evaluación temporal; esta última debe conservarse como el resultado oficial de una sola corrida.

## 5. Precio recomendado y cuantiles

La salida debe estar expresada en MXN y conservar claramente la diferencia entre una predicción y el precio que finalmente podría observarse en el mercado. Un precio recomendado sería un punto de referencia operacional obtenido a partir de las características del anuncio y del modelo elegido; no representa una garantía de demanda ni de venta.

El contrato común de `src/metrics.py` define p10, p35, p50 y p90 como cuantiles de la distribución estimada del precio publicado, condicionada a las características del anuncio. No son intervalos de confianza sobre la media ni probabilidades de reserva. Su interpretación es:

- **p10:** una referencia baja dentro de la distribución estimada de precios para anuncios comparables.
- **p35:** el cuantil recomendado y un punto operativo relativamente conservador para el precio sugerido.
- **p50:** la mediana estimada; divide la distribución prevista en dos partes iguales.
- **p90:** una referencia alta dentro de la distribución estimada.

Estos cuantiles describen incertidumbre del modelo bajo los datos y supuestos utilizados. El intervalo p10–p90 tiene una cobertura nominal del 80 % en el contrato de métricas, pero esa cobertura no está garantizada para cada anuncio ni para el mercado futuro. No son probabilidades de conseguir una reserva, no son límites del mercado y no garantizan que el precio publicado sea pagado. Las comisiones, impuestos, descuentos, limpieza, estancias mínimas, disponibilidad, ocupación y condiciones de cada reserva pueden cambiar el resultado real.

El contrato común de `src/metrics.py` define pinball, MAE y cobertura para comparar baseline, Ridge y LightGBM. La selección de desarrollo se realiza con pinball loss p35; MAE p35 y cobertura p10–p90 son métricas secundarias. En los resultados de junio disponibles, el resumen comparativo es:

| Modelo | Pinball p35 | MAE p35 (MXN) | Cobertura p10–p90 |
|---|---:|---:|---:|
| Baseline | 648.76 | 1,736.50 | 79.4 % |
| Ridge | 706.94 | 1,722.08 | 80.2 % |
| LightGBM | 545.12 | 1,433.27 | 74.1 % |

LightGBM obtuvo el menor pinball p35 y el menor MAE p35 en la validación de junio disponible, por lo que fue seleccionado antes de mirar septiembre. Su cobertura p10–p90 en junio fue 74.1 %, inferior a la cobertura nominal del 80 %. La solución quedó congelada con la configuración registrada en `docs/frozen_solution.json`.

El contrato común también revisa el orden de los cuantiles y registra los cruces de cuantiles de LightGBM antes y después de la corrección. El análisis por bandas de precio observado es descriptivo y post hoc; no se usa para elegir variables, ajustar el modelo ni seleccionar la solución.

### Evaluación temporal final

La evaluación se ejecutó una sola vez sobre 2,604 anuncios con `id` ausente en junio y precio válido. Esta regla identifica anuncios nuevos respecto del snapshot, no anfitriones completamente nuevos: 1,493 anuncios correspondieron a anfitriones que ya tenían al menos un anuncio en junio. No se afirma que esos anuncios fueran necesariamente publicados por primera vez en septiembre. Sus resultados oficiales son:

| Métrica | Resultado |
|---|---:|
| Pinball p35 | 327.8444 |
| MAE p35 | 786.0543 MXN |
| Cobertura p10–p90 | 79.6467 % |
| Amplitud media p10–p90 | 3,364.2481 MXN |
| Cruce de cuantiles después del ordenamiento | 0 % |

La cobertura global quedó ligeramente por debajo del 80 % nominal. También varió por segmento: las habitaciones privadas tuvieron 76.0793 % de cobertura con 857 anuncios, y Q1, el cuartil de precios más bajos, tuvo 64.6697 % con 651 anuncios. Q2 alcanzó 80.0307 %, Q3 92.3547 % y Q4 81.4815 %. El segmento de habitaciones de hotel tuvo un solo anuncio y no permite una conclusión estable.

Estas diferencias son evidencia de generalización en un snapshot temporal concreto, no garantías para cada alojamiento ni para el mercado futuro. La evaluación no observa reservas, ocupación, ingresos ni rentabilidad.

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
3. Mantener separado el snapshot evaluado y conservar `docs/final_evaluation.json` como resultado histórico de una sola ejecución. No usarlo repetidamente para elegir variables, imputaciones, umbrales, hiperparámetros o calibración.
4. Reentrenar con `uv run python -m src.train_models --mode both` y regenerar segmentos con `uv run python -m src.segments`, usando únicamente el protocolo de desarrollo documentado. Las transformaciones deben ajustarse dentro de cada partición de entrenamiento y los folds deben ser reproducibles y estar agrupados por `host_id`.
5. Revisar el resultado comparativo, congelar explícitamente la configuración seleccionada y conservar el artefacto junto con sus metadatos efectivos. Si se regenera el `.joblib`, hacerlo fuera de los archivos oficiales y no repetir la evaluación temporal.
6. Registrar versión de código, datos, modelo, semilla, configuración, métricas y fecha de entrenamiento.
7. Comparar el modelo nuevo con el baseline y revisar pinball, MAE y cobertura en la escala correspondiente. También conviene revisar errores por alcaldía y tipo de alojamiento, sin convertir esos resultados en garantías de desempeño futuro.
8. Definir una señal de alerta para deriva de datos o deterioro de errores y establecer cuándo se debe volver a auditar o entrenar.

## 9. Precauciones de uso

El recomendador debe presentarse como una herramienta informativa para apoyar la decisión de publicación. No debe afirmar ni garantizar:

- que habrá reservas, ocupación o un determinado número de noches vendidas;
- que se obtendrán ingresos, rentabilidad o retorno de inversión;
- que el precio sugerido será el precio de mercado o el precio finalmente pagado;
- que el modelo conoce las condiciones particulares de una propiedad que no estén en los datos.

También debe evitarse ingresar o utilizar información personal innecesaria, identificadores como `id` o `host_id`, métricas de desempeño posteriores a la publicación y variables expresamente prohibidas por la política de features. Cuando falte una característica relevante o el anuncio esté fuera de la cobertura del modelo, la aplicación futura debería informarlo en lugar de ocultar la incertidumbre.

## 10. Estado actual y pendientes

### Implementado en el estado revisado

- Descarga desde enlaces oficiales de Inside Airbnb y conservación de metadatos de procedencia y hash.
- Auditoría básica de archivos, columnas, identificadores y precios.
- Preparación de un conjunto de desarrollo basado en junio y documentación de limpieza.
- Política de variables con exclusiones explícitas, incluyendo `id`, `host_id` y variables posteriores a la publicación.
- Manifiesto de separación temporal y evaluación final documentada sobre el snapshot de septiembre.
- Notebook de exploración descriptiva de junio, notebook de comparación de modelos y pruebas automatizadas para las utilidades existentes.
- Contrato común de cuantiles y métricas en `src/metrics.py`, baseline por alcaldía y tipo de alojamiento, Ridge sobre `log(precio)` y LightGBM cuantílico.
- Ejecutor común que compara los tres modelos con los mismos predictores, folds agrupados por `host_id` y selección por pinball p35.
- Búsqueda de hiperparámetros, sensibilidad y análisis por segmentos de las predicciones de validación de junio. Las bandas de precio observado se interpretan post hoc.
- Solución congelada de LightGBM con modelo, preprocesamiento, metadatos, ruta y SHA-256 del artefacto, junto con las barreras de `src/evaluate_final.py` para impedir una segunda evaluación temporal.
- Recomendador local en `src/recommend.py`, con validación de entradas y resultados p10, p35, p50 y p90 en MXN.
- Dependencias y procedimiento de reproducción local con `uv`.

### Pendiente o no ejecutado

- No volver a ejecutar la evaluación temporal ni reajustar el modelo usando sus resultados.
- Si se pierde el `.joblib`, regenerar una copia fuera de los archivos oficiales usando únicamente junio; el SHA-256 de una nueva serialización no tiene que coincidir con el original.
- API, interfaz web, contenedor, hospedaje y monitoreo productivo.

LightGBM fue seleccionado por el menor pinball p35 de junio y después se evaluó una sola vez sobre septiembre. El resultado temporal no autoriza reajustar ni volver a ejecutar el test; solo informa las limitaciones de generalización documentadas arriba.
