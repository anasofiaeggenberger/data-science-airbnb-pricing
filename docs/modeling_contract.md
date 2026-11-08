# Contrato de modelado de Víctor

Solo junio participa en desarrollo. Se conserva precio positivo, IDs únicos y host_id como texto. No se divide development_clean.csv: ya está imputado sobre todo junio.

`load_development()` devuelve X (predictores sin imputar), y (MXN), metadata (id, host_id). Las filas están ordenadas por id y comparten índice. `make_folds(metadata)` devuelve cinco pares de posiciones entrenamiento/validación mediante GroupKFold: ningún anfitrión aparece en ambos lados. No se modifica la definición temporal del test.

`RidgeQuantileModel(alpha=1, include_optional=False).fit(X, y, groups=metadata.host_id)` aprende únicamente del entrenamiento recibido. `predict(X)` devuelve un DataFrame con p10, p35, p50, p90 en MXN, conservando el índice. Es compatible con el predict del baseline de Pan, revisado en origin/feat/metrics-and-baseline. `predict_quantiles` es un alias.

La recomendación es p35; el intervalo p10-p90 tiene cobertura nominal 80%, no garantizada. Este cuantil representa una preferencia operativa por penalizar más la sobrevaloración, no un costo económico estimado. La comparación usa pinball de p35. Ridge utiliza cuantiles globales de residuos logarítmicos fuera de muestra del entrenamiento para aproximar los cuatro cuantiles; no es una regresión cuantílica condicional.

Se prueban alpha 0.1, 1, 10 y 100 en los mismos cinco folds. La tabla es validación para selección, no una estimación independiente tras la búsqueda. Septiembre queda reservado para Pan tras congelar la solución completa.

Predictores principales: ubicación, capacidad, baños, dormitorios, camas, conteo de amenidades, alcaldía y tipos de propiedad/alojamiento. minimum_nights, maximum_nights, host_has_profile_pic y host_identity_verified se agregan únicamente con include_optional=True. No hay recorte de precios ni eliminación de extremos en la especificación principal.

## Base del curso

- Clase 08, Regresión Lineal desde MLE: Ridge de sklearn y regularización de coeficientes.
- Clase 10, Feature Engineering: fit en entrenamiento, transform en validación, StandardScaler, Pipeline y ColumnTransformer.
- Clase 16, Selección de Modelos: búsqueda pequeña y preprocesamiento dentro de cada partición.
- GitHub/DataScience/HDT5/HDT5.ipynb, celda 14: comparar candidatos mediante validación común.
- GitHub/DataScience/Proyecto1/P1-Saravia.ipynb, celdas 20 y 24: Pipeline que aprende el escalador con entrenamiento.
- GitHub/DataScience/Proyecto2/P2_SaraviaVictor.ipynb, celdas 8, 12 y 15: Pipeline y búsqueda acotada de hiperparámetros.

Se adaptan esos patrones a regresión; no se copian sus métricas de clasificación. GroupKFold, codificación de categorías nuevas y calibración de residuos son extensiones necesarias para este problema y se explican en la guía de defensa.
- GitHub/DataScience/HDT3/HDT3.ipynb, celdas 22 y 24: regresión sobre precios, residuos y transformación logarítmica.
- Parciales 1 y 2: referencia de estilo e interpretación de métricas; no se reutilizan los clasificadores de esos ejercicios para regresión.
