# Reglas de LightGBM cuantílico

Proyecto: `data-science-airbnb-pricing` — Fase 3.

Implementación: [`src/lightgbm_model.py`](../src/lightgbm_model.py). Pruebas: `tests/test_lightgbm_model.py`.
Contrato de cuantiles: [`docs/metrics_contract.md`](metrics_contract.md).

## Estructura

Se entrena **un modelo por cuantil** con `objective="quantile"` y `alpha` igual al nivel: 0.10, 0.35,
0.50 y 0.90. El objetivo es el precio en MXN sin transformar: la pérdida pinball ya es la pérdida que
corresponde a cada cuantil, así que no hace falta pasar por logaritmos ni volver después.

`fit(X, y, groups=None)` acepta `groups` por uniformidad con Ridge y lo ignora: este modelo no
necesita calibración interna porque estima cada cuantil directamente.

## Predictores

Se usan las columnas de `feature_columns(include_optional)` de `src/modeling_data.py`, las mismas que
consume Ridge. `include_optional=True` agrega `minimum_nights`, `maximum_nights`,
`host_has_profile_pic` y `host_identity_verified`, que es la sensibilidad de variables opcionales.

Los predictores son los mismos; la **codificación** no. Ridge necesita one-hot y escalado; los árboles
trabajan mejor con categorías nativas y no necesitan escala. Esa diferencia es deliberada: lo que debe
mantenerse idéntico entre modelos son las columnas de entrada y las particiones, no la representación
interna que cada familia requiere.

## Categorías y faltantes

Las categóricas se codifican como tipo `category` de pandas y se declaran a LightGBM como tales.

| Caso | Tratamiento |
|---|---|
| Categoría con al menos `min_category_count` filas de entrenamiento | Se conserva |
| Categoría rara en entrenamiento | Se agrupa en `Other` |
| Categoría desconocida al predecir | Se agrupa en `Other` |
| Nulo o cadena vacía | Etiqueta `Missing`, sujeta al mismo umbral |

`min_category_count = 20`, el mismo umbral que usan `src/preparation.py` y el baseline.

Las numéricas **no se imputan**: LightGBM maneja los faltantes de forma nativa y aprende hacia qué
lado enviarlos. Imputar con la mediana antes de un árbol agrega una suposición que el modelo no
necesita.

## Cruces de cuantiles

Entrenar cuatro modelos independientes permite que un cuantil alto quede por debajo de uno bajo en
filas concretas. El tratamiento, decidido durante desarrollo y no después de ver resultados:

- `predict_raw()` devuelve las predicciones sin corregir.
- `crossing_rate()` mide la proporción de filas afectadas.
- `predict()` ordena los cuatro valores de cada fila de menor a mayor y devuelve el resultado monótono.

Reordenar es el tratamiento más conservador disponible: no cambia el conjunto de valores predichos
para esa fila, solo su asignación a los niveles. Sobre junio la tasa medida antes de corregir es de
**5.5 %** de las filas de validación.

## Hiperparámetros

Configuración por defecto: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`,
`min_child_samples=20`, con `random_state=42`.

Búsqueda acotada de 12 configuraciones en `PARAM_GRID`: `num_leaves` en 15, 31 y 63;
`learning_rate` en 0.05 y 0.10; `min_child_samples` en 20 y 50. La ejecuta y la registra el ejecutor
de experimentos, sobre las mismas cinco particiones y seleccionando por pinball de `p35`.

## Comprobación sobre junio

Ejecución de verificación con los cinco folds de `make_folds()`, hiperparámetros por defecto y sin
búsqueda, sobre las 29,637 filas con precio válido:

| | baseline | ridge (α=100) | lightgbm |
|---|---:|---:|---:|
| pinball p35 | 648.8 | 706.9 | **547.3** |
| MAE recomendado | 1736.5 | 1722.1 | 1435.7 |
| cobertura | 0.794 | 0.802 | 0.743 |
| amplitud media | 3827.5 | 3661.3 | 2961.5 |
| desviación entre folds | 102.6 | 352.2 | 97.8 |

LightGBM gana en el criterio de selección con intervalos bastante más angostos, pero **su cobertura
queda por debajo del 80 % nominal**: gana amplitud a costa de cubrir menos. Esa es la tensión que la
interpretación debe discutir, no un detalle menor.

**Estas cifras no son el resultado de desarrollo.** Son una comprobación de que el modelo corre sobre
los folds comunes. La comparación válida, con búsqueda de hiperparámetros y sensibilidades, sale del
ejecutor de experimentos.
