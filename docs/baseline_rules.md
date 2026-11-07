# Reglas del baseline

Proyecto: `data-science-airbnb-pricing` — Fase 3.

Implementación: [`src/baseline_model.py`](../src/baseline_model.py). Pruebas: `tests/test_baseline_model.py`.
Contrato de cuantiles: [`docs/metrics_contract.md`](metrics_contract.md).

El baseline conserva la mediana por alcaldía y tipo de alojamiento exigida en la propuesta, y la
extiende a los cuatro cuantiles del contrato para que compita con Ridge y LightGBM bajo las mismas
métricas. `predict()["p50"]` es exactamente la mediana del grupo.

## Agrupación

Grupo principal: `neighbourhood_cleansed` × `room_type`. Ambas columnas están marcadas como
`Permitida` en [`docs/feature_policy.md`](feature_policy.md) y existen antes de publicar.

Las categorías se normalizan a texto sin espacios sobrantes; nulos y cadenas vacías se etiquetan como
`Missing` y se tratan como una categoría más, igual que en la preparación de junio.

## Tamaño mínimo de grupo

`min_group_size = 20`, el mismo umbral que `min_category_count` usa en `src/preparation.py` para
agrupar categorías raras. Con menos de 20 anuncios, los cuantiles extremos de un grupo se apoyan en
dos o tres observaciones y describen ruido, no el mercado de esa zona.

El umbral es un parámetro del constructor y se fija antes de comparar modelos; no se ajusta mirando
resultados de validación.

## Cadena de respaldo

| Nivel | Cuándo aplica | Fuente de los cuantiles |
|---|---|---|
| `group` | El par alcaldía × tipo tiene al menos `min_group_size` filas de entrenamiento | Ese grupo |
| `fallback` | El par no existe o es pequeño, pero el `room_type` sí alcanza el mínimo | Todas las filas de ese `room_type` |
| `global` | El `room_type` es desconocido o tampoco alcanza el mínimo | Todas las filas de entrenamiento |

`predict_with_source()` devuelve el nivel usado y el tamaño del grupo de origen en cada fila, para que
la evaluación pueda separar las predicciones que dependieron de un respaldo.

## Cuantiles y cruces

Los cuantiles son empíricos, calculados con `numpy.quantile` e interpolación lineal. Como las cuatro
cifras de una fila salen siempre de la misma muestra, son monótonas por construcción: este modelo no
puede producir cuantiles cruzados, y las pruebas lo verifican.

## Ajuste y fugas

Todas las tablas (grupos, respaldos y distribución global) se calculan dentro de `fit`, únicamente con
las filas de entrenamiento que recibe. `predict` no recalcula nada. Una partición nueva exige un
`fit` nuevo. El modelo no usa `id` ni `host_id`, y el snapshot de septiembre no participa.

El objetivo debe entregarse en MXN y estrictamente positivo; `fit` rechaza lo contrario.

## Comprobación sobre junio

Ejecución local de verificación, con una partición provisional agrupada por `host_id` (80/20, sin
anfitriones compartidos) sobre las 29,637 filas de junio con precio válido:

- 23,345 filas de entrenamiento, 6,292 de validación.
- 32 grupos alcanzaron el mínimo; 42 filas de validación usaron el respaldo por `room_type`; ninguna
  necesitó la distribución global.
- Cobertura 0.787 contra 0.80 nominal; amplitud media 3,879 MXN; cruces de cuantiles 0.

**Estas cifras no son un resultado de desarrollo.** Son una comprobación de que el código corre sobre
datos reales. La comparación válida usa las cinco particiones comunes y se ejecuta desde el ejecutor
de experimentos.
