# Contrato de cuantiles y métricas

Proyecto: `data-science-airbnb-pricing` — Fase 3.

Este documento fija la configuración con la que se comparan baseline, Ridge y LightGBM. Debe quedar
confirmada por los tres integrantes antes de ejecutar la comparación; no se deben comparar pérdidas
calculadas con cuantiles distintos para declarar un ganador.

Implementación: [`src/metrics.py`](../src/metrics.py). Pruebas: `tests/test_metrics.py`.

## Cuantiles

| Nombre | Cuantil | Uso |
|---|---:|---|
| `p10` | 0.10 | Extremo inferior del rango |
| `p35` | 0.35 | Precio recomendado |
| `p50` | 0.50 | Mediana de referencia |
| `p90` | 0.90 | Extremo superior del rango |

Cobertura nominal del intervalo `p10`–`p90`: **80 %**.

El 0.35 es una decisión operativa conservadora: un anfitrión nuevo sin reseñas prefiere un precio algo
por debajo de la mediana del mercado comparable. No representa un costo económico demostrado ni una
estimación de ingresos, y debe presentarse como tal en la defensa.

## Criterio de selección

La solución se elige por **pinball loss del cuantil recomendado** (`p35`), promediada sobre las mismas
particiones de validación para todos los modelos. El módulo expone esa cifra como `pinball_recommended`
para que ningún modelo se declare ganador con una métrica distinta.

## Métricas reportadas por partición

| Clave | Definición |
|---|---|
| `n_observations` | Filas evaluadas en la partición |
| `pinball_p10`, `pinball_p35`, `pinball_p50`, `pinball_p90` | Pérdida pinball media en cada cuantil |
| `pinball_recommended` | Igual a `pinball_p35`; criterio de selección |
| `mae_recommended` | MAE del precio recomendado (`p35`) |
| `mae_median` | MAE de la mediana predicha (`p50`) |
| `coverage` | Proporción de precios reales dentro de `[p10, p90]`, extremos incluidos |
| `nominal_coverage` | 0.80 |
| `coverage_gap` | `coverage` menos cobertura nominal |
| `mean_interval_width` | Amplitud media de `p90 - p10` en MXN |
| `quantile_crossing_rate` | Proporción de filas con cuantiles no monótonos |

La cobertura se interpreta junto con la amplitud: un intervalo muy ancho puede cubrir casi todo y aun
así ser inútil para decidir un precio.

`summarize_fold_metrics` resume las particiones en media, desviación, mínimo y máximo.

## Escala

Todas las métricas monetarias se calculan en **MXN**, contra el precio original de la partición de
validación, aunque un modelo entrene sobre `log(precio)` o sobre un objetivo recortado. La conversión
de vuelta a MXN es responsabilidad del modelo, no de las métricas.

## Validaciones del módulo

- Rechaza nulos y valores no finitos en objetivo y predicciones; no descarta filas difíciles en silencio.
- Rechaza longitudes distintas entre objetivo y predicciones.
- Rechaza objetivos cero o negativos, que no son precios válidos según la regla del proyecto.
- Exige las cuatro columnas `p10`, `p35`, `p50`, `p90` del contrato de modelos.

Límite conocido: la validación de positividad no detecta un objetivo entregado en escala logarítmica,
porque `log(precio)` también es positivo. Pasar el precio en MXN es responsabilidad de quien llama.
