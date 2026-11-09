# Protocolo de experimentos

Proyecto: `data-science-airbnb-pricing` — Fase 3.

> **Estado del cierre.** Este protocolo conserva el procedimiento que se acordó antes de mirar el test
> temporal. El procedimiento se aplicó: la selección se hizo con junio, se congeló LightGBM y el test
> de septiembre se ejecutó una sola vez. El resultado actual está en [`docs/final_evaluation.json`](final_evaluation.json)
> y la configuración en [`docs/frozen_solution.json`](frozen_solution.json). Los comandos de evaluación
> que aparecen abajo son históricos y no forman parte de la reproducción ordinaria.

Implementación: [`src/train_models.py`](../src/train_models.py), [`src/segments.py`](../src/segments.py)
y [`src/evaluate_final.py`](../src/evaluate_final.py).
Contrato de cuantiles: [`docs/metrics_contract.md`](metrics_contract.md).

## Qué garantiza el ejecutor

1. **Una sola definición de particiones.** Las cinco particiones salen de `make_folds()` y se usan
   para los tres modelos sin excepción. Cada registro guarda `fold_assignment_sha256`, la misma
   huella que calcula la validación de Ridge: si dos corridas comparten ese hash, compararon sobre
   exactamente las mismas filas.
2. **Una sola implementación por modelo.** El ejecutor consume el `RidgeQuantileModel` de Víctor; no
   existe una segunda versión de Ridge en el repositorio.
3. **Una sola definición de métricas.** Todo se mide con `src/metrics.py`, en MXN y contra el precio
   original de validación.
4. **Repetible sin copiar nada a mano.** Un comando regenera la comparación completa, su registro y
   las predicciones de validación.

## Búsqueda de hiperparámetros

| Modelo | Configuraciones | Espacio |
|---|---:|---|
| baseline | 1 | `min_group_size = 20`, fijo y documentado de antemano |
| ridge | 4 | `alpha` en 0.1, 1, 10, 100 |
| lightgbm | 12 | `num_leaves` × `learning_rate` × `min_child_samples` |

La selección es por **media del pinball del cuantil recomendado** entre particiones. El baseline no
se ajusta por validación a propósito: su umbral se fijó por el mismo criterio que la limpieza de
junio, y moverlo mirando resultados lo convertiría en otro modelo ajustado.

Esa tabla es validación usada para elegir, **no una estimación independiente**. La estimación
independiente es el test temporal, una sola vez.

## Sensibilidades

Se cambia una decisión a la vez respecto de la especificación principal, con la configuración ya
seleccionada de cada modelo:

| Variante | Qué cambia |
|---|---|
| `base` | Especificación principal: variables opcionales excluidas, sin tope de precios |
| `optional_features_included` | Agrega `minimum_nights`, `maximum_nights`, `host_has_profile_pic`, `host_identity_verified` |
| `extreme_prices_capped_p99` | Aplica un tope p99 al objetivo **de entrenamiento** |

El tope se calcula dentro del entrenamiento de cada partición, nunca sobre junio completo ni sobre
el test. **No se elimina ninguna fila de validación**: las métricas siempre se calculan contra el
precio original, aunque el entrenamiento haya visto un objetivo recortado. Una prueba verifica que
el objetivo original no se modifica en el camino.

## Evidencia por segmentos

`src/segments.py` produce tablas por alcaldía, tipo de alojamiento y banda de precio. Cada fila lleva
su **tamaño de muestra** y una marca `enough_evidence`, falsa por debajo de 30 observaciones, para que
la interpretación no saque conclusiones de grupos diminutos.

**Las bandas de precio son análisis descriptivo posterior a la predicción, nunca variables
predictoras ni criterio de selección.** Se construyen con cuartiles del precio real observado, que es
el objetivo: usarlas como entrada del modelo sería una fuga directa, y usarlas para elegir modelo o
hiperparámetros sería seleccionar mirando el objetivo. Solo sirven para responder dónde falla más el
modelo una vez que ya predijo, y por eso se calculan después de evaluar, sobre las mismas filas de
validación.

Las tres columnas de segmento se agrupan igual: ninguna entra al modelo. `neighbourhood_cleansed` y
`room_type` sí son predictores permitidos, pero en estas tablas se usan solo para cortar resultados.

Esas tablas son insumo para el análisis de errores; este documento no las interpreta.

## Congelamiento y evaluación final

El orden no es negociable:

1. Se ejecuta la comparación y las sensibilidades **solo con junio**.
2. El equipo acuerda la solución y se guarda con `evaluate_final.py freeze`, que escribe
   `docs/frozen_solution.json` con modelo, hiperparámetros, variables, cuantiles, huella de
   particiones y SHA-256 del snapshot de junio.
3. Sofi revisa que la selección esté documentada antes del test.
4. Recién entonces se ejecuta `evaluate_final.py run --confirm-frozen`.

El evaluador tiene varios candados, y ninguno vive solo en la línea de comandos:

- `run_final_evaluation()` exige `confirm_frozen=True` en la propia llamada de Python, y es estricto:
  un valor meramente parecido a verdadero no sirve. Una llamada no autorizada corta antes de abrir
  ningún archivo, así que no puede leer septiembre por accidente.
- Los requisitos del congelamiento se comprueban antes de tocar disco: configuración construible por
  el código actual, no evaluada antes, y contrato de cuantiles igual al de `src/metrics.py`.
- Sin `--confirm-frozen` la CLI no toca septiembre y termina con código 1.
- Sin `docs/frozen_solution.json` no corre.
- Si el SHA-256 de junio no coincide con el congelado, falla en vez de evaluar sobre otro desarrollo.
- Si ya existe `docs/final_evaluation.json`, se niega a repetirse salvo `--allow-rerun` explícito.

El artefacto del ganador guarda además sus **parámetros efectivos**, no solo los declarados: semilla,
particiones de calibración, umbral de categorías, `n_jobs`, los hiperparámetros reales del estimador
y las columnas predictoras. Sin eso, reconstruir el entrenamiento dependería de que los valores por
defecto del código nunca cambien.

El test se reconstruye con la regla original, la misma del manifiesto: IDs presentes en septiembre,
ausentes en junio y con precio válido. La regla identifica anuncios nuevos respecto del snapshot, no
anfitriones completamente nuevos. En la evaluación oficial, 1,493 de los 2,604 anuncios pertenecían
a anfitriones que ya tenían al menos un anuncio en junio; por eso no se afirma que fueran publicaciones
por primera vez en septiembre. Se verificó con snapshots sintéticos antes de usar los reales.

Después de ver septiembre no se reajusta nada. Pan fue quien ejecutó esa evaluación; los demás
consumen `docs/final_evaluation.json`. No se debe volver a ejecutar `run` ni usar `--allow-rerun`.

## Comandos

```bash
uv run python -m src.train_models --mode both      # comparación, búsqueda y sensibilidades
uv run python -m src.segments                      # tablas por segmento
uv run python -m src.evaluate_final freeze         # congelar la solución elegida
uv run python -m src.evaluate_final run --confirm-frozen   # una sola vez, tras el congelamiento
```

En macOS, LightGBM necesita el runtime de OpenMP: `brew install libomp`. `uv sync` no lo instala
porque es dependencia de sistema.

## Salidas

| Archivo | Contenido | En Git |
|---|---|---|
| `docs/model_comparison.json` | Configuraciones evaluadas, métricas por partición, resumen y ganador | Sí |
| `docs/sensitivity_results.json` | Las tres variantes por modelo | Sí |
| `docs/segment_tables.md` | Tablas legibles para la interpretación | Sí |
| `docs/frozen_solution.json` | Solución congelada antes del test | Sí |
| `docs/final_evaluation.json` | Resultado de la evaluación final | Sí |
| `data/processed/validation_predictions.csv` | Predicciones de validación con `id`, `host_id`, fold y modelo | No, regenerable |
| `data/processed/test_predictions.csv` | Predicciones del test con trazabilidad | No, regenerable |
