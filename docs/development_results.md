# Resultados de desarrollo

Proyecto: `data-science-airbnb-pricing` — Fase 3.

Todo lo que sigue sale **únicamente de junio**. El snapshot de septiembre no se abrió, no se midió y no
participó en ninguna decisión. Procedimiento en [`docs/experiment_protocol.md`](experiment_protocol.md).

Registros completos: [`docs/model_comparison.json`](model_comparison.json) y
[`docs/sensitivity_results.json`](sensitivity_results.json).

- Filas de desarrollo: 29,637 con precio válido, 13,095 anfitriones.
- Particiones: cinco, agrupadas por anfitrión.
- Huella de particiones: `0bc17e53368075dfcc00702ad0e9b2061e04d670b93fb1004b062b7eb5a27508`.
  Coincide con la de `docs/ridge_validation.json`, así que los tres modelos y la validación de Ridge
  compararon sobre exactamente las mismas filas.

La comparación se regeneró después de integrar `main` con los PR #2 y #3 fusionados, para incorporar
la nueva medición de cruces antes de ordenar. Las demás cifras salieron **idénticas** a la corrida
previa, hasta el sexto decimal, y la huella de particiones no cambió. Era lo esperado: las
correcciones posteriores de Víctor no tocaron `modeling_data.py`, `preprocessing.py` ni
`ridge_model.py`, así que los modelos son los mismos. Esa coincidencia vale como comprobación de
reproducibilidad.

## Comparación

Medias entre las cinco particiones, en MXN contra el precio original de validación.

| Modelo | pinball p35 | desv. entre folds | MAE recomendado | MAE mediana | cobertura | amplitud media | cruces antes | cruces después |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 648.76 | 102.59 | 1736.50 | 1693.40 | 0.794 | 3827.52 | 0.0 % | 0.0 % |
| ridge (α=100) | 706.94 | 352.24 | 1722.08 | 1714.44 | 0.802 | 3661.32 | 0.0 % | 0.0 % |
| **lightgbm** | **545.12** | **91.54** | 1433.27 | 1410.61 | 0.741 | 3119.13 | **4.98 %** | 0.0 % |

Las dos últimas columnas son el porcentaje de filas con cuantiles cruzados antes y después del paso
de ordenamiento. El baseline y Ridge no ordenan nada: sus cuantiles no pueden cruzarse por
construcción, así que ambas columnas son cero. LightGBM entrena un modelo por cuantil de forma
independiente, y en el 4.98 % de las filas de validación al menos un par sale fuera de orden; el
contrato las reordena y después del paso no queda ninguna.

Configuración elegida para LightGBM: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`,
`min_child_samples=50`.

Tres cosas que conviene leer juntas:

1. **LightGBM gana el criterio de selección** por 16 % sobre el baseline y 23 % sobre Ridge, y además
   es el más estable entre particiones.
2. **Su cobertura es la peor: 0.741 contra 0.80 nominal.** Entrega intervalos 19 % más angostos que
   el baseline, pero cubre menos de lo que promete. Cobertura y amplitud hay que mirarlas juntas: el
   baseline cubre casi exactamente lo nominal a costa de un rango bastante más ancho.
3. **Ridge es el menos estable**, con una desviación entre particiones de 352 contra ~100 de los
   otros dos. Su intervalo sale de cuantiles globales de residuos, no de una regresión cuantílica
   condicional, y eso se nota.

El baseline no es un rival débil: le gana a Ridge en el criterio de selección. Que una mediana por
alcaldía y tipo de alojamiento supere a una regresión regularizada es un resultado honesto y hay que
decirlo.

## Búsqueda de hiperparámetros

Las 12 configuraciones de LightGBM quedaron muy juntas: entre la mejor (545.12) y la cuarta (546.09)
hay menos de un punto de pinball. La búsqueda no es lo que explica la diferencia con los otros
modelos; el tipo de modelo sí. `min_child_samples=50` aparece en las cuatro mejores.

Ridge eligió α=100, el extremo de la grilla, el mismo valor que encontró la validación independiente
de Víctor.

## Sensibilidades

Se cambia una decisión a la vez respecto de la especificación principal.

| Variante | baseline | ridge | lightgbm |
|---|---:|---:|---:|
| base | 648.76 | 706.94 | 545.12 |
| variables opcionales incluidas | 648.76 (0 %) | **falla** | 542.32 (−0.5 %) |
| tope p99 en el entrenamiento | 648.76 (0 %) | 612.45 (**−13.4 %**) | 544.93 (−0.0 %) |

**Variables opcionales: no conviene incluirlas.** A LightGBM le dan medio punto porcentual, al
baseline nada, y a Ridge lo rompen. La especificación principal las deja fuera.

**Tope de precios extremos: le importa mucho a Ridge y casi nada a los demás.** Ridge mejora 13.4 %
porque una regresión sobre el logaritmo es sensible a objetivos extremos; la pérdida cuantílica de
LightGBM ya es robusta a eso, y los cuantiles empíricos del baseline ni se enteran, porque p10 a p90
quedan todos por debajo del p99 recortado. Aun con el tope, Ridge (612.45) no alcanza a LightGBM
(545.12): **la sensibilidad no cambia al ganador.**

El tope se aprende dentro del entrenamiento de cada partición y no se elimina ninguna fila de
validación. Las métricas se calculan siempre contra el precio original.

### El fallo de Ridge con variables opcionales

No es un error de ejecución que haya que reintentar; es un defecto reproducible y vale documentarlo.

`maximum_nights` usa `2147483647` como centinela de "sin límite". En la partición 3 ese valor aparece
en 9 filas de validación y en **ninguna** de entrenamiento, donde el máximo es 9,999 con desviación
484. El `StandardScaler` ajustado en entrenamiento convierte ese centinela en unos 4.4 millones de
desviaciones, el centro en escala logarítmica cae a −113,900, y `exp()` devuelve cero exacto, que
dispara la validación de rango de `RidgeQuantileModel.predict`.

El ejecutor registra la variante como fallida con su causa y sigue con el resto, en vez de abortar
toda la corrida. Corregirlo requeriría acotar `maximum_nights` dentro del entrenamiento, y eso es
decisión del dueño de `src/ridge_model.py` y `src/preprocessing.py`, no de este componente.

## Evidencia por segmentos

Tablas completas con tamaños de muestra en [`docs/segment_tables.md`](segment_tables.md). Para
LightGBM:

| Dónde funciona peor | n | pinball p35 | cobertura |
|---|---:|---:|---:|
| Tlalpan | 1,064 | 1090.72 | 0.71 |
| Miguel Hidalgo | 4,540 | 627.11 | 0.74 |
| Coyoacán | 2,334 | 624.61 | 0.72 |
| Hotel room | 83 | 1928.75 | 0.43 |

Por banda de precio, los cuartiles del precio observado:

| Banda | n | mediana MXN | pinball p35 | cobertura |
|---|---:|---:|---:|---:|
| Q1 | 7,417 | 731 | 153.25 | 0.64 |
| Q2 | 7,403 | 1,400 | 175.12 | 0.82 |
| Q3 | 7,494 | 2,221 | 241.25 | 0.87 |
| Q4 | 7,323 | 4,409 | 1627.03 | 0.62 |

El error se concentra en los precios altos: el cuartil más caro tiene un pinball diez veces mayor que
el más barato y la peor cobertura junto con Q1. Los extremos de la distribución son donde el modelo
menos sirve.

`Milpa Alta` (n=25) queda marcada con `enough_evidence = false` en los tres modelos: cualquier
conclusión sobre esa alcaldía descansa en muy poca evidencia.

## Modelo ganador y artefacto

`evaluate_final freeze` guarda la configuración ganadora, la reentrena con las 29,637 filas de junio
y serializa `{'model': ..., 'metadata': ...}` en `models/final_<modelo>.joblib`, en el formato que ya
consume `src/recommend.py`. Antes de guardar comprueba sobre desarrollo que los cuantiles salgan
positivos y ordenados.

La metadata del artefacto viaja con el modelo: configuración, política de variables, contrato de
cuantiles, métricas de desarrollo, huella de particiones, SHA-256 de junio y `test_evaluated: false`.

Ese paso es anterior e independiente de la evaluación de septiembre: produce el modelo que se va a
evaluar, no la evaluación. El artefacto queda fuera de Git y se reconstruye con el comando.

**Al momento de escribir esto la solución no ha sido congelada**, porque congelar es una decisión del
equipo y Sofi debe revisar antes que la selección esté documentada. El flujo sí se verificó de punta
a punta sobre junio real, escribiendo en rutas temporales: entrenó con 29,637 filas, serializó, y el
recomendador cargó el artefacto y devolvió los cuatro cuantiles en MXN, positivos y ordenados.

## Qué es y qué no es esto

Esta es **validación usada para seleccionar configuración**, no una estimación independiente del
desempeño futuro. Los hiperparámetros se eligieron mirando estas mismas particiones, así que estos
números están optimistamente sesgados a favor del ganador.

La estimación independiente es el test temporal, una sola vez, después de congelar. Al momento de
escribir esto, **septiembre no ha sido evaluado**.
