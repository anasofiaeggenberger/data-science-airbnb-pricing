# Entender y defender la parte de Víctor

## Recorrido del código

1. modeling_data.py lee junio y convierte cada fila sin calcular promedios. Conserva id/host_id como metadatos; no son predictores.
2. make_folds separa por anfitrión. Si un anfitrión tiene diez anuncios, los diez quedan del mismo lado de cada partición.
3. preprocessing.py rellena números con medianas de entrenamiento, los escala y convierte categorías a indicadores. Categorías raras y nuevas tienen tratamiento definido por OneHotEncoder.
4. ridge_model.py aprende log(precio) con regularización. Alpha controla cuánto penalizamos coeficientes grandes.
5. train_ridge.py compara cuatro alphas usando exactamente cinco folds. Selecciona por pinball de p35 y guarda resultados.
6. recommend.py carga un modelo ya entrenado, valida una entrada y devuelve el precio/rango. Nunca vuelve a entrenar ni abre el test.

## Relación con tus ejercicios

La clase 08 usaba Ridge(alpha=1.0) para estabilizar coeficientes. La clase 10 ajustaba StandardScaler solo con entrenamiento. Proyecto 1 construía un Pipeline de escalador y kNN; aquí el último paso es Ridge. HDT5 y Proyecto 2 comparaban candidatos con la misma validación. Aquí se reemplaza la métrica de clasificación por pinball y se agrupan anfitriones.

En HDT3, celdas 22 y 24, ajustaste regresión de precios de diamantes, calculaste residuos y transformaste precios a logaritmos. Aquí se reutilizan esas ideas: los residuos son diferencias entre log(precio) y predicción. Se calcularon fuera de muestra para evitar que el propio ajuste haga parecer demasiado estrecho el rango.

Los parciales se revisaron como referencia de estilo. El Parcial 1 usa operaciones de pandas y modelos sencillos; el Parcial 2 conecta métricas con una decisión y explica por qué un número aislado puede engañar. Sus clasificadores de dígitos no se trasladaron a este problema de regresión.

## La parte nueva: aproximar cuantiles

Ridge normalmente da un solo número. Dentro del entrenamiento hacemos tres rondas internas: entrenar con dos partes y predecir la tercera. Reunimos esos errores logarítmicos y obtenemos sus percentiles 10, 35, 50 y 90. Después ajustamos Ridge con todo el entrenamiento recibido.

Para predecir: `precio_q = exp(prediccion_log + residuo_q)`. Un ejemplo conceptual: si Ridge da log(1000) y el residuo p35 es log(0.9), recomendamos 900. No sumamos 0.9 pesos: el efecto de la escala logarítmica es multiplicativo.

Es una aproximación con la misma distribución residual para todos los anuncios. No garantiza 80% de cobertura en cada zona ni es conformal. LightGBM cuantílico puede representar una dispersión que depende de las características. Los cuatro residuos están ordenados, por lo que los cuantiles de Ridge no se cruzan.

## Preguntas que debes poder responder

- ¿Por qué log(precio)? Modela cambios relativos y reduce influencia de la cola durante el ajuste. Se reportan errores contra precios originales en MXN.
- ¿Por qué no usar el CSV limpio? Sus imputaciones vieron todo junio, incluyendo lo que luego sería validación.
- ¿Por qué escalar? Ridge penaliza magnitudes de coeficientes; las unidades distintas afectan esa penalización.
- ¿Por qué p35? Sobrevalorar tiene peso 0.65 y subvalorar 0.35 en pinball. Es una preferencia declarada, no una estimación del costo real de perder una reserva.
- ¿Por qué no ajustar después del test? El test dejaría de medir generalización independiente y se volvería otra validación.
- ¿Ridge ganó? Ganó un alpha dentro de Ridge; falta la selección entre familias de modelos de Pan.
- ¿El precio maximiza ingresos? No. El objetivo observado es precio publicado; no tenemos ocupación y precios efectivamente pagados para demostrar ese efecto.
- ¿El intervalo es confianza sobre una media? No: intenta describir un rango de precios individuales publicados.

## Resultado de esta ejecución

Junio contiene 29,637 objetivos válidos. La búsqueda inicial alpha=[0.1,1,10,100] seleccionó 100 con pinball medio p35 de 706.94 MXN. La desviación entre folds es aproximadamente 352.24 MXN: hay variabilidad considerable. El mejor valor está en el borde superior de una búsqueda pequeña; esto es una limitación, no evidencia de un óptimo global. Pan puede ampliar la búsqueda solo con junio antes de congelar la solución.

La entrada sintética de examples/listing.json produce una recomendación aproximada de 1,320.86 MXN y rango 885.35-3,443.93 MXN con este candidato. Es una demostración del flujo, no una validación de precisión para esa propiedad.
