# Bitácora de colaboración con AI: modelado

Fecha: 8 de noviembre de 2026. Registro de la sesión de implementación; la revisión humana de las decisiones y su comprensión deben completarse antes de la presentación.

Se solicitó asistencia para implementar el preprocesamiento, Ridge y la inferencia a partir de los ejemplos de clase, hojas de trabajo y proyectos.

Se usaron patrones de Ridge (clase 08), separación antes de escalar (clase 10), pipelines y selección (clase 16, HDT5, Proyectos 1 y 2), logaritmos y residuos (HDT3). Se revisaron ambos parciales para el estilo y la interpretación de métricas. Se agregó calibración de residuos fuera de muestra y agrupación por anfitrión porque el problema lo requiere; esas extensiones no se atribuyen como código copiado del curso.

Se verificaron hashes de los snapshots, separación de anfitriones, conservación de IDs largos, imputación solo en entrenamiento, categorías desconocidas, cuantiles ordenados, serialización, validación de entradas, ejecución del entrenamiento y pruebas automatizadas. Se comprobó el contrato predict de la rama de métricas y baseline. El notebook y los comandos reproducibles documentan la evidencia.

Se descartó partir el CSV ya imputado, usar reseñas o host_id como predictores, escoger opciones con septiembre, presentar Ridge como ganador global o construir una aplicación web innecesaria. No se generaron resultados del test ni se documentó como realizada una revisión humana pendiente.
