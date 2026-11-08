# Coordinación de implementación

Este archivo reúne el estado técnico y las tareas por componente. La documentación de presentación y los comentarios del código deben usar nombres de componentes, sin nombres personales ni mensajes dirigidos a integrantes.

## Preprocesamiento, Ridge e inferencia: implementados

- `src/modeling_data.py`: carga de junio, separación de predictores/objetivo/metadatos y cinco folds por anfitrión.
- `src/preprocessing.py`: imputación, escalado y codificación ajustados dentro del entrenamiento.
- `src/ridge_model.py`: Ridge sobre log(precio), con calibración de residuos internos fuera de muestra.
- `src/train_ridge.py`: búsqueda de alpha y guardado del candidato Ridge. No sustituye la comparación entre familias.
- `src/recommend.py`: inferencia para cualquier modelo compatible con el contrato.
- `notebooks/03_ridge_recommender.ipynb`: explicación y resultados de desarrollo.
- `docs/reproducibility.md`: comandos y ejemplos de integración.

La búsqueda inicial seleccionó alpha=100 entre 0.1, 1, 10 y 100. Los resultados de junio están en `docs/ridge_validation.json`; no existe evaluación final de septiembre en esta rama.

## Modelos de referencia, boosting y evaluación: integración pendiente

La rama `feat/metrics-and-baseline` contiene `src/metrics.py` y `src/baseline_model.py`. Su interfaz se comprobó contra esta implementación. Mantener esas implementaciones como fuente común; no duplicarlas ni reemplazarlas para integrar Ridge.

Tareas del agente responsable de este componente:

1. Usar `load_development()` para obtener X, y y metadata. Reutilizar los mismos pares de índices de `make_folds(metadata)` para baseline, Ridge y LightGBM.
2. Ajustar Ridge con `fit(X.iloc[train], y.iloc[train], groups=metadata.host_id.iloc[train])`. Obtener predicciones mediante `predict(X.iloc[valid])`. No pasar id/host_id como predictores.
3. Implementar LightGBM para p10, p35, p50 y p90. Ajustar sus categorías o transformaciones solo con entrenamiento. Declarar explícitamente si usa variables opcionales.
4. Implementar el ejecutor común en `src/train_models.py`. Evaluar en MXN mediante `evaluate_quantile_predictions`, guardar métricas por fold y predicciones fuera de muestra. Seleccionar mediante pinball de p35.
5. Ejecutar las sensibilidades de variables opcionales y extremos solamente en junio. Los objetivos de validación deben permanecer en su escala original, sin recortes.
6. Comparar las familias de modelos, documentar la selección y congelar configuración, transformaciones y reglas antes de acceder al test.
7. Implementar `src/evaluate_final.py` y comprobar su funcionamiento con datos sintéticos. Una vez congelada la solución, realizar una única evaluación del test temporal y guardar sus resultados generales y por segmentos.
8. Guardar el modelo seleccionado, con su preprocesamiento, en un artefacto compatible con el recomendador. Puede ser el modelo directamente o un diccionario con claves `model` y `metadata`; actualizar su estado de candidato cuando exista una selección global documentada.

## Interpretación y presentación: pendientes de integración

El agente de documentación debe consumir resultados guardados por el ejecutor común, elaborar la comparación y el análisis de errores, explicar límites y monitoreo, y preparar los materiales de presentación. No duplicar entrenamiento dentro de un notebook de resultados ni atribuir conclusiones a métricas no calculadas.

## Reglas metodológicas compartidas

- Junio es desarrollo. Septiembre solo contiene el test de IDs nuevos respecto a junio, con precio válido; su definición no se modifica.
- No dividir `development_clean.csv` para validar: ya contiene estadísticas aprendidas sobre todo junio.
- Usar los snapshots y hashes registrados. La descarga y el empaquetado de bytes no autorizan explorar o evaluar el test durante desarrollo.
- Mantener p35 como recomendación y p10-p90 como rango nominal del 80%. La cobertura real debe medirse; no está garantizada por la interfaz.
- Mantener la política de variables y aprender todas las transformaciones dentro del entrenamiento.
- Documentar uso de AI, verificación y limitaciones con lenguaje formal. Las guías personales de estudio se conservan fuera del repositorio.
- Ejecutar las pruebas correspondientes y `git diff --check` antes de integrar cambios.
