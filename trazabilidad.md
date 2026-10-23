# Trazabilidad de correcciones

Este libro integra en un solo informe el trabajo de los entregables 2 y 3 y el cierre del proyecto. Por eso esta tabla cumple dos funciones. La primera es mostrar cómo se resolvió cada observación o requisito pendiente del segundo entregable. La segunda es mostrar dónde está cada bloque que exige el entregable final (bloques A a G). Toda la evidencia se puede reproducir con los scripts de `scripts/` y los resultados guardados en `resultados/`.

```{admonition} Cómo leer esta tabla
:class: note
Las observaciones del segundo entregable que se listan aquí son las que el equipo recibió como retroalimentación y las que la guía del entregable pedía cerrar. Si el docente tiene alguna observación que no aparece en una fila, se agrega con el mismo formato.
```

## Bloque A. Observaciones del segundo entregable

| Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|
| Faltan 28 de los 140 modelos, los de regresión | No se ejecutaron. El desenlace es binario (dificultad cognitiva sí o no) y las métricas de error de regresión no tienen sentido sobre una clase. El documento del entregable final habla de la "tarea seleccionada" y da una ruta distinta para clasificación y otra para regresión (Cuadro 3), lo que respalda trabajar una sola. Esta decisión queda declarada como limitación y pendiente de confirmación del docente | Sección 5.1 y conclusiones | Tabla maestra con 108 combinaciones efectivas de 112 (KNN no admite pesos de clase) |
| Faltan matriz de confusión, curva ROC y Accuracy, Precision, Recall, F1 y AUC de cada modelo | Se calcularon para los ocho modelos en la prueba, con el umbral de 0,5 y el umbral de 0,245 | Secciones 5.7.1 y 5.7.2 | `resultados/test_metricas_todos_modelos.csv`, `resultados/figuras/confusion_roc_todos.png`, `scripts/metricas_todos.py` |
| Falta la tabla de espacios de búsqueda con justificación, y justificar los 5 y 3 folds | Tabla con cada hiperparámetro, rango y razón, y una sección con la justificación de 5 folds externos y 3 internos | Secciones 5.3.1 y 5.3.2 | `src/benchmark.py` (diccionarios `ESPACIOS` y `REJILLAS`), `resultados/hiperparametros_finales.json` |
| Diagrama de diferencia crítica de la comparación estadística | Diagrama de Nemenyi de los siete modelos, balanceos y optimizadores | Sección 5.5.1 | `resultados/figuras/cd_diagrama.png`, `resultados/friedman_por_factor.json` |
| Explicación local con SHAP | Un caso acertado y uno errado de XGBoost con cascada de SHAP | Sección 5.9.1 | `resultados/shap_local_casos.csv`, `resultados/figuras/shap_waterfall_*.png` |
| Comparar LIME con SHAP | 150 adultos de la prueba, acuerdo global y local, estabilidad de LIME | Sección 5.9.2 | `resultados/lime_vs_shap.json`, `scripts/lime_vs_shap.py` |
| Calibración y recalibración | Brier y ECE por modelo y balanceo, y recalibración de Platt e isotónica | Secciones 5.8 y 5.8.1 | `resultados/calibracion_por_modelo_balanceo.csv`, `resultados/recalibracion.csv` |
| Optimizadores y su comparación justa | Grid, Random, bayesiano (Optuna) y genético (DEAP) con el mismo presupuesto, semillas, diversidad genética y convergencia | Sección 5.6 | `resultados/semillas_optimizadores.csv`, `resultados/diversidad_genetica.csv`, `resultados/convergencia_resumen.csv` |
| Complejidad teórica frente a tiempo medido, con memoria | Exponentes empíricos de tiempo y memoria para cada modelo, estándar frente a optimizado | Sección 5.6.3 | `resultados/complejidad_exponentes.csv`, `resultados/complejidad_estandar_vs_optimizada.csv` |
| Evaluación final una sola vez, robustez y sin fuga | Los finalistas se evalúan una sola vez en la prueba, y se reportan robustez a faltantes, subgrupos y ruido | Secciones 5.7 y 5.10 | `resultados/test_final_metricas.csv`, `resultados/robustez.csv` |

Después de estas correcciones se recalculó todo lo que dependía de ellas. El ranking de modelos no cambió, lo que se discute en la sección 5.4, y las conclusiones del capítulo 10 usan los resultados actualizados.

## Bloques B a G del entregable final

| Requisito | Qué se hizo | Ubicación en el libro | Evidencia |
|---|---|---|---|
| B. Mejor combinación por modelo base, con media ± desviación | Tabla maestra y selección de la mejor combinación de cada uno de los siete modelos por PR-AUC media del bucle externo | Sección 5.4 | `resultados/tabla_maestra.csv`, `resultados/original_resumen.csv` |
| B. Reentrenamiento final, serialización, `metadata.json` y prueba de ida y vuelta | Pipeline completo reentrenado con todo el entrenamiento y guardado con joblib (compress=3), con metadatos y prueba que exige predicciones idénticas | Capítulo 9 | `artifacts/*.joblib`, `artifacts/*_metadata.json`, `tests/test_artefactos.py` |
| C. Modelo original, implementado por nosotros | Clase `AditivoPenalizado` en numpy, compatible con scikit-learn, con pseudocódigo, complejidad, pruebas unitarias, ablación e interpretabilidad | Capítulo 7 | `src/original_model.py`, `tests/test_original_model.py` |
| C. Revisión de literatura, al menos tres fuentes y cinco cadenas | Diez búsquedas en PubMed, Scopus AI y buscador web, con matriz de novedad y declaración de novedad acotada | Capítulo 6 | `literatura.md` |
| D. Al menos tres pruebas de hipótesis, con ajuste y tamaño del efecto | Friedman y Nemenyi, Wilcoxon, t corregida de Nadeau y Bengio, DeLong y McNemar, con Holm, Cliff, d de Cohen e IC BCa | Capítulo 8 | `resultados/original_pruebas_pareadas.csv`, `resultados/original_pruebas_prueba.csv`, `resultados/original_veredicto.json` |
| E. Dashboard con contexto, EDA y modelos | Dashboard con pestañas de contexto, EDA, modelos base, benchmark, optimización, resultado final y modelo original | Capítulo 9 | `dashboard_dcs/` |
| F. Despliegue público en Render | Servicio web con gunicorn, versión de Python fijada y URL pública | Capítulo 9 | `dashboard_dcs/Procfile`, `dashboard_dcs/runtime.txt`, `render.yaml` |
| G. Informe en Jupyter Book | Este libro, con sus capítulos, referencias y reproducibilidad | Todos | `_toc.yml`, `_config.yml`, `references.bib` |
| Declaración de autoría y uso de herramientas, incluida la IA | Declaración firmada por el equipo | Capítulo 11 | `reproducibilidad.md` |
