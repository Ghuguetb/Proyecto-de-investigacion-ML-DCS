# Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia

Proyecto de investigación de Machine Learning
Maestría en Ingeniería Biomédica, Universidad del Norte · Santiago Díaz, Gina Huguet y Leiry Mares · Octubre de 2026

## Resumen

La dificultad cognitiva subjetiva (DCS), es decir la percepción de la propia persona de que tiene
problemas para recordar o concentrarse, es una señal temprana asociada a cerca del doble de riesgo
de demencia (Pike et al., Neuropsychol Rev, 2022). La importancia de identificar estas
manifestaciones de manera temprana radica en que una proporción considerable de los casos de
demencia está relacionada con factores de riesgo potencialmente modificables, la Comisión Lancet
estima que alrededor del 45 % de los casos podrían atribuirse a estos factores.

Esto plantea la posibilidad de utilizar información sobre dichos factores para identificar, incluso
antes de que exista un diagnóstico de demencia, a personas que presentan dificultades cognitivas
subjetivas. A estos factores se suma el antecedente de accidente cerebrovascular (ACV), una
condición neurológica y vascular asociada con un mayor riesgo de deterioro cognitivo. Por ello, este
proyecto busca evaluar si la combinación de estos factores permite predecir la presencia de DCS en
adultos, mediante un modelo de machine learning, utilizando información de 56.129 participantes de
la National Health Interview Survey (NHIS) 2022–2023 de los CDC.

```{admonition} Pregunta de investigación
:class: tip
¿Cuál es el desempeño de un modelo de clasificación basado en factores de riesgo potencialmente
modificables y antecedente de accidente cerebrovascular para predecir dificultad cognitiva
subjetiva en adultos?
```

Es un problema de clasificación binaria supervisada. Una regresión logística con splines para la
edad alcanza en el conjunto de prueba una PR-AUC de 0,538 [IC 95 %: 0,517-0,561], frente a 0,206 de
la línea base trivial, y un ROC-AUC de 0,804 [0,794-0,814], con buena calibración y sin
sobreajuste. Para saber si ese desempeño moderado es un límite de la información disponible o del
modelo, lo comparamos con un benchmark de 112 combinaciones de siete modelos, cuatro técnicas de
balanceo y cuatro optimizadores de hiperparámetros, con validación cruzada anidada y sin fuga de
información, y con un modelo original, el Explainable Boosting Machine (EBM). El resultado es que
ninguna combinación supera de forma significativa a la regresión logística, que el balanceo con
remuestreo daña la calibración sin mejorar la discriminación, y que las variables que más pesan son
las mismas en SHAP, en los coeficientes y en el EBM, con la depresión y la edad a la cabeza. El techo
parece estar en la información de las 14 predictoras y no en la complejidad del modelo.

## Cómo se organizó el trabajo

```{figure} images/flujo_proyecto.png
:name: fig-flujo-proyecto
:width: 620px

Flujo general del proyecto, de la base de datos al benchmark de modelos, capítulo por capítulo.
```

## Estructura del informe

| Capítulo | Contenido |
|---|---|
| 1. Base de datos | Problema, justificación, fuente y licencia, diccionario, estructura, reserva del conjunto de prueba, tamaño de muestra, calidad de datos y ética |
| 2. Análisis exploratorio | Variable objetivo, análisis uni, bi y multivariado, y aplicabilidad de los componentes temporal y espacial |
| 3. Fuga de datos y preprocesamiento | Auditoría de fuga y pipeline de preprocesamiento |
| 4. Modelo base | Línea base trivial, regresión logística, métricas con IC bootstrap, calibración, residuos, curva de aprendizaje y coeficientes |
| 5. Benchmark de modelos | Diseño sin fuga, 108 combinaciones evaluadas, modelo original (EBM), estadística jerárquica, optimización y complejidad, resultado final, calibración, SHAP y LIME, y robustez |
| Conclusiones | Hallazgos, limitaciones y próximos pasos |
| Referencias | Bibliografía del proyecto |

Además del libro hay un dashboard interactivo (carpeta `dashboard_dcs/`) con seis pestañas, que
recorre el mismo camino. Las tres primeras presentan el contexto, el EDA con filtros y el modelo
base, con un simulador de riesgo, y las tres últimas muestran el benchmark, la optimización y el
costo computacional, y el resultado final con interpretabilidad y robustez.

## Reproducibilidad

- Datos: `data/raw/adult22csv.zip` y `data/raw/adult23csv.zip`, descargados de
  <https://www.cdc.gov/nchs/nhis/documentation/2022-nhis.html> y
  <https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html>. El notebook 1 genera los datos
  procesados en `data/processed/`.
- Código: `src/utils.py`, con la carga, la recodificación según el codebook y la partición,
  `src/preprocesamiento.py`, con el pipeline de scikit-learn, `src/benchmark.py`, con los modelos,
  los balanceos, los optimizadores y la validación anidada, y `src/estadistica.py`, con DeLong y
  Holm. Los scripts de `scripts/` ejecutan cada etapa del benchmark.
- Dependencias: `requirements.txt` para los capítulos 1 a 4 (Python 3.12) y
  `requirements-benchmark.txt` para el capítulo 5 (Python 3.13), con todas las versiones fijadas.
- Semilla aleatoria: `SEED = 42`, usada en la partición, la validación cruzada, el bootstrap y
  todos los algoritmos con componente aleatorio. En el benchmark, la búsqueda de cada fold externo
  usa 42 más el índice del fold.
- Resultados del benchmark: `resultados/tabla_maestra.csv` resume las 108 combinaciones con sus
  hiperparámetros finales, métricas, tiempo y semilla, y `resultados/tareas/` guarda cada una de las
  540 tareas.
- Orden de ejecución: notebooks 01 a 04 en ese orden. Para el capítulo 5, `scripts/run_benchmark.py`,
  `scripts/run_ebm.py`, `scripts/analisis.py`, `scripts/finales.py`, `scripts/complejidad.py`,
  `scripts/lime_vs_shap.py` y `scripts/cierre_guia.py`, y luego el notebook 05. Después `jupyter-book build .` genera este
  informe en `_build/html/index.html`.
