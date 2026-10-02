# Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia

Primer entregable del proyecto de investigación de Machine Learning
Maestría en Ingeniería Biomédica, Universidad del Norte · Santiago Díaz, Gina Huguet y Leiry Mares · Septiembre de 2026

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

Es un problema de clasificación binaria supervisada (ruta A de la guía). Una regresión logística
con splines para la edad alcanza en el conjunto de prueba una PR-AUC de 0,538 [IC 95 %:
0,517-0,561], frente a 0,206 de la línea base trivial, y un ROC-AUC de 0,804 [0,794-0,814], con
buena calibración y sin sobreajuste. El desempeño es real pero moderado, lo que deja margen de
mejora para modelos más complejos en las siguientes entregas.

## Cómo se organizó el trabajo

```{figure} images/flujo_proyecto.png
:name: fig-flujo-proyecto
:width: 620px

Flujo general del proyecto: de la base de datos al modelo base, capítulo por capítulo.
```

## Estructura del informe

| Capítulo | Contenido | Sección de la guía |
|---|---|---|
| 1. Base de datos | Problema, justificación, fuente y licencia, diccionario, estructura, reserva del conjunto de prueba, tamaño de muestra, calidad de datos, ética | 1 |
| 2. Análisis exploratorio | Variable objetivo, análisis uni, bi y multivariado, aplicabilidad de componentes temporal y espacial | 2.1-2.4, 2.6-2.8 |
| 3. Fuga de datos y preprocesamiento | Auditoría de fuga, pipeline de preprocesamiento | 2.5, 2.9 |
| 4. Modelo base | Línea base trivial, regresión logística, métricas con IC bootstrap, calibración, residuos, curva de aprendizaje, coeficientes | 3 |
| Conclusiones | Hallazgos, limitaciones y próximos pasos | |

## Reproducibilidad

- Datos: `data/raw/adult22csv.zip` y `data/raw/adult23csv.zip`, descargados de
  <https://www.cdc.gov/nchs/nhis/documentation/2022-nhis.html> y
  <https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html>. El notebook 1 genera los datos
  procesados en `data/processed/`.
- Código: `src/utils.py`, con la carga, la recodificación según el codebook y la partición, y
  `src/preprocesamiento.py`, con el pipeline de scikit-learn.
- Dependencias: `requirements.txt`, con Python 3.12 y todas las versiones fijadas.
- Semilla aleatoria: `SEED = 42`, usada en la partición, la validación cruzada, el bootstrap y
  todos los algoritmos con componente aleatorio.
- Orden de ejecución: notebooks 01, 02, 03 y 04 en ese orden. Luego `jupyter-book build .` genera
  este informe en `_build/html/index.html`.
