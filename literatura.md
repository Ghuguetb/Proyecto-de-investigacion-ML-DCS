# 6. Revisión de literatura y novedad

Este capítulo documenta cómo se buscó evidencia sobre el modelo original del proyecto y qué se puede afirmar a partir de ella. El modelo original es el **aditivo penalizado por rondas**, programado por nosotros con numpy (capítulo 7). En el capítulo 5 evaluamos el EBM de la librería `interpret`, pero como se importa con una línea no cumple la condición de implementación propia, así que quedó como candidato interpretable.

## 6.1 Protocolo de búsqueda

La pregunta de la revisión es si ya existe un modelo aditivo penalizado, ajustado por rondas y con suavizado entre niveles ordenados, aplicado a la predicción de dificultad cognitiva subjetiva (DCS) con datos de encuesta poblacional y factores de riesgo modificables.

| Fuente | Cómo se consultó | Fecha |
|---|---|---|
| PubMed | Búsqueda directa con sintaxis de PubMed (cinco cadenas, más abajo) | 23 de octubre de 2026 |
| Scopus AI | Dos búsquedas del equipo, con 111 y 72 registros exportados, ocho títulos en común (sección 5.2) | Antes de este capítulo, la fecha no quedó registrada en los archivos exportados |
| Buscador web general | Tres consultas que devuelven arXiv, medRxiv, PubMed Central, revistas y repositorios institucionales (se usó como sustituto de Google Scholar) | 23 de octubre de 2026 |

Dos fuentes que se intentaron y no se pudieron leer, y que por eso **no cuentan** como consultadas. La API de arXiv rechazó la consulta automática por su archivo `robots.txt`, y la API de Europe PMC devolvió error. Tampoco se pudo abrir el texto completo de un artículo de PubMed Central (se pidió una verificación humana), así que para ese trabajo solo se usó el título y el fragmento que muestra el buscador.

## 6.2 Cadenas de búsqueda y matriz de novedad

Cada cadena combina el nombre o la familia del modelo, el problema de aplicación y, en una de ellas, el español. "Revisados" son los registros cuyo título y resumen se leyeron. "Relevantes" son los que predicen DCS o deterioro cognitivo con datos de encuesta, factores de riesgo o variables tabulares usando un modelo de machine learning, para compararlos con el nuestro.

| N.º | Base | Cadena de búsqueda | Result. | Revis. | Relev. | Observación |
|---|---|---|---|---|---|---|
| 1 | PubMed | "subjective cognitive decline" AND ("machine learning" OR "prediction model") AND ("National Health Interview Survey" OR BRFSS OR survey) | 8 | 8 | 2 | Un modelo logístico con nomograma en adultos mayores de Anhui {cite}`liu2025` y un agrupamiento no supervisado con BRFSS 2024 {cite}`wang2026`. Ninguno usa un modelo aditivo |
| 2 | PubMed | "subjective cognitive decline" AND ("explainable boosting machine" OR "generalized additive model") | 2 | 2 | 0 | Un GAM para asociación entre sueño y quejas cognitivas {cite}`lin2022`, no para predecir, y un marco interpretable con imágenes de Alzheimer |
| 3 | PubMed | ("penalized additive model" OR "cyclic boosting" OR "backfitting") AND ("cognitive impairment" OR dementia) AND (survey OR "risk factors") | 10 | 10 | 0 | Resultados sobre resonancia, metabolómica, aleatorización mendeliana y costos. Ninguno aplica el método a DCS |
| 4 | PubMed | "declive cognitivo subjetivo" AND "aprendizaje automático" | 0 | 0 | 0 | Sin resultados en español |
| 5 | PubMed | "subjective cognitive decline" AND "modifiable risk factors" AND ("machine learning" OR "random forest" OR "XGBoost" OR "logistic regression") | 2 | 2 | 0 | Dos estudios con regresión logística ({cite}`voyer2023` con BRFSS, {cite}`allen2026` en Nueva Zelanda) |
| 6 | Buscador web | explainable boosting machine OR additive model "subjective cognitive decline" NHIS survey prediction | 9 | 9 | 1 | Interpretabilidad en envejecimiento cognitivo sin DCS, y un trabajo de Alzheimer con imágenes |
| 7 | Buscador web | arxiv "subjective cognitive decline" tabular survey data interpretable additive boosting | 9 | 9 | 3 | Factores de DCS con aprendizaje automático automatizado, detección temprana con autorreporte, y regresión regularizada frente a árboles en HRS (2025) |
| 8 | Buscador web | medrxiv OR scholar "subjective cognitive decline" "National Health Interview Survey" machine learning modifiable risk factors dementia | 9 | 9 | 2 | Redes bayesianas dinámicas para pronosticar DCS y AutoML para factores de DCS |
| 9 | Scopus AI | Búsqueda 1, modelos de ML para DCS con encuestas poblacionales | 111 | 111 | 1 | Las redes con atención de Akter y colaboradores {cite}`akter2025`, con BRFSS 2021 y prueba externa en NHIS 2022 |
| 10 | Scopus AI | Búsqueda 2, modelos mejorados para datos tabulares categóricos | 72 | 72 | 0 | Ensambles, TabNet y transformers tabulares, sin DCS |

Los números de las filas 6 a 8 son los de la primera página de resultados del buscador y no el total de la web, así que la cobertura es parcial. Esa es una limitación real de esta revisión.

## 6.3 Trabajos más cercanos y en qué se diferencia el modelo propuesto

| Trabajo o familia | Qué hace | Qué conserva nuestro modelo | Qué cambia |
|---|---|---|---|
| Regresión logística con nomograma (Liu 2025), logística con BRFSS (Voyer 2023) | Modelo lineal en el logit para asociar factores con DCS | La salida es una probabilidad y cada variable aporta de forma separable | Las curvas por variable no son lineales, la edad y los niveles ordenados pueden tener forma libre |
| GAM con splines (Hastie y Tibshirani, 1990; Lin 2022 para sueño y DCS) | Curvas suaves por variable, ajustadas por backfitting | La estructura aditiva y el suavizado entre niveles vecinos | Ajuste por rondas con paso derivado de una cota cuadrática de la pérdida logística, con garantía de no aumento del objetivo |
| P-splines (Eilers y Marx, 1996) | Penalización de diferencias entre coeficientes vecinos | La penalización de primera diferencia para variables ordenadas | Se aplica por niveles, también a categorías ordenadas como educación o frecuencia de depresión |
| EBM {cite}`lou2013` | Boosting cíclico con árboles pequeños y bagging, más interacciones por pares | La idea de aprender una curva por variable en rondas cíclicas | Nuestro paso es un sistema lineal por variable en vez de árboles, y no incluye interacciones ni bagging |
| Redes con atención y costo sensible (Akter 2025) | Red convolucional para DCS con BRFSS y NHIS | Misma pregunta y misma familia de datos | Modelo mucho más simple y legible, sin redes neuronales |
| Regresión regularizada frente a árboles en HRS (2025) | Compara elastic net con árboles para función cognitiva | Resultado parecido, los modelos simples compiten bien | Añadimos una estructura no lineal por variable |
| Redes bayesianas dinámicas y AutoML para DCS | Pronóstico con redes bayesianas y selección automática de modelos | Misma enfermedad como desenlace | Un solo modelo con forma conocida y verificable |

## 6.4 Declaración de novedad

Al mejor de nuestro conocimiento y según las búsquedas descritas, no encontramos estudios que apliquen un modelo aditivo penalizado, ajustado por rondas con una cota cuadrática y con suavizado entre niveles ordenados, a la predicción de dificultad cognitiva subjetiva con datos de la NHIS.

Esto es una afirmación sobre las búsquedas que hicimos, no una prueba de que no exista. Y es una novedad **de aplicación y de implementación**, no de mecanismo. Las piezas del método existen desde hace décadas (modelos aditivos, penalización de diferencias, boosting por coordenadas y optimización por cotas cuadráticas, ver {cite}`hastie1990,eilers1996,friedman2001,bohning1988`). Lo propio de este trabajo es la implementación en numpy, la combinación para este problema y la verificación de sus propiedades con pruebas unitarias, y no un algoritmo nuevo. En el capítulo 10 se discute cómo eso afecta la condición R1.
