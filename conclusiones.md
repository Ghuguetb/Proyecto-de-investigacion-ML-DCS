# 10. Conclusiones

## Respuesta a la pregunta de investigación

Los factores de riesgo modificables de demencia y el antecedente de ACV contienen información real pero limitada para identificar a los adultos con DCS. Una regresión logística con esas variables alcanza en datos no vistos una PR-AUC de 0,538 (2,6 veces la referencia trivial de 0,206) y un ROC-AUC de 0,804, detecta el 63,7 % de los casos con una precisión del 46,1 % y entrega probabilidades bien calibradas. El benchmark del capítulo 5 muestra que ese valor es un techo y no un punto de partida. Ninguno de los siete modelos, de las cuatro técnicas de balanceo ni de los cuatro optimizadores lo supera de forma estadísticamente significativa, ni siquiera el modelo original que programamos (aditivo penalizado por rondas, PR-AUC de 0,540 en la prueba), ni el EBM, que llega a 0,544 y es interpretable de forma nativa. El problema es aprendible, pero lo que limita el desempeño es la información de las 14 predictoras y no la complejidad del modelo.

## Hallazgos principales

1. La variable objetivo es frecuente y estable, con el 20,6 % de los adultos reportando alguna dificultad para recordar o concentrarse, sin cambios entre 2022 y 2023.
2. La edad tiene una relación en forma de U, la dificultad es más frecuente en adultos jóvenes y en mayores que en la mediana edad, lo que obligó a modelarla con splines.
3. La salud mental es el factor más fuerte. Los síntomas depresivos muestran un gradiente dosis-respuesta, con un OR ajustado de 4,9 para síntomas diarios frente a nunca.
4. Las pérdidas sensoriales pesan casi tanto como la depresión diagnosticada, con OR ajustados de 2,1 a 2,6, en línea con lo que señala la Comisión Lancet.
5. El ACV es un factor independiente, con un OR crudo de 3,36, ajustado por edad de 2,52 y ajustado por todas las variables de 1,69. Su efecto es mucho mayor en adultos jóvenes que en mayores de 75 años.
6. La escolaridad es protectora, con un gradiente claro (OR de 0,53 para posgrado frente a menos que secundaria), coherente con la hipótesis de la reserva cognitiva.
7. La hipertensión pierde su asociación una vez que se ajusta por edad y por los demás factores.
8. Con 108 combinaciones evaluadas con validación anidada, más el EBM y el modelo original, el desempeño se estanca en una PR-AUC cercana a 0,52 en validación y 0,54 en la prueba. La regresión logística, la SVM lineal, XGBoost, el EBM y el modelo original no se distinguen entre sí (DeLong con corrección de Holm).
9. El balanceo con SMOTE, ADASYN o pesos de clase sube el recall de 0,25 a cerca de 0,6 con el umbral de 0,5, pero no mejora la discriminación y deteriora la calibración, con un error de calibración que pasa de cerca de 0,02 a más de 0,2 en la logística.
10. SHAP sobre XGBoost, los coeficientes de la logística y el EBM señalan las mismas variables, con la frecuencia de síntomas depresivos, el diagnóstico de depresión y la edad al frente. LIME coincide en lo grueso (Spearman global de 0,82), pero es menos estable que SHAP.
11. El modelo original (clase `AditivoPenalizado`, solo numpy) iguala a los mejores en discriminación, con una PR-AUC de 0,520 en 25 estimaciones pareadas frente a 0,520 de la SVM, 0,520 de la logística y 0,521 del EBM. Es significativamente mejor que KNN, árbol, Bayes ingenuo y bosque aleatorio, pero ninguna de las cinco pruebas lo declara superior a todos, así que no cumple la condición de superioridad (R3). Tiene un ECE bajo (0,012 en la prueba), SHAP exacto y entrena en menos de dos segundos.
12. El modelo es razonablemente robusto. Pierde poco con ruido leve, con datos faltantes y al entrenar con 2022 y probar con 2023, pero discrimina peor en mayores de 74 años (ROC-AUC de 0,73).

## Cumplimiento de las condiciones del modelo original

| Condición | Resultado | Evidencia |
|---|---|---|
| R1. Novedad de la aplicación | Parcial. No se encontró el método aplicado a DCS con la NHIS en las búsquedas hechas, pero las piezas del método son conocidas | Capítulo 6 |
| R2. Implementación propia, sin importar el algoritmo | Cumplida. Clase `AditivoPenalizado` en numpy, con ocho pruebas unitarias | Capítulo 7, `src/original_model.py` |
| R3. Superioridad significativa con al menos tres pruebas | No cumplida. Empata con los mejores y ninguna prueba lo declara superior a todos | Capítulo 8 |

Con esta lectura, la nota máxima que se puede esperar por el componente del modelo original es la del nivel intermedio de la rúbrica del docente, y así se declara de forma explícita.

## Decisiones metodológicas clave

- Reserva del conjunto de prueba antes de tomar cualquier decisión basada en los datos.
- Exclusión de la pregunta de seguimiento de cognición, que por sí sola alcanzaba un AUC de 0,929, ya que sin esta auditoría el modelo habría superado artificialmente el umbral de 0,90.
- Uso de la categoría de IMC en lugar del IMC numérico, por el faltante de tipo MNAR de este último.
- Descarte del accuracy como métrica de decisión, ya que el modelo trivial obtiene 0,794 sin detectar un solo caso.
- Umbral de decisión de 0,246 en el modelo base (capítulo 4) y de 0,245 en los finalistas (capítulo 5), elegido con predicciones out-of-fold del entrenamiento.
- Validación cruzada anidada con todo el preprocesamiento y el remuestreo dentro de cada fold, y conjunto de prueba usado una sola vez, solo con los finalistas.
- Comparación estadística jerárquica (Friedman, Nemenyi por factor y DeLong con Holm), declarando que con cinco folds el Nemenyi global no tiene potencia.
- Para comparar el modelo original se repitió la validación anidada cinco veces (25 estimaciones pareadas) y se usaron cinco pruebas con corrección de Holm, con el criterio de decisión fijado antes de ver los resultados.

## Limitaciones

- Diseño transversal, las asociaciones no son causales, y la relación entre depresión y DCS probablemente es bidireccional.
- Autorreporte, tanto el objetivo como los diagnósticos dependen de la percepción y del acceso previo a servicios de salud.
- Cobertura parcial del marco Lancet, con 9 de los 14 factores disponibles en ambos años, y faltan alcohol, inactividad física, aislamiento social, traumatismo craneoencefálico y contaminación del aire.
- Representatividad, la muestra sobrerrepresenta a los mayores de 65 años, excluye a la población institucionalizada y a quienes necesitaron un informante, y el modelo no usa pesos muestrales. Los intervalos de confianza no incorporan el diseño muestral complejo de la NHIS. Ponderar la evaluación cambia poco la discriminación, pero no reemplaza un análisis con ese diseño.
- Presupuesto de búsqueda, los hiperparámetros se buscaron sobre submuestras de 8.000 filas con unas 12 evaluaciones por búsqueda, y la genética hizo menos evaluaciones (6,9), así que su comparación con los demás optimizadores no es del todo justa.
- Las 28 combinaciones de regresión del diseño original no se ejecutaron, porque con un desenlace binario las métricas y los supuestos de regresión no aplican. La regresión logística, con penalización L1 y L2, cubre ese papel.
- La brecha de literatura sobre modelos aditivos penalizados aplicados a DCS con datos de encuesta es una observación sobre las búsquedas que hicimos y no una prueba de que no exista trabajo previo. Además, la novedad es de aplicación y de implementación, no de mecanismo (capítulo 6).
- Validez externa, los datos son de EE. UU., así que aplicar el modelo en otra población, por ejemplo Colombia, requeriría validarlo localmente primero.

## Próximos pasos

1. Incorporar el diseño muestral de la NHIS (pesos, estratos y conglomerados) en los intervalos y en la evaluación.
2. Añadir variables clínicas y de estilo de vida que hoy faltan, como alcohol, actividad física y sueño, que están disponibles solo en 2022, y evaluar si rompen el techo de 0,54.
3. Estudiar por qué el modelo discrimina peor en mayores de 74 años y si conviene un modelo específico para ese grupo.
4. Calibrar y validar el modelo en una población local, por ejemplo colombiana, antes de cualquier uso de tamizaje, y fijar el umbral según el costo de las falsas alarmas.
5. Repetir el benchmark con más presupuesto de búsqueda y con SMOTE-NC para variables categóricas.
