# Conclusiones

## Respuesta a la pregunta de investigación

Los factores de riesgo modificables de demencia y el antecedente de ACV contienen información real pero limitada para identificar a los adultos con DCS. Una regresión logística con esas variables alcanza en datos no vistos una PR-AUC de 0,538 (2,6 veces la referencia trivial de 0,206) y un ROC-AUC de 0,804, detecta el 63,7 % de los casos con una precisión del 46,1 % y entrega probabilidades bien calibradas. El problema es aprendible, pero está lejos de resolverse del todo con un modelo lineal aditivo.

## Hallazgos principales

1. La variable objetivo es frecuente y estable, con el 20,6 % de los adultos reportando alguna dificultad para recordar o concentrarse, sin cambios entre 2022 y 2023.
2. La edad tiene una relación en forma de U, la dificultad es más frecuente en adultos jóvenes y en mayores que en la mediana edad, lo que obligó a modelarla con splines.
3. La salud mental es el factor más fuerte. Los síntomas depresivos muestran un gradiente dosis-respuesta, con un OR ajustado de 4,9 para síntomas diarios frente a nunca.
4. Las pérdidas sensoriales pesan casi tanto como la depresión diagnosticada, con OR ajustados de 2,1 a 2,6, en línea con lo que señala la Comisión Lancet.
5. El ACV es un factor independiente, con un OR crudo de 3,36, ajustado por edad de 2,52 y ajustado por todas las variables de 1,69. Su efecto es mucho mayor en adultos jóvenes que en mayores de 75 años.
6. La escolaridad es protectora, con un gradiente claro (OR de 0,53 para posgrado frente a menos que secundaria), coherente con la hipótesis de la reserva cognitiva.
7. La hipertensión pierde su asociación una vez que se ajusta por edad y por los demás factores.

## Decisiones metodológicas clave

- Reserva del conjunto de prueba antes de tomar cualquier decisión basada en los datos.
- Exclusión de la pregunta de seguimiento de cognición, que por sí sola alcanzaba un AUC de 0,929, ya que sin esta auditoría el modelo habría superado artificialmente el umbral de 0,90.
- Uso de la categoría de IMC en lugar del IMC numérico, por el faltante de tipo MNAR de este último.
- Descarte del accuracy como métrica de decisión, ya que el modelo trivial obtiene 0,794 sin detectar un solo caso.
- Umbral de decisión de 0,246, elegido con predicciones out-of-fold del entrenamiento.

## Limitaciones

- Diseño transversal, las asociaciones no son causales, y la relación entre depresión y DCS probablemente es bidireccional.
- Autorreporte, tanto el objetivo como los diagnósticos dependen de la percepción y del acceso previo a servicios de salud.
- Cobertura parcial del marco Lancet, con 9 de los 14 factores disponibles en ambos años, y faltan alcohol, inactividad física, aislamiento social, traumatismo craneoencefálico y contaminación del aire.
- Representatividad, la muestra sobrerrepresenta a los mayores de 65 años, excluye a la población institucionalizada y a quienes necesitaron un informante, y el modelo no usa pesos muestrales.
- Validez externa, los datos son de EE. UU., así que aplicar el modelo en otra población, por ejemplo Colombia, requeriría validarlo localmente primero.

## Próximos pasos para las siguientes entregas

1. Modelos no lineales, como árboles, random forest o gradient boosting, capaces de aprender las interacciones detectadas (ACV con edad, perfiles afectivo y cardiometabólico).
2. Análisis de sensibilidad sin las variables en observación (síntomas depresivos y dificultades sensoriales) para cuantificar el posible sesgo de estilo de respuesta.
3. Evaluación de equidad del desempeño por sexo, grupo de edad y raza o etnia.
4. Explorar, solo con los datos de 2022, el aporte de alcohol, actividad física y sueño.
