"""Construye el notebook del capítulo 8 (validación estadística del modelo original)."""
import nbformat as nbf


def construir():
    C = []
    md = lambda t: C.append(nbf.v4.new_markdown_cell(t.strip()))
    code = lambda t: C.append(nbf.v4.new_code_cell(t.strip()))

    md(r"""
# 8. Validación estadística del modelo original

La pregunta de este capítulo es si el modelo aditivo penalizado es **significativamente mejor** que los modelos de referencia. Para responderla no basta con una sola partición, porque la diferencia entre modelos buenos suele ser del orden de la variabilidad entre particiones. Por eso se repitió la validación cruzada anidada cinco veces con semillas distintas, lo que da 25 estimaciones pareadas (5 repeticiones por 5 pliegues externos) de cada modelo sobre exactamente las mismas particiones. La repetición 0 es la misma del capítulo 5.

Las pruebas siguen la lógica de Demšar (2006) para comparar varios modelos en varios conjuntos de datos, la corrección de Nadeau y Bengio (2003) para validación cruzada repetida, y DeLong y McNemar sobre la prueba externa. Todas las comparaciones del original contra los demás modelos se corrigen con el método de Holm.
""")
    code(r"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display
warnings.filterwarnings("ignore")
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES, FIG = RAIZ / "resultados", RAIZ / "resultados" / "figuras"
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 30)
""")
    md(r"""
## 8.1 Hipótesis y criterio de decisión

Para cada modelo de referencia $M$:

$$H_0:\ \text{PR-AUC}(\text{original}) = \text{PR-AUC}(M) \qquad H_1:\ \text{PR-AUC}(\text{original}) > \text{PR-AUC}(M)$$

Se considera que el modelo original **supera** a los demás solo si se cumplen tres condiciones a la vez. La primera es que su media de PR-AUC sea la mayor. La segunda es que al menos tres pruebas distintas rechacen $H_0$ contra todos los modelos de referencia después del ajuste de Holm, con $\alpha = 0{,}05$. La tercera es que el tamaño del efecto no sea despreciable. Se fijó este criterio antes de mirar los resultados.
""")
    md("## 8.2 Rendimiento en la validación repetida")
    code(r"""
res = pd.read_csv(RES / "original_resumen.csv")
res.round(4)
""")
    md(r"""
La tabla muestra media, desviación y el intervalo BCa del 95 % obtenido por remuestreo con corrección de sesgo y aceleración. Los intervalos de los mejores modelos se traslapan casi por completo, y eso anticipa el resultado de las pruebas.
""")
    md("## 8.3 Prueba de Friedman y diagrama de diferencia crítica")
    code(r"""
display(pd.read_csv(RES / "original_friedman.csv").round(4))
display(pd.read_csv(RES / "original_rangos.csv").round(3))
""")
    code(r"""display(Image(filename=str(FIG / "original_cd.png")))""")
    md(r"""
Friedman rechaza que todos los modelos sean equivalentes, y era de esperar, porque KNN, árbol de decisión y Bayes ingenuo son claramente peores. Lo que importa es la distancia entre el original y los primeros lugares, que se compara con la diferencia crítica de Nemenyi. Si dos modelos están unidos por una barra en el diagrama, no se pueden distinguir.
""")
    md("## 8.4 Comparaciones pareadas en la validación repetida")
    code(r"""
pp = pd.read_csv(RES / "original_pruebas_pareadas.csv")
cols = ["referencia", "dif_media_PR_AUC", "IC95_BCa_inf", "IC95_BCa_sup", "wilcoxon_p_holm", "nadeau_bengio_p_holm", "cliffs_delta", "cohen_dz"]
pp[cols].round(4)
""")
    md(r"""
Una diferencia positiva significa que el original es mejor. Wilcoxon no supone normalidad, y la t corregida de Nadeau y Bengio agranda la varianza para reconocer que las particiones comparten datos de entrenamiento. El delta de Cliff y la d de Cohen miden el tamaño del efecto, no la significancia.
""")
    md("## 8.5 Comparaciones en la prueba externa")
    code(r"""
pt = pd.read_csv(RES / "original_pruebas_prueba.csv")
pt.round(4)
""")
    md(r"""
DeLong compara las áreas ROC de dos modelos evaluados sobre los mismos individuos. McNemar compara los aciertos con el umbral de 0,245 mirando solo los casos en que uno acierta y el otro falla.
""")
    md("## 8.6 Veredicto")
    code(r"""
v = json.load(open(RES / "original_veredicto.json"))
print("PR-AUC media del original: %.4f" % v["PR_AUC_media_original"])
for m, x in v["PR_AUC_media_por_referencia"].items():
    print("  %-14s %.4f   original gana en media: %s" % (m, x, v["el_original_gana_en_media_a"][m]))
print()
for k, x in v["pruebas_que_rechazan_a_favor_del_original"].items():
    print(("[x] " if x else "[ ] ") + k)
print()
print("Condición 1 (media mayor que todas):", v["condicion_1_media_mayor_que_todas"])
print("Condición 2 (al menos tres pruebas rechazan a su favor):", v["condicion_2_al_menos_tres_pruebas"])
print("Condición 3 (efecto no despreciable a su favor):", v["condicion_3_efecto_no_despreciable_a_favor"])
print("R3 cumplida:", v["R3_cumplida"], "| comparaciones totales corregidas:", v["comparaciones_totales"])
""")
    md(r"""
**Lectura.** El modelo original supera de forma significativa a KNN, al árbol de decisión, a Bayes ingenuo y al bosque aleatorio, y queda empatado con la regresión logística y XGBoost (diferencias de milésimas, con valores p ajustados de 1). Con el SVM y el EBM la diferencia media es levemente desfavorable, y en la prueba externa McNemar con Holm detecta que esos dos modelos aciertan algo más que el original con el umbral de 0,245. Por eso no hay superioridad frente a todos los modelos y la condición R3 **no se cumple**. Es un resultado honesto y útil. Con 14 predictoras casi todas categóricas, la señal aditiva ya está casi toda capturada por la regresión logística, y las interacciones o no existen o aportan muy poco. Lo que sí ofrece el modelo original es calibración mejor que la mayoría (ECE bajo), curvas legibles por variable, SHAP exacto y un costo de entrenamiento bajo. Son ventajas de interpretabilidad y de transparencia, no de poder predictivo.
""")
    md(r"""
## 8.7 Limitaciones de esta validación

1. Las 25 estimaciones no son independientes, porque las repeticiones reutilizan los mismos individuos. Por eso se usa la corrección de Nadeau y Bengio, que es conservadora pero no perfecta.
2. La búsqueda de hiperparámetros dentro de cada pliegue usó un submuestreo de 8 000 filas y 12 evaluaciones, para que 8 modelos por 25 bloques fueran viables. Esto afecta por igual a todos los modelos, pero podría dejar a alguno algo por debajo de su óptimo.
3. Todos los modelos, incluido el EBM, completaron los 25 bloques, así que las comparaciones se hacen sobre el mismo universo.
4. No encontrar diferencia significativa no demuestra equivalencia. Solo indica que con estos datos y este tamaño de muestra no se puede distinguir.
""")
    nb = nbf.v4.new_notebook(); nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    return nb
