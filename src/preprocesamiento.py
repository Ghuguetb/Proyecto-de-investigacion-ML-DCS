"""
Preprocesamiento (sección 2.9 de la guía).

Cada paso se justifica con un hallazgo del EDA (ver notebook 03). Todo queda dentro
de un Pipeline de scikit-learn, de modo que imputación, agrupación de categorías raras,
splines, escalado y codificación se ajustan SOLO con los datos de entrenamiento.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from .utils import CATEGORIAS, ORDENES, PREDICTORAS_CAT, PREDICTORAS_NUM

# Agrupación de categorías raras (< 1 % en entrenamiento, ver EDA 2.2)
AGRUPACION_RARAS = {
    "raza_etnia": {"AIAN no hispano": "Otro / múltiple (incl. AIAN)",
                   "AIAN y otro grupo": "Otro / múltiple (incl. AIAN)",
                   "Otro / múltiple": "Otro / múltiple (incl. AIAN)"},
    "dificultad_auditiva": {"Mucha": "Mucha o no puede", "No puede": "Mucha o no puede"},
    "dificultad_visual": {"Mucha": "Mucha o no puede", "No puede": "Mucha o no puede"},
}


def categorias_finales():
    """Categorías de cada predictora después de agrupar las raras (la primera es la referencia)."""
    cats = {}
    for v in PREDICTORAS_CAT:
        base = CATEGORIAS.get(v) or ORDENES.get(v)
        if v in AGRUPACION_RARAS:
            mapeo = AGRUPACION_RARAS[v]
            base = list(dict.fromkeys(mapeo.get(c, c) for c in base))
        cats[v] = base
    # Referencia de IMC: peso normal (no bajo peso)
    cats["imc_categoria"] = ["Normal", "Bajo peso", "Sobrepeso", "Obesidad"]
    return cats


def agrupar_raras(X):
    X = pd.DataFrame(X, columns=PREDICTORAS_CAT).copy()
    for v, mapeo in AGRUPACION_RARAS.items():
        X[v] = X[v].replace(mapeo)
    return X


# Variables con >= 2 % de faltantes en entrenamiento: se añade un indicador de faltante
# porque el EDA mostró que su faltante no es completamente aleatorio (ver notebook 01).
CON_INDICADOR = ["frecuencia_depresion", "tabaquismo", "imc_categoria"]


def construir_preprocesador(n_knots: int = 4) -> ColumnTransformer:
    """ColumnTransformer con tres ramas:
    - edad: imputación por mediana + spline cúbico (relación en U) + escalado;
    - categóricas: agrupación de categorías raras + imputación por moda + one-hot;
    - indicadores de faltante para las variables con >= 2 % de faltantes.
    """
    cats = categorias_finales()
    num = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
        ("escalar", StandardScaler()),
    ])
    cat = Pipeline([
        ("agrupar_raras", FunctionTransformer(agrupar_raras, feature_names_out="one-to-one")),
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(categories=[cats[v] for v in PREDICTORAS_CAT], drop="first",
                                    handle_unknown="error", sparse_output=False)),
    ])
    faltantes = MissingIndicator(features="all")
    return ColumnTransformer([("num", num, PREDICTORAS_NUM),
                              ("cat", cat, PREDICTORAS_CAT),
                              ("faltante", faltantes, CON_INDICADOR)],
                             verbose_feature_names_out=True)
