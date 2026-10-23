"""
original_tab.py
================
Datos, figuras y modelo de la pestaña 7 (modelo original, el aditivo penalizado por rondas).

El dashboard nunca reentrena: carga el artefacto ya entrenado (artifacts/AditivoPenalizado.joblib) y los resultados
calculados por los scripts del proyecto (copias en data_e3/).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figuras import C, _estilo, perfil_df
from .calculos import ETIQUETAS

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data_e3"
ART = RAIZ / "artifacts"

NOMBRES = {"LogReg": "Regresión logística", "SVM": "SVM lineal", "XGBoost": "XGBoost", "RandomForest": "Random Forest",
           "DecisionTree": "Árbol de decisión", "NaiveBayes": "Naive Bayes", "KNN": "KNN", "EBM": "EBM (candidato, de la librería interpret)",
           "AditivoPenalizado": "Aditivo penalizado (modelo original, propio)"}
COMBO = {"SVM": ("sin_balanceo", "Grid"), "XGBoost": ("class_weight", "Bayesiana"), "LogReg": ("sin_balanceo", "Random"),
         "RandomForest": ("sin_balanceo", "Random"), "DecisionTree": ("sin_balanceo", "Grid"), "NaiveBayes": ("SMOTE", "Grid"),
         "KNN": ("sin_balanceo", "Bayesiana"), "EBM": ("sin_balanceo", "Bayesiana"), "AditivoPenalizado": ("sin_balanceo", "Bayesiana")}
UMBRAL = 0.245

PSEUDOCODIGO = """Algoritmo AditivoPenalizado(X, y; R, eta, lambda, kappa, semilla)
 1. Codificar cada variable con el entrenamiento (tramos por cuantiles o niveles); el ultimo nivel es "faltante o no visto"
 2. b0 <- logit(media de y);  f_j <- 0 para todo j;  F <- b0
 3. Para r = 1, ..., R
 4.     p <- sigmoide(F);  b0 <- b0 - eta * sum(p - y) / (n/4)
 5.     Para cada variable j, en orden aleatorio
 6.         g <- sigmoide(F) - y;  G_b <- suma de g por nivel;  n_b <- filas por nivel
 7.         Resolver (diag(n_b/4 + lambda) + kappa D) d = -(G + lambda f_j + kappa D f_j)
 8.         f_j <- f_j + eta * d;  F <- F + eta * d[nivel de cada fila]
 9.     Guardar el objetivo J(r), que no aumenta
10. Centrar cada f_j y pasar su media al intercepto
Prediccion  P(y=1|x) = sigmoide( b0 + suma_j f_j(nivel de x_j) )   ·   Complejidad O(R p n)"""


def _csv(n, **kw):
    return pd.read_csv(DATA / f"{n}.csv", **kw)


def cargar() -> dict:
    """Lee los resultados del modelo original. Devuelve un diccionario vacío si falta algo (la pestaña lo avisa)."""
    try:
        res = _csv("original_resumen")
        res["Combinación"] = res["modelo"].map(lambda m: f"{COMBO[m][0]} · {COMBO[m][1]}")
        res["Modelo"] = res["modelo"].map(NOMBRES)
        res["PR-AUC (media ± sd)"] = res.apply(lambda r: f"{r.PR_AUC_media:.4f} ± {r.PR_AUC_sd:.4f}", axis=1)
        res["IC 95 % BCa"] = res.apply(lambda r: f"[{r.IC95_BCa_inf:.4f}, {r.IC95_BCa_sup:.4f}]", axis=1)
        lider = res[["Modelo", "Combinación", "PR-AUC (media ± sd)", "IC 95 % BCa", "ROC_AUC_media", "Brier_media", "ECE_media", "n_estimaciones",
                     "t_entrenamiento_s", "t_inferencia_s"]].rename(columns={"ROC_AUC_media": "ROC-AUC", "Brier_media": "Brier", "ECE_media": "ECE",
                     "n_estimaciones": "Estimaciones", "t_entrenamiento_s": "Entrenamiento (s)", "t_inferencia_s": "Inferencia (s)"}).round(4)
        ab = pd.read_csv(DATA / "ablacion_original.csv", header=[0, 1], index_col=0)
        ab.columns = [f"{a} {b}" for a, b in ab.columns]
        return {"lider": lider, "pareadas": _csv("original_pruebas_pareadas"), "prueba": _csv("original_pruebas_prueba"),
                "friedman": _csv("original_friedman"), "ablacion": ab.reset_index().round(4), "test": _csv("test_original_vs_finalistas"),
                "veredicto": json.loads((DATA / "original_veredicto.json").read_text()),
                "hp": json.loads((DATA / "hiperparametros_original.json").read_text()), "ok": True}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": repr(e)}


_MODELO = None


def modelo():
    """Carga una sola vez el artefacto del modelo original (el pickle se hizo con el módulo src.original_model)."""
    global _MODELO
    if _MODELO is None:
        import joblib
        from . import original_model as om
        sys.modules.setdefault("src", types.ModuleType("src"))
        sys.modules["src.original_model"] = om
        _MODELO = joblib.load(ART / "AditivoPenalizado.joblib")
    return _MODELO


def predecir(perfil: dict):
    """Probabilidad y contribuciones exactas (valores SHAP en logit) de un perfil, con el modelo ya entrenado."""
    m = modelo()
    df = perfil_df(perfil)
    prob = float(m.predict_proba(df)[0, 1])
    c = m.contribuciones(df).iloc[0]
    d = pd.DataFrame({"variable": c.index, "factor": [ETIQUETAS[v] for v in c.index],
                      "valor": [f"{int(perfil[v])} años" if v == "edad" else str(perfil[v]) for v in c.index], "logit": c.values})
    return prob, d


def fig_shap(d: pd.DataFrame) -> go.Figure:
    d = d.reindex(d["logit"].abs().sort_values().index)
    fig = go.Figure(go.Bar(y=d["factor"], x=d["logit"], orientation="h",
                           marker_color=[C["con"] if v > 0.005 else (C["teal"] if v < -0.005 else "#C9BDAE") for v in d["logit"]],
                           customdata=d["valor"], hovertemplate="<b>%{y}</b><br>Valor elegido: %{customdata}<br>Aporte al logit: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=C["texto"])
    fig.update_xaxes(title="aporte al logit (valor SHAP exacto, 0 = efecto promedio del entrenamiento)")
    return _estilo(fig, "¿Qué empuja la predicción de este perfil?", 460, dict(l=10, r=20, t=50, b=60))
