"""
benchmark_tabs.py
==================
Datos, figuras y pestañas del benchmark (benchmark de 112 combinaciones, optimización,
resultados finales, interpretabilidad y robustez).

Todo se lee de data_e3/, que son copias de los archivos que genera el capítulo 5 del libro (carpeta resultados/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figuras import C, _estilo, fig_vacia

DATA = Path(__file__).resolve().parents[2] / "data_e3"

NOMBRES = {"LogReg": "Regresión logística", "SVM": "SVM lineal", "XGBoost": "XGBoost", "RandomForest": "Random Forest",
           "DecisionTree": "Árbol de decisión", "NaiveBayes": "Naive Bayes", "KNN": "KNN", "EBM": "EBM (modelo original)"}
BAL_ORDEN = ["sin_balanceo", "class_weight", "SMOTE", "ADASYN"]
OPT_ORDEN = ["Grid", "Random", "Bayesiana", "Genetica"]
METRICAS = {"PR_AUC_media": "PR-AUC", "ROC_AUC_media": "ROC-AUC", "Brier_media": "Brier (menor es mejor)",
            "ECE_media": "Error de calibración ECE (menor es mejor)", "Recall_05_media": "Recall con umbral 0,5",
            "F1_05_media": "F1 con umbral 0,5"}
COL_MODELO = {"LogReg": C["terracota"], "SVM": C["teal"], "XGBoost": C["dorado"], "RandomForest": C["salvia"],
              "DecisionTree": C["ciruela"], "NaiveBayes": "#9A8B7C", "KNN": "#5B4B3E", "EBM": "#8C2F39"}


def _csv(nombre: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{nombre}.csv", **kw)


def _json(nombre: str) -> dict:
    return json.loads((DATA / f"{nombre}.json").read_text())


def cargar() -> dict:
    """Carga todos los resultados una sola vez."""
    D = dict(
        maestra=_csv("tabla_maestra"), rangos=_csv("friedman_rangos"), delong=_csv("delong_vs_mejor"),
        cliffs=_csv("cliffs_delta"), ebm=_csv("ebm_vs_referencias"), test=_csv("test_final_metricas"),
        test_delong=_csv("test_delong"), imp=_csv("importancia_variables", index_col=0),
        conv=_csv("convergencia_resumen"), comp=_csv("complejidad_empirica"), expo=_csv("complejidad_exponentes"),
        costo_m=_csv("costo_total_por_modelo_min"), costo_o=_csv("costo_total_por_optimizador_min"),
        calib=_csv("calibracion_por_modelo_balanceo"), lime=_csv("lime_vs_shap_global", index_col=0),
        robustez=_csv("robustez"), fried=_json("friedman_por_factor"), lime_j=_json("lime_vs_shap"),
        spear=_json("acuerdo_importancias_spearman"))
    return D


E3 = cargar()


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_mapa(metrica: str, opt: str) -> go.Figure:
    """Mapa de calor modelo × balanceo con el promedio de los 5 folds externos."""
    m = E3["maestra"]
    m = m if opt == "todos" else m[m["optimizador"] == opt]
    t = m.pivot_table(index="modelo", columns="balanceo", values=metrica, aggfunc="mean").reindex(columns=BAL_ORDEN)
    t = t.loc[t.mean(axis=1).sort_values(ascending=metrica in ("Brier_media", "ECE_media")).index]
    menor_mejor = metrica in ("Brier_media", "ECE_media")
    fig = go.Figure(go.Heatmap(z=t.values, x=["Sin balanceo", "class_weight", "SMOTE", "ADASYN"], y=[NOMBRES[i] for i in t.index],
                               colorscale=[[0, "#F3E9D8"], [1, C["teal"]]] if not menor_mejor else [[0, C["teal"]], [1, "#F3E9D8"]],
                               text=np.round(t.values, 3), texttemplate="%{text}", hovertemplate="%{y} · %{x}<br>%{z:.4f}<extra></extra>",
                               colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange="reversed")
    return _estilo(fig, f"{METRICAS[metrica]} · promedio de 5 folds" + ("" if opt == "todos" else f" · optimizador {opt}"), 400)


def fig_rangos(factor: str) -> go.Figure:
    f = E3["fried"][factor]
    r = pd.Series(f["rangos"]).sort_values()
    fig = go.Figure(go.Bar(x=r.values, y=[NOMBRES.get(i, i) for i in r.index], orientation="h", marker_color=C["teal"],
                           text=[f"{v:.1f}" for v in r.values], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=r.min() + f["CD"], line_dash="dot", line_color=C["terracota"],
                  annotation_text=f"mejor + diferencia crítica ({f['CD']:.2f})", annotation_position="top")
    fig.update_xaxes(title="Rango medio (1 es el mejor)")
    return _estilo(fig, f"Friedman por {factor} · chi² = {f['chi2']:.1f}, p = {f['p']:.1e}", 340)


def fig_pareto() -> go.Figure:
    m = E3["maestra"]
    fig = go.Figure()
    for k, g in m.groupby("modelo"):
        fig.add_trace(go.Scatter(x=g["t_total_min"], y=g["PR_AUC_media"], mode="markers", name=NOMBRES[k],
                                 marker=dict(color=COL_MODELO[k], size=9, opacity=0.8),
                                 customdata=np.c_[g["balanceo"], g["optimizador"]],
                                 hovertemplate=NOMBRES[k] + "<br>%{customdata[0]} · %{customdata[1]}<br>PR-AUC %{y:.3f}<br>%{x:.2f} min<extra></extra>"))
    fig.update_xaxes(title="Tiempo total de la combinación (min)", type="log")
    fig.update_yaxes(title="PR-AUC (promedio de 5 folds)")
    return _estilo(fig, "Desempeño frente a costo computacional", 420)


def fig_convergencia() -> go.Figure:
    c = E3["conv"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=c["optimizador"], y=c["evals"], name="Evaluaciones", marker_color=C["teal_claro"]))
    fig.add_trace(go.Bar(x=c["optimizador"], y=c["evals_hasta_cerca_del_optimo"], name="Evaluaciones hasta cerca del óptimo", marker_color=C["terracota"]))
    fig.update_layout(barmode="group")
    return _estilo(fig, "Evaluaciones por búsqueda y rapidez de convergencia", 340)


def fig_complejidad() -> go.Figure:
    t = E3["comp"]
    fig = go.Figure()
    for k, g in t.groupby("modelo"):
        fig.add_trace(go.Scatter(x=g["n"], y=g["t_fit_s"], mode="lines+markers", name=k))
    fig.update_xaxes(type="log", title="Filas de entrenamiento (n)")
    fig.update_yaxes(type="log", title="Tiempo de ajuste (s)")
    return _estilo(fig, "Complejidad empírica (escala log-log; la pendiente es el exponente)", 420)


def fig_calibracion_balanceo() -> go.Figure:
    t = E3["calib"]
    fig = go.Figure()
    for b, col in zip(["SMOTE", "ADASYN", "class_weight", "sin_balanceo"], [C["dorado"], C["ciruela"], C["terracota"], C["teal"]]):
        g = t[t["balanceo"].str.lower() == b.lower()] if "balanceo" in t else t
        fig.add_trace(go.Bar(x=g["modelo"], y=g["ECE"], name=b, marker_color=col))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="ECE (menor es mejor)")
    return _estilo(fig, "Balancear daña la calibración (error de calibración por modelo)", 380)


def fig_importancia() -> go.Figure:
    t = E3["imp"].copy()
    t = t / t.sum()
    t = t.sort_values("SHAP_XGBoost").tail(12)
    fig = go.Figure()
    for col, nom, color in [("SHAP_XGBoost", "SHAP · XGBoost", C["teal"]), ("abs_coef_LogReg", "|coeficiente| · logística", C["terracota"]),
                            ("EBM_importancia", "EBM", C["dorado"])]:
        fig.add_trace(go.Bar(y=t.index, x=t[col], name=nom, orientation="h", marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Importancia relativa (suma = 1)")
    return _estilo(fig, "Importancia por variable según tres métodos", 520)


def fig_lime() -> go.Figure:
    t = E3["lime"].copy()
    t = t / t.sum()
    t = t.sort_values("SHAP").tail(12)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t.index, x=t["SHAP"], name="SHAP", orientation="h", marker_color=C["teal"]))
    fig.add_trace(go.Bar(y=t.index, x=t["LIME"], name="LIME", orientation="h", marker_color=C["terracota"]))
    fig.update_layout(barmode="group")
    return _estilo(fig, "LIME frente a SHAP en XGBoost (importancia global relativa)", 460)


def tabla_robustez(analisis: str) -> pd.DataFrame:
    r = E3["robustez"]
    g = r[r["analisis"].str.startswith(analisis)].copy()
    g["valor"] = g["valor"].map(lambda v: f"{float(v):.3f}" if str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else v)
    return g.rename(columns={"modelo_o_grupo": "modelo o grupo", "metrica": "métrica"}).drop(columns="analisis")
