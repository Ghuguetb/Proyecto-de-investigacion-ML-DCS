"""Tabla maestra de experimentos con las 140 combinaciones del diseño del segundo entregable (112 de clasificación y
28 de regresión), el EBM y el modelo original, con metricas del bucle externo, tiempos y semillas."""
import itertools, sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import pandas as pd
from src.benchmark import BALANCEOS, MODELOS, OPTIMIZADORES
OUT = RAIZ / "resultados"
m = pd.read_csv(OUT / "tabla_maestra.csv")
cols = ["PR_AUC_media", "PR_AUC_sd", "ROC_AUC_media", "Brier_media", "ECE_media", "t_busqueda_s", "t_refit_s", "semilla_global", "semillas_por_fold"]
filas = []
for mod, bal, opt in itertools.product(MODELOS, BALANCEOS, OPTIMIZADORES):
    r = m[(m.modelo == mod) & (m.balanceo == bal) & (m.optimizador == opt)]
    if len(r):
        r = r.iloc[0]; filas.append({"tarea": "clasificación", "modelo": mod, "balanceo": bal, "optimizador": opt, "estado": "evaluada",
                                      "n_estimaciones": int(r.n_folds), **{c: r[c] for c in cols}, "nota": "5 pliegues externos del segundo entregable"})
    else:
        filas.append({"tarea": "clasificación", "modelo": mod, "balanceo": bal, "optimizador": opt, "estado": "no aplica",
                      "nota": "KNN no admite pesos de clase"})
for mod, opt in itertools.product(MODELOS, OPTIMIZADORES):
    filas.append({"tarea": "regresión", "modelo": mod, "balanceo": "no aplica", "optimizador": opt, "estado": "no ejecutada",
                  "nota": "desenlace binario, las métricas de error de regresión no aplican (ver capítulo 5 y conclusiones)"})
fo = pd.read_csv(OUT / "original_resultados_por_fold.csv")
for mod in ["EBM", "AditivoPenalizado"]:
    d = fo[fo.modelo == mod]
    filas.append({"tarea": "clasificación", "modelo": mod, "balanceo": "sin_balanceo", "optimizador": "Bayesiana", "estado": "evaluada",
                  "n_estimaciones": len(d), "PR_AUC_media": d.PR_AUC.mean(), "PR_AUC_sd": d.PR_AUC.std(), "ROC_AUC_media": d.ROC_AUC.mean(),
                  "Brier_media": d.Brier.mean(), "ECE_media": d.ECE.mean(), "t_busqueda_s": d.t_busqueda_s.sum(), "t_refit_s": d.t_refit_s.sum(),
                  "semilla_global": 42, "semillas_por_fold": "42 + 1000 r + pliegue (r = 0 a 4)",
                  "nota": "modelo original (propio)" if mod == "AditivoPenalizado" else "candidato interpretable de la librería interpret, no cumple R2"})
t = pd.DataFrame(filas); t.to_csv(OUT / "tabla_maestra_completa.csv", index=False)
print(t.groupby(["tarea", "estado"]).size())
assert len(t[t.tarea == "clasificación"]) == 112 + 2 and len(t[t.tarea == "regresión"]) == 28
