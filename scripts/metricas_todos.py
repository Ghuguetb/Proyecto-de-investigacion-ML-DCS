"""Matriz de confusion, curva ROC y tabla de Accuracy, Precision, Recall, F1 y AUC para los 8 modelos.

Los cuatro finalistas (LogReg, SVM, XGBoost, EBM) reutilizan las probabilidades guardadas en
`resultados/test_probs.npy`. Los otros cuatro (KNN, Naive Bayes, Arbol, Random Forest) se ajustan aqui con
la misma receta, sin balanceo: busqueda bayesiana (TPE) sobre la submuestra de 8,000 filas del entrenamiento,
reajuste con todo el entrenamiento y una sola evaluacion en la prueba. Es un reporte descriptivo, ningun
modelo se elige con estos resultados.
"""
import json, os, sys, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
import optuna
from src import benchmark as B
from src.utils import PREDICTORAS, OBJETIVO, SEED, cargar_particiones
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"
train, test = cargar_particiones()
Xtr, ytr = train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values
Xte, yte = test[PREDICTORAS].reset_index(drop=True), test[OBJETIVO].values
UMBRAL = 0.245  # umbral que maximiza F1 fuera de particion en el entrenamiento (LogReg), el mismo de los finalistas

# 1. Probabilidades de los finalistas, ya calculadas
guardado = np.load(OUT / "test_probs.npy")
assert np.array_equal(guardado[0].astype(int), yte)
probs = dict(zip(["LogReg", "SVM", "XGBoost", "EBM"], guardado[1:5]))

# 2. Los otros cuatro modelos
sub, _ = train_test_split(np.arange(len(Xtr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=SEED)
ruta = OUT / "hiperparametros_otros_modelos.json"
params = json.load(open(ruta)) if ruta.exists() else {}
for m in ["KNN", "NaiveBayes", "DecisionTree", "RandomForest"]:
    if m not in params:
        ev = B.Evaluador(m, "sin_balanceo", Xtr.iloc[sub], ytr[sub], SEED)
        B.buscar_bayesiana(ev, m, SEED); params[m] = B.mejores_params(ev); print(m, params[m], flush=True)
        json.dump(params, open(ruta, "w"), indent=1, default=float)
    modelo = B.crear_pipeline(m, "sin_balanceo", params[m]).fit(Xtr, ytr)
    probs[m] = modelo.predict_proba(Xte)[:, 1]

# 3. Tabla de metricas y matrices de confusion
orden = ["LogReg", "SVM", "XGBoost", "EBM", "RandomForest", "DecisionTree", "NaiveBayes", "KNN"]
filas, cms = [], {}
for m in orden:
    p = probs[m]
    for u in (0.5, UMBRAL):
        yh = (p >= u).astype(int); tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
        filas.append({"modelo": m, "umbral": u, "Accuracy": accuracy_score(yte, yh), "Precision": precision_score(yte, yh, zero_division=0),
                      "Recall": recall_score(yte, yh), "F1": f1_score(yte, yh), "AUC": roc_auc_score(yte, p),
                      "PR_AUC": average_precision_score(yte, p), "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        cms[(m, u)] = np.array([[tn, fp], [fn, tp]])
tabla = pd.DataFrame(filas).round(3); tabla.to_csv(OUT / "test_metricas_todos_modelos.csv", index=False)
print(tabla.to_string(index=False), flush=True)

# 4. Figura: ROC de los 8 modelos y 8 matrices de confusion (umbral 0.245, filas normalizadas)
fig = plt.figure(figsize=(14, 7.5)); gs = fig.add_gridspec(2, 6)
ax = fig.add_subplot(gs[:, :2])
for m in orden:
    fpr, tpr, _ = roc_curve(yte, probs[m]); ax.plot(fpr, tpr, label=f"{m} ({roc_auc_score(yte, probs[m]):.3f})", lw=1.6)
ax.plot([0, 1], [0, 1], ":", c="gray"); ax.set_xlabel("tasa de falsos positivos"); ax.set_ylabel("tasa de verdaderos positivos")
ax.set_title("Curva ROC en la prueba"); ax.legend(fontsize=8, loc="lower right")
for k, m in enumerate(orden):
    a = fig.add_subplot(gs[k // 4, 2 + k % 4]); cm = cms[(m, UMBRAL)]; pr = cm / cm.sum(1, keepdims=True)
    a.imshow(pr, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({pr[i, j]:.0%})", ha="center", va="center", fontsize=8, color="white" if pr[i, j] > .5 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["sin", "con"], fontsize=8); a.set_yticklabels(["sin", "con"], fontsize=8)
    a.set_title(m, fontsize=9)
    if k % 4 == 0: a.set_ylabel("real", fontsize=8)
    if k // 4 == 1: a.set_xlabel("predicho", fontsize=8)
fig.suptitle(f"Matriz de confusion con umbral {UMBRAL} (sin = sin dificultad, con = con dificultad)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "confusion_roc_todos.png", dpi=130); plt.close()
print("FIN")
