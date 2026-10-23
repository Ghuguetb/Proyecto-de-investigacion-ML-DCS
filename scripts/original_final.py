"""Modelo original en la prueba, ablacion, convergencia, curva de aprendizaje, complejidad e interpretabilidad.

1. Busqueda final de hiperparametros (TPE, 12 evaluaciones, submuestra de 8,000) sobre el entrenamiento completo,
   reajuste con todo el entrenamiento y UNA sola evaluacion en la prueba, junto a los finalistas del Entregable 2.
2. Ablacion con los hiperparametros de cada pliegue externo de la repeticion 0 (sin reajustar, ver el libro).
3. Convergencia (objetivo penalizado por ronda), curva de aprendizaje y complejidad empirica.
4. Valores SHAP exactos del modelo aditivo frente a los de XGBoost."""
import json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from scripts.run_original import buscar_original, crear_original  # noqa: E402
from src import benchmark as B
from src.original_model import AditivoPenalizado
from src.utils import ORDENES, OBJETIVO, PREDICTORAS, SEED, cargar_particiones
warnings.filterwarnings("ignore")
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"; REP = OUT / "repetido"
train, test = cargar_particiones()
Xtr, ytr = train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values
Xte, yte = test[PREDICTORAS].reset_index(drop=True), test[OBJETIVO].values
UMBRAL = 0.245

# ---------------- 1. Modelo final y prueba ----------------
ruta = OUT / "hiperparametros_original.json"
if ruta.exists():
    params = json.load(open(ruta))
else:
    sub, _ = train_test_split(np.arange(len(Xtr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=SEED)
    params, hist, _ = buscar_original(Xtr.iloc[sub], ytr[sub], SEED)
    json.dump(params, open(ruta, "w"), indent=1)
print("hiperparametros finales", params, flush=True)
t0 = time.time(); final = crear_original(params).fit(Xtr, ytr); t_fit = time.time() - t0
p_orig = final.predict_proba(Xte)[:, 1]
guardado = np.load(OUT / "test_probs.npy"); assert np.array_equal(guardado[0].astype(int), yte)
probs = dict(zip(["LogReg", "SVM", "XGBoost", "EBM"], guardado[1:5])); probs["AditivoPenalizado"] = p_orig
np.save(OUT / "test_probs_original.npy", np.stack([yte, p_orig]).astype(np.float32))
filas = []
from sklearn.metrics import confusion_matrix
for m, p in probs.items():
    yh = (p >= UMBRAL).astype(int); tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    met = B.metricas(yte, p)
    filas.append({"modelo": m, **{k: met[k] for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE"]},
                  "Recall_u": tp / (tp + fn), "Precision_u": tp / (tp + fp), "TN": tn, "FP": fp, "FN": fn, "TP": tp})
tabla = pd.DataFrame(filas).round(4); tabla.to_csv(OUT / "test_original_vs_finalistas.csv", index=False)
print(tabla.to_string(index=False), flush=True)

# ---------------- 2. Ablacion ----------------
fold0 = {int(json.loads(f.read_text())["fold"]): json.loads(f.read_text())["mejores_params"] for f in REP.glob("AditivoPenalizado__rep0__f*.json")}
y_all = ytr; X_all = Xtr
filas = []
variantes = {"completo": {}, "sin_suavizado (kappa=0)": {"kappa": 0.0}, "sin_ridge (lambda=0.001)": {"lambda_": 1e-3},
             "sin_orden_categorico": {"ordenes": {}}, "una_ronda_de_prueba (n_rondas=1)": {"n_rondas": 1}}
for fold, (tr, va) in enumerate(B.indices_externos(y_all)):
    bp = fold0[fold] if isinstance(fold0[fold], dict) else eval(fold0[fold])
    for nombre, cambio in variantes.items():
        kw = {"ordenes": ORDENES, **bp, **cambio}
        m = AditivoPenalizado(random_state=SEED, **kw).fit(X_all.iloc[tr], y_all[tr]); p = m.predict_proba(X_all.iloc[va])[:, 1]
        f = m.funciones_["edad"][:-1]
        filas.append({"variante": nombre, "fold": fold, **B.metricas(y_all[va], p), "rugosidad_edad": float(np.abs(np.diff(f)).sum()),
                      "cambios_de_signo_edad": int((np.diff(np.sign(np.diff(f))) != 0).sum())})
abl = pd.DataFrame(filas); abl.to_csv(OUT / "ablacion_original_por_fold.csv", index=False)
res_abl = abl.groupby("variante", sort=False)[["PR_AUC", "ROC_AUC", "Brier", "ECE", "rugosidad_edad", "cambios_de_signo_edad"]].agg(["mean", "std"]).round(4)
res_abl.to_csv(OUT / "ablacion_original.csv"); print(res_abl.to_string(), flush=True)

# ---------------- 3. Convergencia, aprendizaje, complejidad ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(np.arange(1, len(final.perdida_) + 1), final.perdida_, color="#1F2A80"); ax[0].set_xlabel("ronda"); ax[0].set_ylabel("objetivo penalizado J")
ax[0].set_title("Convergencia: J no aumenta en ninguna ronda"); ax[0].set_yscale("log")
tr, va = B.indices_externos(y_all)[0]; bp = fold0[0] if isinstance(fold0[0], dict) else eval(fold0[0])
tam, pr = [], []
for frac in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
    idx = tr if frac == 1.0 else train_test_split(tr, train_size=frac, stratify=y_all[tr], random_state=SEED)[0]
    m = crear_original(bp).fit(X_all.iloc[idx], y_all[idx]); tam.append(len(idx)); pr.append(B.metricas(y_all[va], m.predict_proba(X_all.iloc[va])[:, 1])["PR_AUC"])
ax[1].plot(tam, pr, "o-", color="#C8503A"); ax[1].set_xscale("log"); ax[1].set_xlabel("filas de entrenamiento"); ax[1].set_ylabel("PR-AUC en el pliegue de validacion")
ax[1].set_title("Curva de aprendizaje (pliegue 0)")
aprend = pd.DataFrame({"n_train": tam, "PR_AUC": pr}); aprend.to_csv(OUT / "curva_aprendizaje_original.csv", index=False)
# complejidad empirica con 100 rondas fijas, mediana de 3 repeticiones
cx = []
for n in (2000, 4000, 8000, 16000, 32000, 44903):
    ii = np.random.default_rng(SEED).choice(len(Xtr), n, replace=False)
    for rep in range(3):
        t = time.time(); AditivoPenalizado(n_rondas=100, ordenes=ORDENES).fit(Xtr.iloc[ii], ytr[ii]); cx.append(("n", n, 14, time.time() - t))
for pcols in (2, 4, 6, 8, 10, 14):
    for rep in range(3):
        t = time.time(); AditivoPenalizado(n_rondas=100, ordenes=ORDENES).fit(Xtr[PREDICTORAS[:pcols]], ytr); cx.append(("p", 44903, pcols, time.time() - t))
cx = pd.DataFrame(cx, columns=["eje", "n", "p", "t"]).groupby(["eje", "n", "p"], as_index=False)["t"].median()
cx.to_csv(OUT / "complejidad_original.csv", index=False)
a = cx[cx.eje == "n"]; b = cx[cx.eje == "p"]
ex_n = np.polyfit(np.log(a.n), np.log(a.t), 1)[0]; ex_p = np.polyfit(np.log(b.p), np.log(b.t), 1)[0]
ax[2].loglog(a.n, a.t, "o-", label=f"vs n (pendiente {ex_n:.2f})"); ax[2].loglog(b.p * 3000, b.t, "s--", label=f"vs p (pendiente {ex_p:.2f}, eje p x 3000)")
ax[2].set_xlabel("n (o p x 3000)"); ax[2].set_ylabel("segundos de entrenamiento"); ax[2].set_title("Complejidad empirica, 100 rondas"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "original_convergencia_complejidad.png", dpi=130); plt.close()
json.dump({"exponente_n": ex_n, "exponente_p": ex_p, "t_fit_final_s": t_fit}, open(OUT / "complejidad_original.json", "w"))
print("exponentes", ex_n, ex_p, flush=True)

# ---------------- 4. Interpretabilidad ----------------
contr = final.contribuciones(Xte); imp = contr.abs().mean().sort_values(ascending=False)
ref = pd.read_csv(OUT / "importancia_variables.csv", index_col=0)
comp = pd.DataFrame({"SHAP_exacto_AditivoPenalizado": imp}).join(ref[["SHAP_XGBoost", "EBM_importancia"]]).fillna(0)
sp = {"vs_SHAP_XGBoost": float(spearmanr(comp.iloc[:, 0], comp.SHAP_XGBoost)[0]), "vs_EBM": float(spearmanr(comp.iloc[:, 0], comp.EBM_importancia)[0])}
comp.round(4).to_csv(OUT / "importancia_original_vs_referencias.csv"); json.dump(sp, open(OUT / "importancia_original_spearman.json", "w"))
print(comp.round(3).head(8).to_string(), sp, flush=True)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a_, v in zip(ax, ["edad", "frecuencia_depresion", "dificultad_auditiva", "educacion"]):
    f = final.funciones_[v]
    if v == "edad":
        cortes = final.cortes_[v]; x = np.r_[cortes[0] - 1, cortes] if len(cortes) else [0]; a_.step(np.r_[cortes[0] - 2, cortes, cortes[-1] + 2][:len(f) - 1], f[:-1], where="post", color="#1F2A80"); a_.set_xlabel("edad (años)")
    else:
        niv = ORDENES[v]; a_.bar(range(len(niv)), f[:len(niv)], color="#14496B"); a_.set_xticks(range(len(niv))); a_.set_xticklabels([s[:12] for s in niv], rotation=40, ha="right", fontsize=7)
    a_.axhline(0, color="gray", lw=.6); a_.set_title(v, fontsize=9); a_.set_ylabel("aporte al logit")
plt.tight_layout(); plt.savefig(FIG / "original_curvas.png", dpi=130); plt.close()
print("FIN")
