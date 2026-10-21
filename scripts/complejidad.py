"""Convergencia de los optimizadores (anytime performance) y analisis empirico de complejidad."""
import glob, json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src import benchmark as B
from src.utils import PREDICTORAS, OBJETIVO, SEED, cargar_particiones
warnings.filterwarnings("ignore")
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"

# ---------------- 1. Convergencia ----------------
curvas, filas, div = [], [], []
for f in glob.glob(str(OUT / "tareas" / "*.json")):
    r = json.load(open(f))
    if r["modelo"] == "EBM": continue
    h = r["historia"]; bs = np.array([x["mejor_hasta_ahora"] for x in h]); fin = bs[-1]
    reg = fin - np.pad(bs, (0, max(0, B.PRESUPUESTO - len(bs))), mode="edge")[:B.PRESUPUESTO]
    curvas.append((r["optimizador"], r["modelo"], reg, bs, [x["t"] for x in h]))
    alc = next((i + 1 for i, v in enumerate(bs) if v >= fin - 0.002), len(bs))
    filas.append((r["optimizador"], r["modelo"], len(h), fin, alc, h[-1]["t"], h[-1]["t"] / len(h)))
    if r["optimizador"] == "Genetica" and r.get("diversidad_ga"): div.append(r["diversidad_ga"])
cv = pd.DataFrame(filas, columns=["optimizador", "modelo", "n_evals", "mejor_PR_AUC_interno", "evals_para_llegar_a_0.002", "t_busqueda_s", "t_por_eval_s"])
res = cv.groupby("optimizador").agg(evals=("n_evals", "mean"), PR_AUC_interno=("mejor_PR_AUC_interno", "mean"),
      evals_hasta_cerca_del_optimo=("evals_para_llegar_a_0.002", "mean"), t_busqueda_s=("t_busqueda_s", "mean"), t_por_eval_s=("t_por_eval_s", "mean")).round(4)
res.to_csv(OUT / "convergencia_resumen.csv"); print(res.to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for o in B.OPTIMIZADORES:
    reg = np.mean([c[2] for c in curvas if c[0] == o], axis=0); ax[0].plot(range(1, B.PRESUPUESTO + 1), reg, "o-", label=o)
ax[0].set_xlabel("evaluaciones realizadas"); ax[0].set_ylabel("arrepentimiento medio (mejor final - mejor hasta ahora)"); ax[0].legend(); ax[0].set_title("Convergencia de los optimizadores")
if div: ax[1].plot(range(1, len(div[0]) + 1), np.mean([d[:len(div[0])] for d in div], axis=0), "o-", c="#b2182b"); ax[1].set_xlabel("generacion"); ax[1].set_ylabel("diversidad (distancia media entre individuos)"); ax[1].set_title("Algoritmo genetico, diversidad de la poblacion")
plt.tight_layout(); plt.savefig(FIG / "convergencia.png", dpi=130); plt.close()

# ---------------- 2. Complejidad empirica ----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from src.preprocesamiento import construir_preprocesador
train, _ = cargar_particiones()
Xtr, ytr = train[PREDICTORAS], train[OBJETIVO].values
Xp = construir_preprocesador().fit_transform(Xtr); print("dims preprocesadas", Xp.shape)
rs = np.random.RandomState(SEED); orden = rs.permutation(len(Xp)); NS = [2000, 4000, 8000, 16000, 32000]
cfg = {
 "KNN (brute)": lambda: KNeighborsClassifier(n_neighbors=30, algorithm="brute"),
 "KNN (kd_tree)": lambda: KNeighborsClassifier(n_neighbors=30, algorithm="kd_tree"),
 "NaiveBayes": lambda: GaussianNB(),
 "LogReg (liblinear)": lambda: LogisticRegression(solver="liblinear", max_iter=500),
 "LogReg (saga)": lambda: LogisticRegression(solver="saga", max_iter=300, tol=1e-3),
 "LogReg (lbfgs)": lambda: LogisticRegression(solver="lbfgs", max_iter=500),
 "DecisionTree": lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED),
 "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1, random_state=SEED),
 "XGBoost (hist)": lambda: XGBClassifier(tree_method="hist", n_estimators=100, max_depth=4, n_jobs=1, verbosity=0),
 "XGBoost (exact)": lambda: XGBClassifier(tree_method="exact", n_estimators=100, max_depth=4, n_jobs=1, verbosity=0),
 "LinearSVC": lambda: LinearSVC(dual=False, max_iter=2000),
 "SGD (hinge)": lambda: SGDClassifier(loss="hinge", random_state=SEED),
 "SVC (kernel rbf)": lambda: SVC(kernel="rbf"),
}
TOPE = {"SVC (kernel rbf)": 8000, "XGBoost (exact)": 16000}
filas = []
for nombre, mk in cfg.items():
    for n in NS:
        if n > TOPE.get(nombre, 10**9): continue
        idx = orden[:n]; m = mk(); t = time.time(); m.fit(Xp[idx], ytr[idx]); tf = time.time() - t
        t = time.time(); m.predict(Xp[orden[-2000:]]); tp = time.time() - t
        filas.append((nombre, n, tf, tp)); print(f"{nombre:20s} n={n:6d} fit={tf:7.2f}s pred(2000)={tp:6.2f}s", flush=True)
ebm_t = []
for n in NS[:4]:
    idx = orden[:n]; m = ExplainableBoostingClassifier(interactions=5, outer_bags=4, n_jobs=1, random_state=SEED); t = time.time()
    m.fit(Xtr.iloc[idx].assign(**{c: Xtr.iloc[idx][c].astype("category") for c in Xtr.columns if c != "edad"}), ytr[idx]); tf = time.time() - t
    filas.append(("EBM", n, tf, np.nan)); print(f"EBM n={n} fit={tf:.2f}s", flush=True)
cx = pd.DataFrame(filas, columns=["modelo", "n", "t_fit_s", "t_pred_2000_s"]); cx.to_csv(OUT / "complejidad_empirica.csv", index=False)
pend = cx.groupby("modelo").apply(lambda d: np.polyfit(np.log(d.n), np.log(d.t_fit_s), 1)[0] if len(d) > 2 else np.nan).rename("exponente_empirico_fit")
pend.round(2).to_csv(OUT / "complejidad_exponentes.csv"); print(pend.round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 5))
for nombre, d in cx.groupby("modelo"): ax.loglog(d.n, d.t_fit_s, "o-", label=nombre)
ax.set_xlabel("n (filas de entrenamiento)"); ax.set_ylabel("tiempo de ajuste (s)"); ax.legend(fontsize=7, ncol=2); ax.set_title("Costo empirico de ajuste (escala log-log)")
plt.tight_layout(); plt.savefig(FIG / "complejidad.png", dpi=130); plt.close()
# costo total del benchmark
fo = pd.read_csv(OUT / "resultados_por_fold.csv"); fo["t_total_s"] = fo.t_busqueda_s + fo.t_refit_s
fo.groupby("modelo").t_total_s.sum().div(60).round(1).to_csv(OUT / "costo_total_por_modelo_min.csv")
fo.groupby("optimizador").t_total_s.sum().div(60).round(1).to_csv(OUT / "costo_total_por_optimizador_min.csv")
print("FIN")
