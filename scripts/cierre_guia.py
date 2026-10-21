"""Cierre de requisitos de la guia: diagrama de diferencia critica, diversidad del genetico,
matrices de confusion y ROC de los finalistas, SHAP local, sensibilidad a semillas,
complejidad estandar vs optimizada con memoria, y recalibracion (Platt e isotonica).

Uso:  python scripts/cierre_guia.py [paso ...]   (sin argumentos corre todos los pasos)
Pasos: cd, diversidad, finalistas, shap_local, semillas, complejidad, recalibracion
Todo se calcula con TRAIN; solo `finalistas` lee las probabilidades de test ya guardadas.
"""
import json, os, subprocess, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"
PASOS = sys.argv[1:] or ["cd", "diversidad", "finalistas", "shap_local", "semillas", "complejidad", "recalibracion"]
NOM = {"modelo": "Modelo", "balanceo": "Balanceo", "optimizador": "Optimizador"}


def paso_cd():
    """Diagrama de diferencia critica de Nemenyi por factor (rangos medios y CD de friedman_por_factor.json)."""
    f = json.load(open(OUT / "friedman_por_factor.json"))
    fig, axs = plt.subplots(len(NOM), 1, figsize=(9, 2.6 * len(NOM)))
    for ax, (k, titulo) in zip(axs, NOM.items()):
        r = pd.Series(f[k]["rangos"]).sort_values(); cd = f[k]["CD"]; n = len(r)
        lo, hi = 1 - 0.1, n + 0.1
        ax.set_xlim(hi, lo); ax.set_ylim(0, 1); ax.axis("off")
        ax.plot([hi, lo], [0.62, 0.62], color="#333", lw=1)
        for t in range(1, n + 1):
            ax.plot([t, t], [0.62, 0.66], color="#333", lw=1); ax.text(t, 0.72, str(t), ha="center", fontsize=9)
        ax.plot([lo, lo + cd], [0.93, 0.93], color="#B4532A", lw=2); ax.text(lo + cd / 2, 0.97, f"CD = {cd:.2f}", ha="center", fontsize=9, color="#B4532A")
        ax.set_title(f"{titulo} (Friedman chi2 = {f[k]['chi2']:.1f}, p = {f[k]['p']:.1e})", fontsize=10, loc="left")
        der, izq = r.index[: (n + 1) // 2], r.index[(n + 1) // 2:][::-1]
        for i, m in enumerate(izq):
            y = 0.5 - 0.11 * i; ax.plot([r[m], r[m]], [0.62, y], color="#444", lw=0.8); ax.plot([r[m], hi + 0.2], [y, y], color="#444", lw=0.8)
            ax.text(hi + 0.25, y, f"{m} ({r[m]:.1f})", ha="right", va="center", fontsize=8.5)
        for i, m in enumerate(der):
            y = 0.5 - 0.11 * i; ax.plot([r[m], r[m]], [0.62, y], color="#444", lw=0.8); ax.plot([r[m], lo - 0.2], [y, y], color="#444", lw=0.8)
            ax.text(lo - 0.25, y, f"{m} ({r[m]:.1f})", ha="left", va="center", fontsize=8.5)
        # barras gruesas: grupos cuya diferencia de rangos es menor que CD (no distinguibles)
        vals = list(r.values); yb = 0.58; usados = set()
        for i in range(n):
            j = max(jj for jj in range(i, n) if vals[jj] - vals[i] < cd)
            if j > i and (i, j) not in usados and not any(i >= a and j <= b and (a, b) != (i, j) for a, b in usados):
                ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yb, yb], color="#1F6F78", lw=3.5, solid_capstyle="round"); yb -= 0.035; usados.add((i, j))
    plt.tight_layout(); plt.savefig(FIG / "cd_diagrama.png", dpi=140); plt.close(); print("cd ok", flush=True)


def paso_diversidad():
    """Diversidad genetica media por generacion (distancia euclidiana media entre individuos)."""
    import glob
    filas = []
    for f in glob.glob(str(OUT / "tareas" / "*__Genetica__f*.json")):
        d = json.load(open(f))
        if d.get("diversidad_ga"):
            for g, v in enumerate(d["diversidad_ga"], 1): filas.append({"modelo": d["modelo"], "generacion": g, "diversidad": v})
    df = pd.DataFrame(filas)
    tab = df.groupby(["modelo", "generacion"])["diversidad"].mean().unstack().round(3); tab.to_csv(OUT / "diversidad_genetica.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m, r in tab.iterrows(): ax.plot(r.index, r.values, "o-", label=m)
    g = df.groupby("generacion")["diversidad"].mean(); ax.plot(g.index, g.values, "k--", lw=2, label="promedio")
    ax.set_xlabel("Generación"); ax.set_ylabel("Distancia media entre individuos"); ax.set_xticks(sorted(df.generacion.unique())); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Diversidad genética por generación"); plt.tight_layout(); plt.savefig(FIG / "diversidad_genetica.png", dpi=140); plt.close()
    print("diversidad ok", g.round(3).to_dict(), flush=True)


def paso_finalistas():
    """Matriz de confusion, ROC y tabla accuracy/precision/recall/F1/AUC de los finalistas en test (umbrales 0.5 y 0.245)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
    a = np.load(OUT / "test_probs.npy"); y = a[0].astype(int); nombres = ["LogReg", "SVM", "XGBoost", "EBM"]
    filas = []
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4))
    for i, m in enumerate(nombres):
        p = a[i + 1]
        for u in (0.5, 0.245):
            yh = (p >= u).astype(int)
            filas.append({"modelo": m, "umbral": u, "Accuracy": accuracy_score(y, yh), "Precision": precision_score(y, yh), "Recall": recall_score(y, yh),
                          "F1": f1_score(y, yh), "AUC": roc_auc_score(y, p)})
        cm = confusion_matrix(y, (p >= 0.245).astype(int), normalize="true")
        axs[i].imshow(cm, cmap="Blues", vmin=0, vmax=1); axs[i].set_title(f"{m} (umbral 0,245)", fontsize=9)
        for r in range(2):
            for c in range(2): axs[i].text(c, r, f"{cm[r, c]:.0%}", ha="center", va="center", color="white" if cm[r, c] > .5 else "black")
        axs[i].set_xticks([0, 1], ["pred. no", "pred. sí"]); axs[i].set_yticks([0, 1], ["real no", "real sí"])
        fpr, tpr, _ = roc_curve(y, p); axs[4].plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(y, p):.3f})")
    axs[4].plot([0, 1], [0, 1], "k:"); axs[4].set_xlabel("1 - especificidad"); axs[4].set_ylabel("Sensibilidad"); axs[4].legend(fontsize=8); axs[4].set_title("Curvas ROC en prueba", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "confusion_roc_finalistas.png", dpi=140); plt.close()
    pd.DataFrame(filas).round(3).to_csv(OUT / "test_clasificacion_finalistas.csv", index=False); print("finalistas ok", flush=True)


def _train_test():
    from src.utils import PREDICTORAS, OBJETIVO, cargar_particiones
    tr, te = cargar_particiones()
    return tr[PREDICTORAS].reset_index(drop=True), tr[OBJETIVO].values, te[PREDICTORAS].reset_index(drop=True), te[OBJETIVO].values


def paso_shap_local():
    """Waterfall de SHAP para un acierto y un error grave del XGBoost final (observaciones del conjunto de prueba)."""
    import shap
    from src import benchmark as B
    Xtr, ytr, Xte, yte = _train_test()
    params = json.load(open(OUT / "hiperparametros_finales.json"))["XGBoost"]
    pipe = B.crear_pipeline("XGBoost", "sin_balanceo", params).fit(Xtr, ytr)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]; Xp = pre.transform(Xte); nombres = list(pre.get_feature_names_out())
    p = pipe.predict_proba(Xte)[:, 1]
    acierto = int(np.where((yte == 1) & (p >= 0.245))[0][np.argmax(p[(yte == 1) & (p >= 0.245)])])
    error = int(np.where(yte == 1)[0][np.argmin(p[yte == 1])])
    exp = shap.TreeExplainer(clf)
    sv = exp(Xp[[acierto, error]]); sv.feature_names = nombres
    casos = []
    for k, (nom, i) in enumerate([("acierto", acierto), ("error", error)]):
        plt.figure(figsize=(8, 4.2)); shap.plots.waterfall(sv[k], max_display=9, show=False); plt.title(f"Caso {nom}: prob. predicha {p[i]:.3f}, real = sí", fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / f"shap_waterfall_{nom}.png", dpi=140, bbox_inches="tight"); plt.close()
        casos.append({"caso": nom, "indice_test": i, "prob_predicha": round(float(p[i]), 3), **{c: Xte.iloc[i][c] for c in Xte.columns}})
    pd.DataFrame(casos).to_csv(OUT / "shap_local_casos.csv", index=False); print("shap_local ok", casos[0]["prob_predicha"], casos[1]["prob_predicha"], flush=True)


def paso_semillas():
    """Sensibilidad a la semilla: modelos con aleatoriedad (XGBoost, RandomForest) y optimizadores (Random, Bayesiana, Genetica)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from src import benchmark as B
    from src.preprocesamiento import construir_preprocesador
    Xtr, ytr, _, _ = _train_test(); semillas = [42, 43, 44, 45, 46]
    cv = StratifiedKFold(5, shuffle=True, random_state=42); pxgb = json.load(open(OUT / "hiperparametros_finales.json"))["XGBoost"]
    filas = []
    for s in semillas:
        for nom, clf in [("XGBoost", XGBClassifier(tree_method="hist", n_jobs=2, random_state=s, verbosity=0, **pxgb)),
                         ("RandomForest", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, n_jobs=2, random_state=s))]:
            pipe = B.crear_pipeline("LogReg", "sin_balanceo", {"C": 1.0, "penalty": "l2"}); pipe.steps[-1] = ("clf", clf)
            sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="average_precision"); filas.append({"modelo": nom, "semilla": s, "PR_AUC_cv": sc.mean()})
        print("semilla modelos", s, flush=True)
    pd.DataFrame(filas).to_csv(OUT / "semillas_modelos.csv", index=False)
    sub, _ = train_test_split(np.arange(len(Xtr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=42); filas = []
    for opt in ["Random", "Bayesiana", "Genetica"]:
        for s in semillas:
            ev = B.Evaluador("XGBoost", "sin_balanceo", Xtr.iloc[sub], ytr[sub], s); t0 = time.time(); B.BUSCADORES[opt](ev, "XGBoost", s)
            filas.append({"optimizador": opt, "semilla": s, "mejor_PR_AUC_interno": ev.mejor, "evals": len(ev.historia), "t_s": time.time() - t0, "mejores": json.dumps(B.mejores_params(ev), default=float)})
        print("semillas opt", opt, flush=True)
    pd.DataFrame(filas).to_csv(OUT / "semillas_optimizadores.csv", index=False); print("semillas ok", flush=True)


def paso_recalibracion():
    """Platt (sigmoide) e isotonica sobre modelos mal calibrados por el balanceo (se evalua en un 30 % de TRAIN)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from src import benchmark as B
    Xtr, ytr, _, _ = _train_test(); params = json.load(open(OUT / "hiperparametros_finales.json"))
    a, b = train_test_split(np.arange(len(Xtr)), test_size=0.3, stratify=ytr, random_state=42); filas = []
    for m in ["LogReg", "XGBoost"]:
        for bal in ["class_weight", "SMOTE"]:
            base = B.crear_pipeline(m, bal, params[m]).fit(Xtr.iloc[a], ytr[a]); p0 = base.predict_proba(Xtr.iloc[b])[:, 1]
            r = {"modelo": m, "balanceo": bal, "sin_recalibrar": B.metricas(ytr[b], p0)}
            for met in ["sigmoid", "isotonic"]:
                cal = CalibratedClassifierCV(B.crear_pipeline(m, bal, params[m]), method=met, cv=3).fit(Xtr.iloc[a], ytr[a]); r[met] = B.metricas(ytr[b], cal.predict_proba(Xtr.iloc[b])[:, 1])
            for k in ["sin_recalibrar", "sigmoid", "isotonic"]:
                filas.append({"modelo": m, "balanceo": bal, "calibracion": {"sin_recalibrar": "ninguna", "sigmoid": "Platt (sigmoide)", "isotonic": "isotónica"}[k],
                              "ECE": round(r[k]["ECE"], 4), "Brier": round(r[k]["Brier"], 4), "PR_AUC": round(r[k]["PR_AUC"], 4)})
            print("recal", m, bal, flush=True)
    pd.DataFrame(filas).to_csv(OUT / "recalibracion.csv", index=False); print("recalibracion ok", flush=True)


# ---------------- complejidad estandar vs optimizada, con memoria (cada configuracion en un proceso limpio) ----------------
CONFIGS = ["KNN brute", "KNN kd_tree", "XGBoost exact", "XGBoost hist", "LogReg lbfgs", "LogReg liblinear", "LogReg saga", "SVC rbf", "LinearSVC", "SGD hinge", "GaussianNB fit", "GaussianNB partial_fit"]
TEORIA = {"KNN brute": ("O(1) entrenar; O(n·p) por consulta", "O(n·p)"), "KNN kd_tree": ("O(n·p·log n) construir; O(log n) por consulta en baja dimensión", "O(n·p)"),
          "XGBoost exact": ("O(T·d·n·p·log n)", "O(T·d)"), "XGBoost hist": ("O(T·d·n·p) (más O(n·p) para el histograma)", "O(T·d)"),
          "LogReg lbfgs": ("O(it·n·p)", "O(p)"), "LogReg liblinear": ("O(it·n·p)", "O(p)"), "LogReg saga": ("O(it·n·p), cada paso lineal en n y p", "O(p)"),
          "SVC rbf": ("entre O(n²·p) y O(n³)", "O(n_sv·p)"), "LinearSVC": ("O(it·n·p)", "O(p)"), "SGD hinge": ("O(it·n·p)", "O(p)"),
          "GaussianNB fit": ("O(n·p)", "O(c·p)"), "GaussianNB partial_fit": ("O(n·p) por lote", "O(c·p)")}


def _hijo(nombre):
    from sklearn.metrics import average_precision_score

    def pico_mb():
        """Pico de memoria residente del proceso (VmHWM), que no hereda el pico del proceso padre."""
        for l in open('/proc/self/status'):
            if l.startswith('VmHWM'): return int(l.split()[1]) / 1024

    d = np.load(os.environ["MATRIZ"]); Xa, ya, Xb, yb = d["Xa"], d["ya"], d["Xb"], d["yb"]; base = pico_mb()
    from sklearn.neighbors import KNeighborsClassifier; from sklearn.linear_model import LogisticRegression, SGDClassifier; from sklearn.svm import SVC, LinearSVC; from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    M = {"KNN brute": lambda: KNeighborsClassifier(15, algorithm="brute"), "KNN kd_tree": lambda: KNeighborsClassifier(15, algorithm="kd_tree"),
         "XGBoost exact": lambda: XGBClassifier(tree_method="exact", n_estimators=200, max_depth=4, n_jobs=1, verbosity=0),
         "XGBoost hist": lambda: XGBClassifier(tree_method="hist", n_estimators=200, max_depth=4, n_jobs=1, verbosity=0),
         "LogReg lbfgs": lambda: LogisticRegression(solver="lbfgs", max_iter=300), "LogReg liblinear": lambda: LogisticRegression(solver="liblinear"),
         "LogReg saga": lambda: LogisticRegression(solver="saga", max_iter=300), "SVC rbf": lambda: SVC(kernel="rbf"), "LinearSVC": lambda: LinearSVC(dual=False),
         "SGD hinge": lambda: SGDClassifier(loss="hinge", random_state=0), "GaussianNB fit": lambda: GaussianNB(), "GaussianNB partial_fit": lambda: GaussianNB()}
    m = M[nombre](); t0 = time.time()
    if nombre == "GaussianNB partial_fit":
        for i in range(0, len(Xa), 4000): m.partial_fit(Xa[i:i + 4000], ya[i:i + 4000], classes=[0, 1])
    else: m.fit(Xa, ya)
    tf = time.time() - t0; t0 = time.time()
    s = m.predict_proba(Xb)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xb); tp = time.time() - t0
    print(json.dumps({"config": nombre, "t_fit_s": tf, "t_pred_s": tp, "mem_extra_MB": pico_mb() - base, "PR_AUC": average_precision_score(yb, s)}))


def paso_complejidad():
    from sklearn.model_selection import train_test_split
    from src import benchmark as B
    Xtr, ytr, _, _ = _train_test(); a, b = train_test_split(np.arange(len(Xtr)), test_size=0.3, stratify=ytr, random_state=42)
    pre = B.crear_pipeline("LogReg", "sin_balanceo", {"C": 1.0, "penalty": "l2"}).named_steps["pre"].fit(Xtr.iloc[a], ytr[a])
    Xa = np.asarray(pre.transform(Xtr.iloc[a][:32000]), dtype=float); Xb = np.asarray(pre.transform(Xtr.iloc[b][:4000]), dtype=float)
    ruta = "/tmp/claude-0/matriz_cierre.npz"; np.savez(ruta, Xa=Xa, ya=ytr[a][:32000], Xb=Xb, yb=ytr[b][:4000]); filas = []
    for c in CONFIGS:
        r = subprocess.run([sys.executable, __file__, "--hijo", c], capture_output=True, text=True, env={**os.environ, "MATRIZ": ruta}, timeout=1800)
        try: d = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception: d = {"config": c, "error": (r.stderr or r.stdout)[-200:]}
        d["entrenamiento"] = TEORIA[c][0]; d["memoria_modelo"] = TEORIA[c][1]; filas.append(d); print(d, flush=True)
    pd.DataFrame(filas).round(4).to_csv(OUT / "complejidad_estandar_vs_optimizada.csv", index=False); print("complejidad ok", Xa.shape, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--hijo": _hijo(sys.argv[2])
    else:
        for p in PASOS: {"cd": paso_cd, "diversidad": paso_diversidad, "finalistas": paso_finalistas, "shap_local": paso_shap_local, "semillas": paso_semillas,
                         "complejidad": paso_complejidad, "recalibracion": paso_recalibracion}[p]()
