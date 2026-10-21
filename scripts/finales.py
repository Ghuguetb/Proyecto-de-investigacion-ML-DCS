"""Evaluacion final (UNA sola vez sobre test.csv), calibracion, SHAP, EBM global y robustez.
Finalistas: LogReg, SVM, XGBoost (todos sin balanceo, que fue lo mejor calibrado) y EBM."""
import json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
import optuna
from src import benchmark as B
from src.utils import PREDICTORAS, OBJETIVO, SEED, cargar_particiones
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"
train, test = cargar_particiones()
Xtr, ytr = train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values
Xte, yte = test[PREDICTORAS].reset_index(drop=True), test[OBJETIVO].values
wte = test["peso_muestral"].values; wtr = train["peso_muestral"].values

# ---------------- 1. Busqueda final de hiperparametros (solo con TRAIN) ----------------
sub, _ = train_test_split(np.arange(len(Xtr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=SEED)
params = json.load(open(OUT / "hiperparametros_finales.json")) if (OUT / "hiperparametros_finales.json").exists() else {}
for m in ([] if params else ["LogReg", "SVM", "XGBoost"]):
    ev = B.Evaluador(m, "sin_balanceo", Xtr.iloc[sub], ytr[sub], SEED)
    B.buscar_bayesiana(ev, m, SEED); params[m] = B.mejores_params(ev)
    print(m, params[m], flush=True)

def ebm_X(df):
    d = df.copy()
    for c in d.columns:
        if c != "edad":
            d[c] = pd.Categorical(d[c], categories=sorted(Xtr[c].dropna().unique()))
    return d
tipos = ["continuous" if c == "edad" else "nominal" for c in PREDICTORAS]
def crear_ebm(p):
    return ExplainableBoostingClassifier(feature_types=tipos, random_state=SEED, n_jobs=1, outer_bags=4, **p)
from sklearn.model_selection import cross_val_score
cvi = StratifiedKFold(3, shuffle=True, random_state=SEED); Xs_e, ys_e = ebm_X(Xtr).iloc[sub], ytr[sub]
def obj(t):
    p = {"learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True), "max_leaves": t.suggest_int("max_leaves", 2, 5),
         "interactions": t.suggest_int("interactions", 0, 10), "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100, log=True)}
    return float(np.mean(cross_val_score(crear_ebm(p), Xs_e, ys_e, cv=cvi, scoring="average_precision")))
if "EBM" not in params:
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4))
    est.optimize(obj, n_trials=B.PRESUPUESTO); params["EBM"] = est.best_params; print("EBM", params["EBM"], flush=True)
json.dump(params, open(OUT / "hiperparametros_finales.json", "w"), indent=1, default=float)

# ---------------- 2. Ajuste final en TRAIN completo y evaluacion UNICA en TEST ----------------
modelos, probs = {}, {}
for m in ["LogReg", "SVM", "XGBoost"]:
    modelos[m] = B.crear_pipeline(m, "sin_balanceo", params[m]).fit(Xtr, ytr); probs[m] = modelos[m].predict_proba(Xte)[:, 1]
modelos["EBM"] = crear_ebm(params["EBM"]).fit(ebm_X(Xtr), ytr); probs["EBM"] = modelos["EBM"].predict_proba(ebm_X(Xte))[:, 1]
rng = np.random.RandomState(SEED); idx_b = [rng.randint(0, len(yte), len(yte)) for _ in range(1000)]
filas = []
for m, p in probs.items():
    pt = B.metricas(yte, p); bs = pd.DataFrame([B.metricas(yte[i], p[i]) for i in idx_b])
    filas.append({"modelo": m, **{k: f"{pt[k]:.3f} [{bs[k].quantile(.025):.3f}-{bs[k].quantile(.975):.3f}]" for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE"]}})
pd.DataFrame(filas).to_csv(OUT / "test_final_metricas.csv", index=False); print(pd.DataFrame(filas).to_string(index=False), flush=True)
np.save(OUT / "test_probs.npy", np.vstack([yte] + [probs[m] for m in probs]))

# DeLong en test entre finalistas (con Holm)
from src.estadistica import delong_p, holm
pares = [("LogReg", "SVM"), ("LogReg", "XGBoost"), ("LogReg", "EBM"), ("XGBoost", "EBM"), ("SVM", "EBM")]
dl = []
for a, b in pares:
    d, p = delong_p(yte, probs[a], probs[b]); dl.append((a, b, d, p))
dl = pd.DataFrame(dl, columns=["A", "B", "dif_AUC", "p"]); dl["p_Holm"] = holm(dl["p"]); dl.to_csv(OUT / "test_delong.csv", index=False)
print(dl.round(4).to_string(index=False), flush=True)

# ---------------- 3. Calibracion ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot([0, 1], [0, 1], ":", c="gray")
for m, p in probs.items():
    fp, pm = calibration_curve(yte, p, n_bins=10, strategy="quantile"); ax[0].plot(pm, fp, "o-", label=f"{m} (Brier {brier_score_loss(yte, p):.3f})")
ax[0].set_xlabel("probabilidad predicha"); ax[0].set_ylabel("proporcion observada"); ax[0].legend(fontsize=8); ax[0].set_title("Calibracion en el conjunto de prueba")
# efecto del balanceo sobre calibracion (CV externa)
fo = pd.read_csv(OUT / "resultados_por_fold.csv"); fo = fo[fo.modelo != "EBM"]
t = fo.groupby("balanceo")["Brier"].mean().sort_values(); ax[1].bar(t.index, t.values, color="#4393c3"); ax[1].set_ylabel("Brier medio (CV externa)"); ax[1].set_title("El balanceo empeora la calibracion")
plt.tight_layout(); plt.savefig(FIG / "calibracion.png", dpi=130); plt.close()
fo.groupby(["modelo", "balanceo"])[["Brier", "ECE"]].mean().round(4).to_csv(OUT / "calibracion_por_modelo_balanceo.csv")

# ---------------- 4. SHAP (XGBoost) + coeficientes (LogReg) + EBM global ----------------
import shap
pre = modelos["XGBoost"].named_steps["pre"]; Xp = pre.transform(Xte); nombres = list(pre.get_feature_names_out())
def var_de(n):
    n = n.split("__", 1)[1]
    for v in sorted(PREDICTORAS, key=len, reverse=True):
        if n.startswith(v): return v
    return "indicador_faltante:" + n
rs = np.random.RandomState(SEED).choice(len(Xp), 3000, replace=False)
sv = shap.TreeExplainer(modelos["XGBoost"].named_steps["clf"]).shap_values(Xp[rs])
imp_cols = pd.Series(np.abs(sv).mean(0), index=nombres)
imp_var = imp_cols.groupby([var_de(n) for n in nombres]).sum().sort_values(ascending=False)
plt.figure(figsize=(7, 5)); shap.summary_plot(sv, Xp[rs], feature_names=nombres, max_display=15, show=False); plt.tight_layout(); plt.savefig(FIG / "shap_xgboost.png", dpi=130); plt.close()
coef = pd.Series(np.abs(modelos["LogReg"].named_steps["clf"].coef_[0]), index=nombres).groupby([var_de(n) for n in nombres]).sum()
ebm_imp = pd.Series(modelos["EBM"].term_importances(), index=modelos["EBM"].term_names_)
comp = pd.DataFrame({"SHAP_XGBoost": imp_var, "abs_coef_LogReg": coef}).join(ebm_imp.rename("EBM_importancia")).fillna(0)
comp.sort_values("SHAP_XGBoost", ascending=False).round(4).to_csv(OUT / "importancia_variables.csv")
from scipy.stats import spearmanr
sp = {"SHAP_vs_LogReg": spearmanr(comp.SHAP_XGBoost, comp.abs_coef_LogReg)[0], "SHAP_vs_EBM": spearmanr(comp.SHAP_XGBoost, comp.EBM_importancia)[0],
      "LogReg_vs_EBM": spearmanr(comp.abs_coef_LogReg, comp.EBM_importancia)[0]}
json.dump(sp, open(OUT / "acuerdo_importancias_spearman.json", "w"), indent=1); print("Spearman importancias", sp, flush=True)
ex = modelos["EBM"].explain_global()
i_edad = list(modelos["EBM"].term_names_).index("edad"); d = ex.data(i_edad)
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].step(d["names"][:len(d["scores"])], d["scores"], where="post", c="#b2182b")
ax[0].set_xlabel("edad"); ax[0].set_ylabel("contribucion al log-odds"); ax[0].set_title("EBM, funcion de forma de la edad (forma de U)")
top = ebm_imp.sort_values().tail(12); ax[1].barh(top.index, top.values, color="#4393c3"); ax[1].set_title("EBM, importancia por termino")
plt.tight_layout(); plt.savefig(FIG / "ebm_global.png", dpi=130); plt.close()

# ---------------- 5. Robustez ----------------
rob = []
# (a) pesos muestrales en la evaluacion
for m, p in probs.items():
    rob.append(("pesos_muestrales_en_evaluacion", m, "PR_AUC sin pesos", average_precision_score(yte, p)))
    rob.append(("pesos_muestrales_en_evaluacion", m, "PR_AUC con pesos", average_precision_score(yte, p, sample_weight=wte)))
    rob.append(("pesos_muestrales_en_evaluacion", m, "Brier sin pesos", brier_score_loss(yte, p)))
    rob.append(("pesos_muestrales_en_evaluacion", m, "Brier con pesos", brier_score_loss(yte, p, sample_weight=wte)))
# (b) reajuste de la logistica con pesos
pw = B.crear_pipeline("LogReg", "sin_balanceo", params["LogReg"]); pw.fit(Xtr, ytr, clf__sample_weight=wtr / wtr.mean()); pp = pw.predict_proba(Xte)[:, 1]
rob.append(("reajuste_con_pesos", "LogReg", "PR_AUC (eval. con pesos)", average_precision_score(yte, pp, sample_weight=wte)))
rob.append(("reajuste_con_pesos", "LogReg", "Brier (eval. con pesos)", brier_score_loss(yte, pp, sample_weight=wte)))
# (c) umbral: elegido fuera de particion en train
oof = cross_val_predict(modelos["LogReg"], Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
from sklearn.metrics import precision_recall_curve
pr, rc, th = precision_recall_curve(ytr, oof); f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None); u = float(th[np.argmax(f1s)])
for t_ in [0.15, 0.2, u, 0.3, 0.4, 0.5]:
    yh = (probs["LogReg"] >= t_).astype(int)
    rob.append(("umbral_LogReg", f"{t_:.3f}" + (" (F1 OOF)" if t_ == u else ""), "F1 / recall / precision", f"{f1_score(yte, yh):.3f} / {recall_score(yte, yh):.3f} / {precision_score(yte, yh):.3f}"))
# (d) datos faltantes
falt = Xte.isna().any(axis=1).values
for m, p in probs.items():
    rob.append(("faltantes", m, f"ROC_AUC con algun faltante (n={falt.sum()})", roc_auc_score(yte[falt], p[falt])))
    rob.append(("faltantes", m, f"ROC_AUC casos completos (n={(~falt).sum()})", roc_auc_score(yte[~falt], p[~falt])))
# (e) perturbaciones
def perturbar(df, sigma_edad, p_flip, rs):
    d = df.copy(); d["edad"] = (d["edad"] + rs.normal(0, sigma_edad, len(d))).clip(18, 85)
    for c in [c for c in PREDICTORAS if c not in ("edad", "sexo", "raza_etnia")]:
        m_ = rs.rand(len(d)) < p_flip; vals = Xtr[c].dropna().unique()
        d.loc[m_, c] = rs.choice(vals, m_.sum())
    return d
for nombre, (s_, f_) in {"ruido_leve (edad+-2, 5% cambiado)": (2, 0.05), "ruido_fuerte (edad+-5, 15% cambiado)": (5, 0.15)}.items():
    for m in probs:
        aucs = []
        for r in range(3):
            Xp_ = perturbar(Xte, s_, f_, np.random.RandomState(SEED + r))
            pr_ = modelos[m].predict_proba(ebm_X(Xp_) if m == "EBM" else Xp_)[:, 1]; aucs.append(roc_auc_score(yte, pr_))
        rob.append(("perturbacion_de_entradas", m, nombre + " ROC_AUC", float(np.mean(aucs))))
# (f) subgrupos
prb = probs["LogReg"]; grp = {"edad": pd.cut(Xte["edad"], [17, 39, 59, 74, 90]).astype(str), "sexo": Xte["sexo"], "raza_etnia": Xte["raza_etnia"]}
for g, serie in grp.items():
    for nivel in sorted(serie.dropna().unique()):
        mk = (serie == nivel).values
        if mk.sum() > 100 and 0 < yte[mk].mean() < 1:
            rob.append(("subgrupos_LogReg", f"{g}={nivel} (n={mk.sum()})", "ROC_AUC / Brier / prev. obs. / prev. pred.", f"{roc_auc_score(yte[mk], prb[mk]):.3f} / {brier_score_loss(yte[mk], prb[mk]):.3f} / {yte[mk].mean():.3f} / {prb[mk].mean():.3f}"))
# (g) temporal 2022 -> 2023 (solo con datos de train)
a22 = (train["anio"] == 2022).values
for m in ["LogReg", "XGBoost", "EBM"]:
    if m == "EBM":
        mod = crear_ebm(params["EBM"]).fit(ebm_X(Xtr[a22]), ytr[a22]); p_ = mod.predict_proba(ebm_X(Xtr[~a22]))[:, 1]
    else:
        mod = B.crear_pipeline(m, "sin_balanceo", params[m]).fit(Xtr[a22], ytr[a22]); p_ = mod.predict_proba(Xtr[~a22])[:, 1]
    rob.append(("temporal_2022_a_2023", m, "PR_AUC / ROC_AUC", f"{average_precision_score(ytr[~a22], p_):.3f} / {roc_auc_score(ytr[~a22], p_):.3f}"))
pd.DataFrame(rob, columns=["analisis", "modelo_o_grupo", "metrica", "valor"]).to_csv(OUT / "robustez.csv", index=False)
print(pd.DataFrame(rob, columns=["analisis", "modelo_o_grupo", "metrica", "valor"]).to_string(index=False), flush=True)
print("FIN")
