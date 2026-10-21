"""Comparacion LIME vs SHAP sobre el XGBoost final (guia, Seccion 5).
Si 'lime' no se puede instalar con pip, copiar la carpeta lime/ del sdist a PYTHONPATH."""
import json, os, sys, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
if os.environ.get("LIME_PATH"): sys.path.append(os.environ["LIME_PATH"])
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from src import benchmark as B
from src.utils import PREDICTORAS, OBJETIVO, SEED, cargar_particiones
warnings.filterwarnings("ignore")
OUT = RAIZ / "resultados"; FIG = OUT / "figuras"
train, test = cargar_particiones()
Xtr, ytr, Xte = train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values, test[PREDICTORAS].reset_index(drop=True)
params = json.load(open(OUT / "hiperparametros_finales.json"))["XGBoost"]
pipe = B.crear_pipeline("XGBoost", "sin_balanceo", params).fit(Xtr, ytr)
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
Ztr, Zte = pre.transform(Xtr), pre.transform(Xte); nombres = list(pre.get_feature_names_out())
def var_de(n):
    n = n.split("__", 1)[1]
    for v in sorted(PREDICTORAS, key=len, reverse=True):
        if n.startswith(v): return v
    return "indicador_faltante"
vars_ = np.array([var_de(n) for n in nombres])
def agrupar(M):  # (n, columnas) -> (n, variables), sumando |valor|
    df = pd.DataFrame(np.abs(M), columns=nombres).T.groupby(vars_).sum().T
    return df
rs = np.random.RandomState(SEED); idx = rs.choice(len(Zte), 150, replace=False)
sv = shap.TreeExplainer(clf).shap_values(Zte[idx])
def correr_lime(semilla, ids):
    ex = LimeTabularExplainer(Ztr, feature_names=nombres, class_names=["no", "si"], mode="classification",
                              discretize_continuous=False, random_state=semilla)
    W = np.zeros((len(ids), len(nombres)))
    for k, i in enumerate(ids):
        e = ex.explain_instance(Zte[i], clf.predict_proba, num_features=len(nombres), num_samples=3000)
        for j, w in e.as_map()[1]: W[k, j] = w
    # LIME entrega pendientes en unidades estandarizadas; la contribucion local respecto a la media
    # (comparable con SHAP) es pendiente x valor estandarizado de la instancia.
    z = (Zte[ids] - ex.scaler.mean_) / ex.scaler.scale_
    return W * z
W1 = correr_lime(SEED, idx); W2 = correr_lime(SEED + 1, idx[:40])
g_shap, g_lime = agrupar(sv).mean(), agrupar(W1).mean()
comp = pd.DataFrame({"SHAP": g_shap, "LIME": g_lime}).sort_values("SHAP", ascending=False); comp.round(4).to_csv(OUT / "lime_vs_shap_global.csv")
rho_g = spearmanr(comp.SHAP, comp.LIME)[0]
# acuerdo local: por instancia, correlacion de Spearman entre atribuciones firmadas por variable
def firmadas(M):
    return pd.DataFrame(M, columns=nombres).T.groupby(vars_).sum().T.values
Sf, Lf = firmadas(sv), firmadas(W1)
rho_loc = np.array([spearmanr(Sf[i], Lf[i])[0] for i in range(len(idx))])
top3 = np.mean([len(set(np.argsort(-np.abs(Sf[i]))[:3]) & set(np.argsort(-np.abs(Lf[i]))[:3])) / 3 for i in range(len(idx))])
# estabilidad de LIME ante otra semilla (top 5 columnas, Jaccard)
jac = np.mean([len(set(np.argsort(-np.abs(W1[k]))[:5]) & set(np.argsort(-np.abs(W2[k]))[:5])) / len(set(np.argsort(-np.abs(W1[k]))[:5]) | set(np.argsort(-np.abs(W2[k]))[:5])) for k in range(40)])
res = {"n_instancias": len(idx), "spearman_global_SHAP_LIME": float(rho_g), "spearman_local_mediana": float(np.median(rho_loc)),
       "spearman_local_rango_intercuartil": [float(np.percentile(rho_loc, 25)), float(np.percentile(rho_loc, 75))],
       "coincidencia_top3_variables": float(top3), "estabilidad_LIME_jaccard_top5": float(jac),
       "tiempo_nota": "SHAP exacto para arboles, LIME aproxima con perturbaciones (3000 muestras por instancia)"}
json.dump(res, open(OUT / "lime_vs_shap.json", "w"), indent=1); print(json.dumps(res, indent=1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); c = comp.head(10)[::-1]
y = np.arange(len(c)); ax[0].barh(y - 0.2, c.SHAP / c.SHAP.max(), 0.4, label="SHAP"); ax[0].barh(y + 0.2, c.LIME / c.LIME.max(), 0.4, label="LIME")
ax[0].set_yticks(y); ax[0].set_yticklabels(c.index); ax[0].set_xlabel("importancia relativa"); ax[0].legend(); ax[0].set_title("Importancia global, XGBoost")
ax[1].hist(rho_loc, bins=20, color="#4393c3"); ax[1].set_xlabel("Spearman local SHAP vs LIME"); ax[1].set_title("Acuerdo por instancia (n=150)")
plt.tight_layout(); plt.savefig(FIG / "lime_vs_shap.png", dpi=130); plt.close(); print(comp.round(3).head(8))
