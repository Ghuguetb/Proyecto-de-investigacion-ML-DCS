"""Tabla maestra + estadistica jerarquica (Friedman -> Nemenyi/CD -> DeLong con Holm) + tamanos de efecto."""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
RAIZ = Path(__file__).resolve().parents[1]
TAR = RAIZ / "resultados" / "tareas"; OUT = RAIZ / "resultados"

# ---------------- 1. Tabla maestra -----------------
filas = []
for f in sorted(TAR.glob("*.json")):
    r = json.loads(f.read_text()); r.pop("historia"); r.pop("diversidad_ga"); filas.append(r)
fold = pd.DataFrame(filas)
fold.to_csv(OUT / "resultados_por_fold.csv", index=False)
ebm = fold[fold["modelo"] == "EBM"].copy()
fold = fold[fold["modelo"] != "EBM"].copy()  # la rejilla 7x4x4 se analiza sin el modelo original
llaves = ["modelo", "balanceo", "optimizador"]
met = ["PR_AUC", "ROC_AUC", "Brier", "ECE", "F1_05", "Recall_05"]
g = fold.groupby(llaves)
maestra = g[met].mean().add_suffix("_media").join(g[met].std().add_suffix("_sd")).join(
    g.agg(n_folds=("fold", "count"), evals=("n_evals", "mean"), t_busqueda_s=("t_busqueda_s", "sum"),
          t_refit_s=("t_refit_s", "sum"), PR_AUC_interno=("PR_AUC_interno", "mean"))).reset_index()
maestra["t_total_min"] = (maestra["t_busqueda_s"] + maestra["t_refit_s"]) / 60
maestra["rango_PR_AUC"] = maestra["PR_AUC_media"].rank(ascending=False, method="min")
maestra = maestra.sort_values("rango_PR_AUC")
maestra.to_csv(OUT / "tabla_maestra.csv", index=False)
print(f"{len(maestra)} combinaciones en la tabla maestra (con KNN class_weight excluido por no aplicable)")
print(maestra.head(10)[llaves + ["PR_AUC_media", "PR_AUC_sd", "ROC_AUC_media", "Brier_media", "t_total_min"]].round(3).to_string(index=False))

# ---------------- 2. Friedman (bloques = folds externos) -----------------
fold["combo"] = fold["modelo"] + "|" + fold["balanceo"] + "|" + fold["optimizador"]
M = fold.pivot(index="fold", columns="combo", values="PR_AUC").dropna(axis=1)
k, N = M.shape[1], M.shape[0]
chi, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
ranks = M.rank(axis=1, ascending=False).mean()
print(f"\nFriedman (PR-AUC): k={k} combinaciones, N={N} folds, chi2={chi:.1f}, p={p:.3g}")
# Nemenyi: diferencia critica
from scipy.stats import studentized_range
q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * N))
print(f"Diferencia critica de Nemenyi (alfa=0.05): CD = {CD:.1f} rangos (con k={k}, N={N})")
ranks.sort_values().to_frame("rango_medio").to_csv(OUT / "friedman_rangos.csv")

# Friedman por factor (promedio sobre los otros factores dentro de cada fold)
res_f = {}
for factor in ["modelo", "balanceo", "optimizador"]:
    P = fold.groupby(["fold", factor])["PR_AUC"].mean().unstack()
    chi_f, p_f = stats.friedmanchisquare(*[P[c].values for c in P.columns])
    r = P.rank(axis=1, ascending=False).mean().sort_values()
    kf, nf = P.shape[1], P.shape[0]
    cd_f = studentized_range.ppf(0.95, kf, np.inf) / np.sqrt(2) * np.sqrt(kf * (kf + 1) / (6 * nf))
    res_f[factor] = {"chi2": chi_f, "p": p_f, "CD": cd_f, "rangos": r.round(2).to_dict(), "PR_AUC_media": P.mean().round(4).to_dict()}
json.dump(res_f, open(OUT / "friedman_por_factor.json", "w"), indent=1, default=float)
for f_, v in res_f.items():
    print(f"\nFactor {f_}: chi2={v['chi2']:.2f} p={v['p']:.3g} CD={v['CD']:.2f}\n  rangos: {v['rangos']}")

# ---------------- 3. DeLong + Holm contra el mejor modelo -----------------
def oof(nombre):
    v = np.full(44903, np.nan, dtype=np.float32); y = np.full(44903, -1)
    return None

def cargar_oof(combo):
    m, b, o = combo.split("|")
    idx, pr = [], []
    for fo in range(5):
        a = np.load(TAR / f"{m}__{b}__{o}__f{fo}.npy")
        idx.append(a[0].astype(int)); pr.append(a[1])
    idx, pr = np.concatenate(idx), np.concatenate(pr)
    o_ = np.argsort(idx); return pr[o_]

def midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]: a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1; i = a
    out = np.empty(n); out[j] = t; return out

def delong_cov(y, preds):
    pos = preds[:, y == 1]; neg = preds[:, y == 0]; m, n = pos.shape[1], neg.shape[1]; K = preds.shape[0]
    tx = np.array([midrank(pos[r]) for r in range(K)]); ty = np.array([midrank(neg[r]) for r in range(K)])
    tz = np.array([midrank(preds[r]) for r in range(K)])
    aucs = tz[:, :0].sum(1)  # placeholder
    aucs = np.array([(tz[r, y == 1].sum() - m * (m + 1) / 2) / (m * n) for r in range(K)])
    v01 = (tz[:, y == 1] - tx) / n; v10 = 1 - (tz[:, y == 0] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    return aucs, s01 / m + s10 / n

def delong_p(y, p1, p2):
    aucs, S = delong_cov(y, np.vstack([p1, p2]))
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-18))
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))

def holm(pv):
    pv = np.asarray(pv); o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * pv[i]); adj[i] = min(1, run)
    return adj

# y global en el orden original de train
sys.path.insert(0, str(RAIZ))
from src.utils import cargar_particiones, OBJETIVO
y = cargar_particiones()[0][OBJETIVO].values
orden = maestra.assign(combo=maestra["modelo"] + "|" + maestra["balanceo"] + "|" + maestra["optimizador"])["combo"].tolist()
mejor = orden[0]; top = orden[1:int(sys.argv[1]) if len(sys.argv) > 1 else 20]
pm = cargar_oof(mejor)
filas = []
for c in top:
    d, pv = delong_p(y, pm, cargar_oof(c)); filas.append((c, d, pv))
dl = pd.DataFrame(filas, columns=["combo", "dif_AUC_vs_mejor", "p"]); dl["p_Holm"] = holm(dl["p"])
dl.to_csv(OUT / "delong_vs_mejor.csv", index=False)
print(f"\nDeLong (ROC-AUC OOF) del mejor por PR-AUC ({mejor}) vs los {len(top)} siguientes, con Holm:")
print(dl.round(4).to_string(index=False))

# ---------------- 4. Tamano de efecto: Cliff's delta sobre PR-AUC por fold -----------------
def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b))
ef = [(c, cliff(M[mejor], M[c])) for c in top if c in M.columns]
pd.DataFrame(ef, columns=["combo", "cliffs_delta_mejor_vs"]).to_csv(OUT / "cliffs_delta.csv", index=False)

# ---------------- 5. Modelo original (EBM) aparte -----------------
if len(ebm) == 5:
    print("\nEBM (modelo original, mismo nested CV):", ebm[["PR_AUC", "ROC_AUC", "Brier", "ECE"]].mean().round(4).to_dict())
    pe = cargar_oof("EBM|sin_balanceo|Bayesiana")
    filas = []
    ref = {"mejor_PR_AUC": mejor,
           "mejor_LogReg": maestra[maestra["modelo"] == "LogReg"].assign(c=lambda d: d.modelo + "|" + d.balanceo + "|" + d.optimizador).iloc[0]["c"],
           "mejor_XGBoost": maestra[maestra["modelo"] == "XGBoost"].assign(c=lambda d: d.modelo + "|" + d.balanceo + "|" + d.optimizador).iloc[0]["c"]}
    for nombre, c in ref.items():
        d, pv = delong_p(y, pe, cargar_oof(c))
        delta = cliff(ebm["PR_AUC"].values, M[c].values)
        filas.append((nombre, c, d, pv, delta))
    de = pd.DataFrame(filas, columns=["referencia", "combo", "dif_AUC_EBM_menos_ref", "p", "cliffs_delta_PR_AUC"])
    de["p_Holm"] = holm(de["p"]); de.to_csv(OUT / "ebm_vs_referencias.csv", index=False)
    print(de.round(4).to_string(index=False))
