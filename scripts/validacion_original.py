"""Verificacion estadistica de la superioridad del modelo original (Bloque D, condicion R3).

Protocolo comun: mismas particiones externas, misma semilla, mismo preprocesamiento y mismo presupuesto de busqueda,
con validacion cruzada externa repetida (5 repeticiones x 5 pliegues = 25 estimaciones pareadas por modelo).

Pruebas (datos de corte transversal, Cuadro 3 del documento del profesor):
  P1 Friedman + Nemenyi (diagrama de diferencia critica), modelo original incluido, 25 bloques.     [global]
  P2 Wilcoxon de rangos con signo pareado, original frente a cada mejor benchmark, Holm.
  P3 t corregida de Nadeau y Bengio (validacion repetida), original frente a cada benchmark, Holm.
  P4 DeLong (AUC) en la prueba, original frente a cada finalista, Holm.
  P5 McNemar (umbral 0.245) en la prueba, Holm.
Tamano del efecto Cliff's delta y d de Cohen pareada. IC bootstrap BCa de la diferencia media de PR-AUC (25 diferencias)."""
import json, sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from src.estadistica import delong_p, holm
OUT = RAIZ / "resultados"; REP = OUT / "repetido"; TAR = OUT / "tareas"; FIG = OUT / "figuras"
UMBRAL = 0.245
BASE = {"SVM": ("sin_balanceo", "Grid"), "XGBoost": ("class_weight", "Bayesiana"), "LogReg": ("sin_balanceo", "Random"),
        "RandomForest": ("sin_balanceo", "Random"), "DecisionTree": ("sin_balanceo", "Grid"),
        "NaiveBayes": ("SMOTE", "Grid"), "KNN": ("sin_balanceo", "Bayesiana")}

# ---------------- datos por pliegue ----------------
filas = []
for f in REP.glob("*.json"):
    r = json.loads(f.read_text()); filas.append({k: r[k] for k in ["modelo", "rep", "fold", "PR_AUC", "ROC_AUC", "Brier", "ECE", "t_busqueda_s", "t_refit_s", "t_pred_s"]})
for m, (bal, opt) in {**BASE, "EBM": ("sin_balanceo", "Bayesiana")}.items():  # repeticion 0 = Entregable 2
    for fold in range(5):
        r = json.loads((TAR / f"{m}__{bal}__{opt}__f{fold}.json").read_text())
        filas.append({"modelo": m, "rep": 0, "fold": fold, **{k: r[k] for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE", "t_busqueda_s", "t_refit_s", "t_pred_s"]}})
D = pd.DataFrame(filas); D.to_csv(OUT / "original_resultados_por_fold.csv", index=False)
modelos = sorted(D.modelo.unique())
pr = D.pivot_table(index=["rep", "fold"], columns="modelo", values="PR_AUC")
sin_ebm = pr.drop(columns=["EBM"], errors="ignore").dropna()
con_ebm = pr.dropna()
print("bloques completos con EBM:", len(con_ebm), "sin EBM:", len(sin_ebm), flush=True)
ORIG = "AditivoPenalizado"
refs = [m for m in pr.columns if m != ORIG]

# ---------------- tabla resumen ----------------
def bca(v):
    return stats.bootstrap((np.asarray(v),), np.mean, confidence_level=0.95, method="BCa", n_resamples=5000, random_state=42).confidence_interval
res = []
for m in pr.columns:
    v = pr[m].dropna().to_numpy(); lo, hi = bca(v); d = D[D.modelo == m]
    res.append({"modelo": m, "n_estimaciones": len(v), "PR_AUC_media": v.mean(), "PR_AUC_sd": v.std(ddof=1), "IC95_BCa_inf": lo, "IC95_BCa_sup": hi,
                "ROC_AUC_media": d.ROC_AUC.mean(), "Brier_media": d.Brier.mean(), "ECE_media": d.ECE.mean(),
                "t_entrenamiento_s": (d.t_refit_s).mean(), "t_inferencia_s": d.t_pred_s.mean()})
resumen = pd.DataFrame(res).sort_values("PR_AUC_media", ascending=False); resumen.round(4).to_csv(OUT / "original_resumen.csv", index=False)
print(resumen.round(4).to_string(index=False), flush=True)
o = D[D.modelo == ORIG]; ro = {k: [float(o[k].mean()), float(o[k].std(ddof=1))] for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE"]}
pd.DataFrame([{**{f"{k}_media": v[0] for k, v in ro.items()}, **{f"{k}_sd": v[1] for k, v in ro.items()}}]).to_csv(OUT / "original_metricas_externas.csv", index=False)

# ---------------- P1 Friedman y Nemenyi (global) ----------------
def friedman_nemenyi(M, etiqueta):
    k, N = M.shape[1], M.shape[0]
    est, p = stats.friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
    rangos = M.rank(axis=1, ascending=False).mean()
    cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    return {"universo": etiqueta, "k_modelos": k, "N_bloques": N, "chi2": est, "p": p, "CD_nemenyi": cd}, rangos.sort_values()
f1a, rg_a = friedman_nemenyi(sin_ebm, "7 mejores + original, 25 bloques (5 rep x 5 pliegues)")
f1b, rg_b = friedman_nemenyi(con_ebm, "7 mejores + EBM + original")
pd.DataFrame([f1a, f1b]).round(5).to_csv(OUT / "original_friedman.csv", index=False)
rg_b.rename("rango_medio").to_csv(OUT / "original_rangos.csv")
print("Friedman", f1a, "\n", rg_a.round(2).to_dict(), flush=True)
def dibujar_cd(rangos, cd, ruta, titulo):
    r = rangos.sort_values(); k = len(r)
    fig, ax = plt.subplots(figsize=(9, 2.9)); ax.set_xlim(0.8, k + 0.2); ax.set_ylim(0, 1); ax.axis("off")
    ax.hlines(0.62, 1, k, color="black", lw=1)
    for t in range(1, k + 1):
        ax.vlines(0.62 + 0, t, t, color="black"); ax.text(t, 0.68, str(t), ha="center", fontsize=8)
    for i, (n, v) in enumerate(r.items()):
        ax.plot([v, v], [0.62, 0.55 - 0.04 * (i % 4)], color="#1F2A80", lw=0.8)
        ax.text(v, 0.50 - 0.04 * (i % 4), f"{n} ({v:.2f})", ha="center", va="top", fontsize=7.5, color="#C8503A" if n == ORIG else "black")
    ax.hlines(0.9, 1, 1 + cd, color="#C8503A", lw=2); ax.text(1 + cd / 2, 0.93, f"CD = {cd:.2f}", ha="center", fontsize=8)
    vals = r.values; y0 = 0.80
    for i in range(k):                       # barras entre modelos cuya diferencia de rango es menor que CD
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd: j += 1
        if j > i and (i == 0 or vals[i - 1] < vals[j] - cd or True):
            ax.hlines(y0, vals[i], vals[j], color="black", lw=2.5); y0 -= 0.05
    ax.set_title(titulo, fontsize=9); plt.tight_layout(); plt.savefig(ruta, dpi=130); plt.close()
dibujar_cd(rg_a, f1a["CD_nemenyi"], FIG / "original_cd.png", "Diferencia critica (Nemenyi), rango 1 = mejor PR-AUC, 25 bloques")
nem = []
for a in rg_a.index:
    for b in rg_a.index:
        if a < b: nem.append({"a": a, "b": b, "dif_rangos": abs(rg_a[a] - rg_a[b]), "significativa": abs(rg_a[a] - rg_a[b]) > f1a["CD_nemenyi"]})
pd.DataFrame(nem).round(3).to_csv(OUT / "original_nemenyi.csv", index=False)

# ---------------- P2, P3 pareadas por pliegue ----------------
def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))
J = len(sin_ebm); fr_test = 1 / 4                       # n_test / n_train de un 5-fold
filas = []
for r_ in refs:
    M = pr[[ORIG, r_]].dropna(); d = (M[ORIG] - M[r_]).to_numpy(); n = len(d)
    w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    t = d.mean() / np.sqrt((1 / n + fr_test) * d.var(ddof=1)); p_nb = 2 * stats.t.sf(abs(t), n - 1)
    lo, hi = bca(d)
    filas.append({"referencia": r_, "n_pares": n, "dif_media_PR_AUC": d.mean(), "IC95_BCa_inf": lo, "IC95_BCa_sup": hi, "wilcoxon_p": w.pvalue,
                  "nadeau_bengio_t": t, "nadeau_bengio_p": p_nb, "cliffs_delta": cliff(M[ORIG], M[r_]), "cohen_dz": d.mean() / d.std(ddof=1)})
P = pd.DataFrame(filas)
P["wilcoxon_p_holm"] = holm(P.wilcoxon_p.values); P["nadeau_bengio_p_holm"] = holm(P.nadeau_bengio_p.values)
P = P.sort_values("dif_media_PR_AUC", ascending=False); P.round(5).to_csv(OUT / "original_pruebas_pareadas.csv", index=False)
print(P.round(4).to_string(index=False), flush=True)

# ---------------- P4 DeLong y P5 McNemar en la prueba ----------------
a = np.load(OUT / "test_probs.npy"); y = a[0].astype(int); pt = dict(zip(["LogReg", "SVM", "XGBoost", "EBM"], a[1:5]))
b = np.load(OUT / "test_probs_original.npy"); assert np.array_equal(b[0].astype(int), y); po = b[1]
filas = []
for r_, pr_ in pt.items():
    dif, p_d = delong_p(y, po, pr_)
    yo, yr = (po >= UMBRAL), (pr_ >= UMBRAL)
    n01 = int(((yo == y) & (yr != y)).sum()); n10 = int(((yo != y) & (yr == y)).sum())
    p_m = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    filas.append({"referencia": r_, "dif_AUC": dif, "delong_p": p_d, "mcnemar_orig_acierta_ref_falla": n01, "mcnemar_orig_falla_ref_acierta": n10, "mcnemar_p": p_m})
T = pd.DataFrame(filas); T["delong_p_holm"] = holm(T.delong_p.values); T["mcnemar_p_holm"] = holm(T.mcnemar_p.values)
T.round(5).to_csv(OUT / "original_pruebas_prueba.csv", index=False); print(T.round(4).to_string(index=False), flush=True)

# ---------------- Veredicto R3 ----------------
# Condicion 1: la media de PR-AUC del original es mayor que la de la mejor combinacion de cada uno de los 7 modelos base (y que la del EBM)
mejor_ref = pr[[c for c in pr.columns if c != ORIG]].mean()
gana_media = {r_: bool(pr[ORIG].mean() > mejor_ref[r_]) for r_ in refs}
c1 = all(gana_media.values())
# Condicion 2: al menos tres pruebas distintas rechazan la igualdad A FAVOR del original frente a TODAS las referencias (p ajustado < 0.05)
rangos_otros = rg_a[[c for c in rg_a.index if c != ORIG]]
pruebas = {"Friedman y Nemenyi (diferencia de rangos mayor que CD frente a todos)": bool(((rangos_otros - rg_a[ORIG]) > f1a["CD_nemenyi"]).all()),
           "Wilcoxon pareado con Holm": bool(((P.wilcoxon_p_holm < .05) & (P.dif_media_PR_AUC > 0)).all()),
           "t corregida de Nadeau y Bengio con Holm": bool(((P.nadeau_bengio_p_holm < .05) & (P.dif_media_PR_AUC > 0)).all()),
           "DeLong en la prueba con Holm": bool(((T.delong_p_holm < .05) & (T.dif_AUC > 0)).all()),
           "McNemar en la prueba con Holm": bool(((T.mcnemar_p_holm < .05) & (T.mcnemar_orig_acierta_ref_falla > T.mcnemar_orig_falla_ref_acierta)).all())}
c2 = sum(pruebas.values()) >= 3
# Condicion 3: el tamano del efecto no es despreciable (|Cliff| >= 0.147 o |d| >= 0.2) y a favor del original frente a todas las referencias
c3 = bool(((P.dif_media_PR_AUC > 0) & ((P.cliffs_delta.abs() >= 0.147) | (P.cohen_dz.abs() >= 0.2))).all())
ver = {"PR_AUC_media_original": float(pr[ORIG].mean()), "PR_AUC_media_por_referencia": {k: float(v) for k, v in mejor_ref.items()},
       "el_original_gana_en_media_a": gana_media, "condicion_1_media_mayor_que_todas": c1,
       "pruebas_realizadas": len(pruebas), "pruebas_que_rechazan_a_favor_del_original": pruebas, "condicion_2_al_menos_tres_pruebas": c2,
       "condicion_3_efecto_no_despreciable_a_favor": c3, "R3_cumplida": bool(c1 and c2 and c3),
       "comparaciones_totales": int(len(P) * 2 + len(T) * 2 + 1)}
json.dump(ver, open(OUT / "original_veredicto.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(ver, indent=1, ensure_ascii=False))
