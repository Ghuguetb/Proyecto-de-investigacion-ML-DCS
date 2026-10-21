"""Modelo original: Explainable Boosting Machine (EBM, paquete interpret) con el MISMO nested CV.
El EBM maneja nativamente categorias y faltantes, asi que se alimenta con las predictoras crudas
(sin one-hot ni splines): cada variable conserva su propia funcion de forma interpretable."""
import json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np, optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from src import benchmark as B
from src.utils import PREDICTORAS, SEED
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

carpeta = RAIZ / "resultados" / "tareas"; carpeta.mkdir(parents=True, exist_ok=True)
X, y = B.cargar_xy()
for c in X.columns:
    if c != "edad":
        X[c] = X[c].astype("category")
tipos = ["continuous" if c == "edad" else "nominal" for c in PREDICTORAS]
RAPIDO = "--rapido" in sys.argv

def crear(p):
    return ExplainableBoostingClassifier(feature_types=tipos, random_state=SEED, n_jobs=1, outer_bags=4,
                                         max_rounds=p.get("max_rounds", 2000), **{k: v for k, v in p.items() if k != "max_rounds"})

for fold, (tr, va) in enumerate(B.indices_externos(y)):
    nombre = f"EBM__sin_balanceo__Bayesiana__f{fold}"
    if (carpeta / f"{nombre}.json").exists():
        continue
    t0 = time.time()
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    sub, _ = train_test_split(np.arange(len(tr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=SEED + fold)
    Xs, ys = Xtr.iloc[sub], ytr[sub]
    cv = StratifiedKFold(B.N_INTERNO, shuffle=True, random_state=SEED + fold)
    hist, mejor = [], [-1]
    def obj(trial):
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
             "max_leaves": trial.suggest_int("max_leaves", 2, 5),
             "interactions": trial.suggest_int("interactions", 0, 10),
             "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True)}
        s = float(np.mean(cross_val_score(crear(p), Xs, ys, cv=cv, scoring="average_precision")))
        mejor[0] = max(mejor[0], s)
        hist.append({"eval": len(hist) + 1, "score": s, "mejor_hasta_ahora": mejor[0], "t": time.time() - t0, "params": p})
        return s
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + fold, n_startup_trials=4))
    est.optimize(obj, n_trials=4 if RAPIDO else B.PRESUPUESTO)
    t_busq = time.time() - t0
    bp = est.best_params
    t1 = time.time(); modelo = crear(bp).fit(Xtr, ytr); t_fit = time.time() - t1
    p = modelo.predict_proba(Xva)[:, 1]
    res = {"modelo": "EBM", "balanceo": "sin_balanceo", "optimizador": "Bayesiana", "fold": fold,
           **B.metricas(yva, p), "PR_AUC_interno": mejor[0], "n_evals": len(hist), "t_busqueda_s": t_busq,
           "t_refit_s": t_fit, "t_pred_s": 0.0, "balanceo_aplicable": True, "mejores_params": bp, "historia": hist,
           "diversidad_ga": None}
    np.save(carpeta / f"{nombre}.npy", np.stack([va, p.astype(np.float32)]).astype(np.float32))
    (carpeta / f"{nombre}.json").write_text(json.dumps(res, default=str))
    print(f"fold {fold}: PR-AUC={res['PR_AUC']:.3f} ROC-AUC={res['ROC_AUC']:.3f} Brier={res['Brier']:.3f} t={time.time()-t0:.0f}s", flush=True)
    if fold == 0 and RAPIDO:
        break
