"""Nested CV repetido del modelo original (AditivoPenalizado) y de las mejores combinaciones del benchmark.

Cinco repeticiones de la validacion externa de 5 pliegues (25 estimaciones pareadas). La repeticion 0 usa exactamente
los mismos pliegues y semillas del Entregable 2 (B.indices_externos, semilla 42 + fold). Las repeticiones 1 a 4 usan
otra particion externa con semilla 42 + 1000 r y semilla de busqueda 42 + 1000 r + fold. Todos los modelos de una
repeticion comparten particion externa, preprocesamiento y presupuesto de busqueda (12 evaluaciones, TPE, submuestra
de 8,000 filas, 3 pliegues internos). Uso: python scripts/run_original.py <repeticion> [<repeticion> ...]"""
import json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import numpy as np, optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from src import benchmark as B
from src.original_model import AditivoPenalizado
from src.utils import ORDENES, PREDICTORAS, SEED
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = RAIZ / "resultados" / "repetido"; OUT.mkdir(parents=True, exist_ok=True)

# Mejores combinaciones del Entregable 2 por modelo base (criterio, PR-AUC media del bucle externo)
MEJORES = {"SVM": ("sin_balanceo", "Grid"), "XGBoost": ("class_weight", "Bayesiana"), "LogReg": ("sin_balanceo", "Random"),
           "RandomForest": ("sin_balanceo", "Random"), "DecisionTree": ("sin_balanceo", "Grid"),
           "NaiveBayes": ("SMOTE", "Grid"), "KNN": ("sin_balanceo", "Bayesiana")}
# Espacio de busqueda del modelo original
ESPACIO_ORIGINAL = {"n_rondas": ("int", 20, 300), "tasa": ("float", 0.1, 1.0), "lambda_": ("float", 1.0, 300.0),
                    "kappa": ("float", 0.1, 200.0)}


def indices(y, rep):
    """Pliegues externos de la repeticion rep. La repeticion 0 es la del Entregable 2."""
    if rep == 0:
        return B.indices_externos(y)
    return list(StratifiedKFold(B.N_EXTERNO, shuffle=True, random_state=SEED + 1000 * rep).split(np.zeros(len(y)), y))


def crear_original(p):
    return AditivoPenalizado(ordenes=ORDENES, random_state=SEED, **p)


def buscar_original(Xs, ys, semilla):
    cv = StratifiedKFold(B.N_INTERNO, shuffle=True, random_state=semilla)
    hist, mejor = [], [-1.0]
    def obj(trial):
        p = {"n_rondas": trial.suggest_int("n_rondas", 20, 300), "tasa": trial.suggest_float("tasa", 0.1, 1.0),
             "lambda_": trial.suggest_float("lambda_", 1.0, 300.0, log=True),
             "kappa": trial.suggest_float("kappa", 0.1, 200.0, log=True)}
        s = float(np.mean(cross_val_score(crear_original(p), Xs, ys, cv=cv, scoring="average_precision")))
        mejor[0] = max(mejor[0], s); hist.append({"eval": len(hist) + 1, "score": s, "mejor_hasta_ahora": mejor[0], "params": p})
        return s
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=4))
    est.optimize(obj, n_trials=B.PRESUPUESTO)
    return est.best_params, hist, mejor[0]


def buscar_ebm(Xs, ys, semilla, tipos):
    """Busqueda del EBM (mismo espacio que scripts/run_ebm.py)."""
    from interpret.glassbox import ExplainableBoostingClassifier
    cv = StratifiedKFold(B.N_INTERNO, shuffle=True, random_state=semilla); hist, mejor = [], [-1.0]
    def crear(p):
        return ExplainableBoostingClassifier(feature_types=tipos, random_state=SEED, n_jobs=1, outer_bags=4, max_rounds=2000, **p)
    def obj(trial):
        p = {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True), "max_leaves": trial.suggest_int("max_leaves", 2, 5),
             "interactions": trial.suggest_int("interactions", 0, 10), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True)}
        s = float(np.mean(cross_val_score(crear(p), Xs, ys, cv=cv, scoring="average_precision")))
        mejor[0] = max(mejor[0], s); hist.append(s); return s
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=4))
    est.optimize(obj, n_trials=B.PRESUPUESTO)
    return est.best_params, mejor[0], crear


def tarea(nombre, modelo, bal, opt, rep, fold, X, y, tr, va):
    f_json = OUT / f"{nombre}__rep{rep}__f{fold}.json"
    if f_json.exists():
        return
    semilla = SEED + 1000 * rep + fold
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    sub, _ = train_test_split(np.arange(len(tr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=semilla)
    t0 = time.time()
    if nombre == "EBM":
        Xc = X.copy()
        for c in Xc.columns:
            if c != "edad": Xc[c] = Xc[c].astype("category")
        tipos = ["continuous" if c == "edad" else "nominal" for c in PREDICTORAS]
        mejores, interno, crear = buscar_ebm(Xc.iloc[tr].iloc[sub], ytr[sub], semilla, tipos); t_busq = time.time() - t0
        t1 = time.time(); m = crear(mejores).fit(Xc.iloc[tr], ytr); t_fit = time.time() - t1
        t2 = time.time(); p = m.predict_proba(Xc.iloc[va])[:, 1]; t_pred = time.time() - t2
    elif nombre == "AditivoPenalizado":
        mejores, hist, interno = buscar_original(Xtr.iloc[sub], ytr[sub], semilla); t_busq = time.time() - t0
        t1 = time.time(); m = crear_original(mejores).fit(Xtr, ytr); t_fit = time.time() - t1
        t2 = time.time(); p = m.predict_proba(Xva)[:, 1]; t_pred = time.time() - t2
    else:
        ev = B.Evaluador(modelo, bal, Xtr.iloc[sub], ytr[sub], semilla); B.BUSCADORES[opt](ev, modelo, semilla)
        t_busq = time.time() - t0; mejores = B.mejores_params(ev); interno = ev.mejor
        t1 = time.time(); pipe = B.crear_pipeline(modelo, bal, mejores).fit(Xtr, ytr); t_fit = time.time() - t1
        t2 = time.time(); p = pipe.predict_proba(Xva)[:, 1]; t_pred = time.time() - t2
    res = {"modelo": nombre, "balanceo": bal, "optimizador": opt, "rep": rep, "fold": fold, **B.metricas(yva, p),
           "PR_AUC_interno": interno, "t_busqueda_s": t_busq, "t_refit_s": t_fit, "t_pred_s": t_pred,
           "mejores_params": mejores}
    np.save(OUT / f"{nombre}__rep{rep}__f{fold}.npy", np.stack([va, p.astype(np.float32)]).astype(np.float32))
    f_json.write_text(json.dumps(res, default=str))
    print(f"rep{rep} f{fold} {nombre:18s} PR-AUC={res['PR_AUC']:.4f} t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    X, y = B.cargar_xy()
    if sys.argv[1] == "--ebm":  # EBM en las repeticiones indicadas (la repeticion 0 ya esta en resultados/tareas)
        for rep in [int(a) for a in sys.argv[2:]]:
            for fold, (tr, va) in enumerate(indices(y, rep)):
                tarea("EBM", "EBM", "sin_balanceo", "Bayesiana", rep, fold, X, y, tr, va)
        sys.exit()
    for rep in [int(a) for a in sys.argv[1:]]:
        for fold, (tr, va) in enumerate(indices(y, rep)):
            tarea("AditivoPenalizado", "AditivoPenalizado", "sin_balanceo", "Bayesiana", rep, fold, X, y, tr, va)
            if rep == 0:
                continue  # los siete modelos base de la repeticion 0 ya estan en resultados/tareas (Entregable 2)
            for m, (bal, opt) in MEJORES.items():
                tarea(m, m, bal, opt, rep, fold, X, y, tr, va)
    print("FIN", sys.argv[1:])
