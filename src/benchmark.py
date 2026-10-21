"""
Entregable 3. Benchmark 7 modelos x 4 balanceos x 4 optimizadores con validacion cruzada anidada.

Reglas de no-fuga:
- Todo el preprocesamiento (imputacion, splines, escalado, one-hot) y el balanceo (SMOTE/ADASYN)
  viven DENTRO del Pipeline, asi que se ajustan solo con los datos de entrenamiento de cada fold.
- Lazo interno (busqueda de hiperparametros) usa solo el entrenamiento del fold externo.
- Las metricas se calculan unicamente en el fold externo de validacion.
- El conjunto de prueba reservado (test.csv) NO se toca aqui.
"""
import json
import math
import random
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .preprocesamiento import construir_preprocesador
from .utils import OBJETIVO, PREDICTORAS, SEED, cargar_particiones

warnings.filterwarnings("ignore")

MODELOS = ["KNN", "NaiveBayes", "LogReg", "DecisionTree", "RandomForest", "XGBoost", "SVM"]
BALANCEOS = ["sin_balanceo", "SMOTE", "ADASYN", "class_weight"]
OPTIMIZADORES = ["Grid", "Random", "Bayesiana", "Genetica"]

# Configuracion de costo computacional (se documenta en el informe)
N_EXTERNO = 5          # folds externos
N_INTERNO = 3          # folds internos
N_SUBMUESTRA = 8000    # filas usadas en la busqueda interna (multi-fidelidad simple)
PRESUPUESTO = 12       # evaluaciones por busqueda (Random, Bayesiana, Genetica)

# ---------------------------------------------------------------------------
# Espacios de busqueda. Tipos: ("int"|"float", lo, hi, log) o ("cat", [opciones])
# Grid usa una rejilla gruesa (~12 puntos); los demas buscan en el espacio completo.
# ---------------------------------------------------------------------------
ESPACIOS = {
    "KNN": {"n_neighbors": ("int", 5, 100, True), "weights": ("cat", ["uniform", "distance"]),
            "p": ("cat", [1, 2])},
    "NaiveBayes": {"var_smoothing": ("float", 1e-11, 1e-2, True)},
    "LogReg": {"penalty": ("cat", ["l1", "l2"]), "C": ("float", 1e-3, 1e2, True)},
    "DecisionTree": {"max_depth": ("int", 2, 20, False), "min_samples_leaf": ("int", 5, 300, True),
                     "criterion": ("cat", ["gini", "entropy"])},
    "RandomForest": {"n_estimators": ("int", 50, 150, False), "max_depth": ("int", 4, 20, False),
                     "min_samples_leaf": ("int", 5, 100, True), "max_features": ("cat", ["sqrt", "log2"])},
    "XGBoost": {"n_estimators": ("int", 50, 300, False), "max_depth": ("int", 2, 8, False),
                "learning_rate": ("float", 0.01, 0.3, True), "subsample": ("float", 0.6, 1.0, False),
                "colsample_bytree": ("float", 0.5, 1.0, False)},
    "SVM": {"C": ("float", 1e-3, 10.0, True)},
}
REJILLAS = {
    "KNN": {"n_neighbors": [15, 30, 60], "weights": ["uniform", "distance"], "p": [1, 2]},
    "NaiveBayes": {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2]},
    "LogReg": {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "DecisionTree": {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [20, 50, 100]},
    "RandomForest": {"n_estimators": [100], "max_depth": [6, 10, 16], "min_samples_leaf": [10, 30],
                     "max_features": ["sqrt", "log2"]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
    "SVM": {"C": [0.001, 0.01, 0.1, 1, 10]},
}


class PesosBalanceados(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador sin class_weight (XGBoost, Naive Bayes) para pasarle pesos
    'balanced' por muestra, calculados con las etiquetas del fold de entrenamiento."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        sw = compute_sample_weight("balanced", y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=sw)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def balanceo_aplicable(modelo, balanceo):
    """KNN no admite ni class_weight ni sample_weight: class_weight no tiene efecto."""
    return not (modelo == "KNN" and balanceo == "class_weight")


def crear_clasificador(modelo, p, cw):
    sem = SEED
    if modelo == "KNN":
        return KNeighborsClassifier(n_jobs=1, **p)
    if modelo == "NaiveBayes":
        nb = GaussianNB(**p)
        return PesosBalanceados(nb) if cw else nb
    if modelo == "LogReg":
        return LogisticRegression(solver="liblinear", max_iter=500, random_state=sem,
                                  class_weight="balanced" if cw else None, **p)
    if modelo == "DecisionTree":
        return DecisionTreeClassifier(random_state=sem, class_weight="balanced" if cw else None, **p)
    if modelo == "RandomForest":
        return RandomForestClassifier(random_state=sem, n_jobs=1, class_weight="balanced" if cw else None, **p)
    if modelo == "XGBoost":
        xgb = XGBClassifier(tree_method="hist", n_jobs=1, eval_metric="logloss", random_state=sem, verbosity=0, **p)
        return PesosBalanceados(xgb) if cw else xgb
    if modelo == "SVM":
        base = LinearSVC(dual=False, max_iter=2000, random_state=sem, class_weight="balanced" if cw else None, **p)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    raise ValueError(modelo)


def crear_pipeline(modelo, balanceo, params):
    pasos = [("pre", construir_preprocesador())]
    if balanceo == "SMOTE":
        pasos.append(("bal", SMOTE(random_state=SEED)))
    elif balanceo == "ADASYN":
        pasos.append(("bal", ADASYN(random_state=SEED)))
    pasos.append(("clf", crear_clasificador(modelo, params, balanceo == "class_weight" and balanceo_aplicable(modelo, balanceo))))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------------
# Decodificacion de genes [0,1] -> hiperparametros (usada por Random y Genetica)
# ---------------------------------------------------------------------------
def decodificar(genes, espacio):
    p = {}
    for g, (nombre, spec) in zip(genes, espacio.items()):
        g = min(max(float(g), 0.0), 1.0)
        if spec[0] == "cat":
            op = spec[1]
            p[nombre] = op[min(int(g * len(op)), len(op) - 1)]
        else:
            _, lo, hi, log = spec
            v = math.exp(math.log(lo) + g * (math.log(hi) - math.log(lo))) if log else lo + g * (hi - lo)
            p[nombre] = int(round(v)) if spec[0] == "int" else float(v)
    return p


def _clave(p):
    return json.dumps(p, sort_keys=True, default=str)


class Evaluador:
    """Evalua una configuracion con CV interna (PR-AUC) y lleva el historial de convergencia."""
    def __init__(self, modelo, balanceo, X, y, semilla):
        self.modelo, self.balanceo, self.X, self.y = modelo, balanceo, X, y
        self.cv = StratifiedKFold(N_INTERNO, shuffle=True, random_state=semilla)
        self.cache, self.historia, self.mejor = {}, [], -np.inf
        self.t0 = time.time()

    def __call__(self, params):
        k = _clave(params)
        if k in self.cache:
            return self.cache[k]
        try:
            pipe = crear_pipeline(self.modelo, self.balanceo, params)
            s = float(np.mean(cross_val_score(pipe, self.X, self.y, cv=self.cv, scoring="average_precision")))
        except Exception:
            s = 0.0
        self.cache[k] = s
        self.mejor = max(self.mejor, s)
        self.historia.append({"eval": len(self.historia) + 1, "score": s, "mejor_hasta_ahora": self.mejor,
                              "t": time.time() - self.t0, "params": params})
        return s


def mejores_params(ev):
    return max(ev.historia, key=lambda h: h["score"])["params"]


# ---------------------------------------------------------------------------
# Los cuatro optimizadores
# ---------------------------------------------------------------------------
def buscar_grid(ev, modelo, semilla):
    for p in ParameterGrid(REJILLAS[modelo]):
        ev(p)


def buscar_random(ev, modelo, semilla):
    rng = np.random.RandomState(semilla)
    esp = ESPACIOS[modelo]
    for _ in range(PRESUPUESTO):
        ev(decodificar(rng.rand(len(esp)), esp))


def buscar_bayesiana(ev, modelo, semilla):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    esp = ESPACIOS[modelo]

    def objetivo(trial):
        p = {}
        for nombre, spec in esp.items():
            if spec[0] == "cat":
                p[nombre] = trial.suggest_categorical(nombre, spec[1])
            elif spec[0] == "int":
                p[nombre] = trial.suggest_int(nombre, spec[1], spec[2], log=spec[3])
            else:
                p[nombre] = trial.suggest_float(nombre, spec[1], spec[2], log=spec[3])
        return ev(p)

    sampler = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=4)  # TPE: surrogate + EI
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    estudio.optimize(objetivo, n_trials=PRESUPUESTO)


def buscar_genetica(ev, modelo, semilla, tam_pob=4, n_gen=3, cxpb=0.7, mutpb=0.4):
    """Algoritmo genetico con DEAP: genes reales en [0,1], seleccion por torneo, cruce uniforme,
    mutacion gaussiana y elitismo (el mejor individuo pasa intacto a la siguiente generacion)."""
    from deap import base, creator, tools
    if not hasattr(creator, "FitnessMaxHP"):
        creator.create("FitnessMaxHP", base.Fitness, weights=(1.0,))
        creator.create("IndividuoHP", list, fitness=creator.FitnessMaxHP)
    esp = ESPACIOS[modelo]
    n = len(esp)
    rng = random.Random(semilla)
    tb = base.Toolbox()
    tb.register("attr", rng.random)
    tb.register("individuo", tools.initRepeat, creator.IndividuoHP, tb.attr, n=n)
    tb.register("select", tools.selTournament, tournsize=2)
    tb.register("mate", tools.cxUniform, indpb=0.5)

    def evaluar(ind):
        ind.fitness.values = (ev(decodificar(ind, esp)),)

    pob = [tb.individuo() for _ in range(tam_pob)]
    for ind in pob:
        evaluar(ind)
    ev.diversidad = []
    for _ in range(n_gen):
        elite = tools.selBest(pob, 1)[0]
        hijos = [creator.IndividuoHP(x) for x in map(list, tb.select(pob, tam_pob - 1))]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if rng.random() < cxpb:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for ind in hijos:
            if rng.random() < mutpb:
                for i in range(n):
                    if rng.random() < 0.5:
                        ind[i] = min(max(ind[i] + rng.gauss(0, 0.25), 0.0), 1.0)
                del ind.fitness.values
        for ind in hijos:
            if not ind.fitness.valid:
                evaluar(ind)
        pob = [creator.IndividuoHP(elite)] + hijos
        pob[0].fitness.values = elite.fitness.values
        d = np.mean([np.linalg.norm(np.array(a) - np.array(b)) for i, a in enumerate(pob) for b in pob[i + 1:]])
        ev.diversidad.append(float(d))


BUSCADORES = {"Grid": buscar_grid, "Random": buscar_random, "Bayesiana": buscar_bayesiana, "Genetica": buscar_genetica}


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -1e-9, 1 + 1e-9
    idx = np.digitize(p, q[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metricas(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p),
            "ECE": ece(y, p), "F1_05": f1_score(y, yhat, zero_division=0),
            "Recall_05": recall_score(y, yhat, zero_division=0)}


# ---------------------------------------------------------------------------
# Una tarea = (modelo, balanceo, optimizador, fold externo)
# ---------------------------------------------------------------------------
def indices_externos(y):
    return list(StratifiedKFold(N_EXTERNO, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def ejecutar_tarea(modelo, balanceo, opt, fold, X, y, carpeta):
    """Corre un fold externo completo y guarda JSON + predicciones fuera de fold."""
    from pathlib import Path
    carpeta = Path(carpeta)
    nombre = f"{modelo}__{balanceo}__{opt}__f{fold}"
    f_json = carpeta / f"{nombre}.json"
    if f_json.exists():
        return nombre
    tr, va = indices_externos(y)[fold]
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    semilla = SEED + fold
    n_sub = min(N_SUBMUESTRA, len(tr))
    sub, _ = train_test_split(np.arange(len(tr)), train_size=n_sub, stratify=ytr, random_state=semilla)
    ev = Evaluador(modelo, balanceo, Xtr.iloc[sub], ytr[sub], semilla)
    t0 = time.time()
    BUSCADORES[opt](ev, modelo, semilla)
    t_busq = time.time() - t0
    mejores = mejores_params(ev)
    t1 = time.time()
    pipe = crear_pipeline(modelo, balanceo, mejores).fit(Xtr, ytr)
    t_fit = time.time() - t1
    t2 = time.time()
    p = pipe.predict_proba(Xva)[:, 1]
    t_pred = time.time() - t2
    res = {"modelo": modelo, "balanceo": balanceo, "optimizador": opt, "fold": fold,
           **metricas(yva, p), "PR_AUC_interno": ev.mejor, "n_evals": len(ev.historia),
           "t_busqueda_s": t_busq, "t_refit_s": t_fit, "t_pred_s": t_pred,
           "balanceo_aplicable": balanceo_aplicable(modelo, balanceo),
           "mejores_params": mejores, "historia": ev.historia, "diversidad_ga": getattr(ev, "diversidad", None)}
    np.save(carpeta / f"{nombre}.npy", np.stack([va, p.astype(np.float32)]).astype(np.float32))
    f_json.write_text(json.dumps(res, default=str))
    return nombre


def cargar_xy():
    train, _ = cargar_particiones()
    return train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values
