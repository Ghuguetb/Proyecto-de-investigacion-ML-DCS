"""
calculos.py
===========
Cálculos pesados del Entregable 2 (dashboard + JBook).

Todo lo que tarda más de un par de segundos (entrenar los modelos, el bootstrap, la curva de
aprendizaje, el PCA, etc.) se hace una sola vez aquí y se guarda en ``cache/artefactos.pkl``.
Así el dashboard abre rápido y el JBook usa exactamente los mismos números que el dashboard.

Secciones del archivo
---------------------
1. Utilidades estadísticas (Wilson, métricas, V de Cramér).
2. Modelos base: regresión logística (la principal) y SVM lineal (alternativa).
3. Evaluación: bootstrap, calibración, residuos por subgrupo, curva de aprendizaje.
4. Interpretación: odds ratios y dependencia parcial de la edad.
5. Insumos del EDA que no dependen de los filtros (información mutua, PCA, clusters, fuga).
6. ``obtener_artefactos``: construye o carga la caché.
"""
from __future__ import annotations

import pickle
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy import stats

from .preprocesamiento import construir_preprocesador
from .utils import (CATEGORIAS, OBJETIVO, ORDENES, PREDICTORAS, PREDICTORAS_CAT, RAIZ, SEED,
                    SOSPECHOSAS, a_ordinal, cargar_particiones, cramers_v)

warnings.filterwarnings("ignore")

RUTA_CACHE = RAIZ / "cache" / "artefactos.pkl"
VERSION_CACHE = 3  # súbala si cambia la estructura de los artefactos para forzar el recálculo

# Etiquetas legibles (se usan en gráficos y tablas)
ETIQUETAS = {
    "edad": "Edad", "sexo": "Sexo", "raza_etnia": "Raza / etnia", "educacion": "Escolaridad",
    "hipertension": "Hipertensión", "colesterol_alto": "Colesterol alto", "diabetes": "Diabetes",
    "acv": "Antecedente de ACV", "depresion_dx": "Diagnóstico de depresión",
    "frecuencia_depresion": "Frecuencia de síntomas depresivos",
    "dificultad_auditiva": "Dificultad auditiva", "dificultad_visual": "Dificultad visual",
    "tabaquismo": "Tabaquismo", "imc_categoria": "Categoría de IMC",
    "anio": "Año", "trimestre": "Trimestre",
    "frecuencia_dificultad_cog": "Frecuencia de dificultad cognitiva (seguimiento)",
    "discapacidad": "Indicador de discapacidad", "demencia_dx": "Diagnóstico de demencia",
    "imc": "IMC numérico", "peso_muestral": "Peso muestral", "id_hogar": "ID del hogar",
    "nivel_dificultad_cognitiva": "Nivel original del objetivo", "dificultad_cognitiva": "Dificultad cognitiva (objetivo)",
    "grupo_edad_amplio": "Grupo de edad", "grupo_edad_q": "Grupo de edad (5 años)", "grupo_edad_acv": "Grupo de edad",
    "peso_talla_suprimidos": "Peso o talla suprimidos",
}
# Bloque temático de cada predictora (para colorear gráficos)
BLOQUES = {
    "Salud mental": ["depresion_dx", "frecuencia_depresion"],
    "Sensorial": ["dificultad_auditiva", "dificultad_visual"],
    "Cardiometabólico": ["hipertension", "colesterol_alto", "diabetes", "imc_categoria"],
    "Neurovascular": ["acv"],
    "Estilo de vida": ["tabaquismo"],
    "Sociodemográfico": ["edad", "sexo", "raza_etnia", "educacion"],
}
BLOQUE_DE = {v: b for b, vs in BLOQUES.items() for v in vs}

GRUPOS_EDAD_CORTES = [17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 85]


# ---------------------------------------------------------------------------
# 1. UTILIDADES ESTADÍSTICAS
# ---------------------------------------------------------------------------
def wilson(k, n, z: float = 1.96):
    """Intervalo de Wilson para una proporción (vectorizado). Devuelve (inferior, superior)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    n_safe = np.where(n == 0, 1, n)
    p = k / n_safe
    den = 1 + z ** 2 / n_safe
    centro = (p + z ** 2 / (2 * n_safe)) / den
    mitad = z * np.sqrt(p * (1 - p) / n_safe + z ** 2 / (4 * n_safe ** 2)) / den
    lo, hi = centro - mitad, centro + mitad
    return np.where(n == 0, np.nan, lo), np.where(n == 0, np.nan, hi)


def prevalencia_por(df: pd.DataFrame, col, orden=None) -> pd.DataFrame:
    """Prevalencia de la variable objetivo por categoría de ``col`` con IC 95 % de Wilson."""
    g = df.groupby(col, observed=True)[OBJETIVO].agg(["sum", "count"])
    if orden is not None:
        g = g.reindex([c for c in orden if c in g.index])
    g = g.rename(columns={"sum": "casos", "count": "n"})
    g["prevalencia"] = g["casos"] / g["n"]
    g["ic_inf"], g["ic_sup"] = wilson(g["casos"], g["n"])
    return g.reset_index()


def metricas(y, p, yhat) -> dict:
    """Las mismas ocho métricas del Entregable 1."""
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 brier_score_loss, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    return {"PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p) if len(set(p)) > 1 else 0.5,
            "Recall": recall_score(y, yhat, zero_division=0),
            "Precisión": precision_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat, zero_division=0),
            "Balanced acc.": balanced_accuracy_score(y, yhat),
            "Brier": brier_score_loss(y, p),
            "Accuracy": accuracy_score(y, yhat)}


def umbral_f1(y, p):
    """Umbral que maximiza el F1 sobre un conjunto de probabilidades (se usa con las OOF)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, um = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(um[i]), float(f1[i]), float(prec[i]), float(rec[i])


def grupo_edad(serie, cortes=(17, 34, 49, 64, 74, 85)):
    return pd.cut(serie, list(cortes))


# ---------------------------------------------------------------------------
# 2. MODELOS BASE
# ---------------------------------------------------------------------------
def _pipeline_lr(C=1.0, class_weight=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("pre", construir_preprocesador()),
                     ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000,
                                               random_state=SEED))])


def _pipeline_svm(C=0.01):
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    return Pipeline([("pre", construir_preprocesador()),
                     ("svm", LinearSVC(C=C, max_iter=50000, random_state=SEED))])


def entrenar_logistica(X_tr, y_tr, cv):
    """Rejilla (C, class_weight) con PR-AUC como criterio, igual que el Entregable 1."""
    from sklearn.model_selection import GridSearchCV
    rejilla = {"lr__C": [0.001, 0.01, 0.1, 1, 10], "lr__class_weight": [None, "balanced"]}
    bus = GridSearchCV(_pipeline_lr(), rejilla, cv=cv, n_jobs=-1, refit="PR-AUC",
                       return_train_score=True,
                       scoring={"PR-AUC": "average_precision", "ROC-AUC": "roc_auc",
                                "Brier": "neg_brier_score"})
    bus.fit(X_tr, y_tr)
    r = pd.DataFrame(bus.cv_results_)
    tabla = pd.DataFrame({
        "C": r["param_lr__C"].astype(float),
        "class_weight": r["param_lr__class_weight"].map(lambda v: "balanced" if v == "balanced" else "ninguno"),
        "PR-AUC (val.)": r["mean_test_PR-AUC"], "de PR-AUC": r["std_test_PR-AUC"],
        "PR-AUC (entr.)": r["mean_train_PR-AUC"], "ROC-AUC (val.)": r["mean_test_ROC-AUC"],
        "Brier (val.)": -r["mean_test_Brier"]}).sort_values("PR-AUC (val.)", ascending=False)
    return bus.best_estimator_, bus.best_params_, tabla.reset_index(drop=True)


def entrenar_svm(X_tr, y_tr, cv):
    """SVM lineal (LinearSVC). Se elige C por PR-AUC y luego se calibra con Platt (sigmoide)
    para poder leer sus salidas como probabilidades y compararlas con la logística."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    rejilla = {"svm__C": [0.001, 0.01, 0.1, 1]}
    bus = GridSearchCV(_pipeline_svm(), rejilla, cv=cv, n_jobs=-1, refit="PR-AUC",
                       return_train_score=True,
                       scoring={"PR-AUC": "average_precision", "ROC-AUC": "roc_auc"})
    bus.fit(X_tr, y_tr)
    r = pd.DataFrame(bus.cv_results_)
    tabla = pd.DataFrame({"C": r["param_svm__C"].astype(float),
                          "PR-AUC (val.)": r["mean_test_PR-AUC"], "de PR-AUC": r["std_test_PR-AUC"],
                          "PR-AUC (entr.)": r["mean_train_PR-AUC"],
                          "ROC-AUC (val.)": r["mean_test_ROC-AUC"]}
                         ).sort_values("PR-AUC (val.)", ascending=False).reset_index(drop=True)
    C = float(bus.best_params_["svm__C"])
    cal = CalibratedClassifierCV(Pipeline([("pre", construir_preprocesador()),
                                           ("svm", LinearSVC(C=C, max_iter=50000, random_state=SEED))]),
                                 method="sigmoid", cv=5)
    cal.fit(X_tr, y_tr)
    return cal, C, tabla


# ---------------------------------------------------------------------------
# 3. EVALUACIÓN
# ---------------------------------------------------------------------------
def bootstrap_metricas(y, probs: dict, preds: dict, B: int = 1000):
    """IC 95 % por percentiles con B remuestreos del conjunto de prueba (misma semilla que el E1)."""
    rng = np.random.RandomState(SEED)
    idx = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    puntual, boots = {}, {}
    for nombre in probs:
        p, yh = probs[nombre], preds[nombre]
        puntual[nombre] = metricas(y, p, yh)
        boots[nombre] = pd.DataFrame([metricas(y[i], p[i], yh[i]) for i in idx])
    return puntual, boots


def subgrupos(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Observado vs. predicho, residuo medio y AUC dentro de cada subgrupo (diagnóstico de residuos)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = test.assign(p=p, residuo=test[OBJETIVO].values - p,
                    grupo_edad=pd.cut(test["edad"], [17, 34, 49, 64, 74, 85],
                                      labels=["18-34", "35-49", "50-64", "65-74", "75-85"]).astype(str),
                    anio=test["anio"].astype(str), trimestre=test["trimestre"].astype(str))
    filas = []
    for var in ["anio", "trimestre", "sexo", "grupo_edad", "raza_etnia", "educacion", "acv",
                "depresion_dx", "frecuencia_depresion", "dificultad_auditiva", "hipertension"]:
        for nivel, g in d.groupby(var, observed=True):
            if len(g) < 30:
                continue
            ic = 1.96 * g["residuo"].std() / np.sqrt(len(g))
            ok = g[OBJETIVO].nunique() > 1
            filas.append([var, str(nivel), len(g), g[OBJETIVO].mean(), g["p"].mean(),
                          g["residuo"].mean(), ic,
                          roc_auc_score(g[OBJETIVO], g["p"]) if ok else np.nan,
                          average_precision_score(g[OBJETIVO], g["p"]) if ok else np.nan])
    return pd.DataFrame(filas, columns=["variable", "nivel", "n", "observado", "predicho",
                                        "residuo", "ic95", "roc_auc", "pr_auc"])


def curva_aprendizaje(modelo, X, y, cv):
    from sklearn.model_selection import learning_curve
    tam, sc_tr, sc_val = learning_curve(modelo, X, y, cv=cv, scoring="average_precision",
                                        train_sizes=np.linspace(0.05, 1.0, 8), n_jobs=-1,
                                        shuffle=True, random_state=SEED)
    return pd.DataFrame({"n": tam, "train": sc_tr.mean(1), "train_sd": sc_tr.std(1),
                         "val": sc_val.mean(1), "val_sd": sc_val.std(1)})


# ---------------------------------------------------------------------------
# 4. INTERPRETACIÓN
# ---------------------------------------------------------------------------
def tabla_or(modelo, X_tr, y_tr) -> pd.DataFrame:
    """Odds ratios con IC 95 % reajustando el mismo modelo sin penalización con statsmodels."""
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    pre = modelo.named_steps["pre"]
    nombres = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    Xm = pd.DataFrame(pre.transform(X_tr), columns=nombres)
    logit = sm.Logit(y_tr, sm.add_constant(Xm)).fit(disp=0)
    ic = np.exp(logit.conf_int())
    t = pd.DataFrame({"termino": nombres, "OR": np.exp(logit.params[nombres]).values,
                      "ic_inf": ic.loc[nombres, 0].values, "ic_sup": ic.loc[nombres, 1].values,
                      "p": logit.pvalues[nombres].values})
    t = t[~t["termino"].str.startswith(("edad_sp", "missingindicator"))].copy()
    t["p_holm"] = multipletests(t["p"], method="holm")[1]
    variables = sorted(PREDICTORAS_CAT, key=len, reverse=True)
    def split(term):
        for v in variables:
            if term.startswith(v + "_"):
                return v, term[len(v) + 1:]
        return term, ""
    t["variable"], t["categoria"] = zip(*t["termino"].map(split))
    t["bloque"] = t["variable"].map(BLOQUE_DE)
    t["etiqueta"] = t["variable"].map(ETIQUETAS) + ": " + t["categoria"]
    corr = np.corrcoef(modelo.named_steps["lr"].coef_[0], logit.params[nombres])[0, 1]
    return t.reset_index(drop=True), float(corr)


def dependencia_parcial_edad(modelo, X_tr):
    muestra = X_tr.sample(4000, random_state=SEED)
    edades = np.arange(18, 86)
    pd_ = [modelo.predict_proba(muestra.assign(edad=float(e)))[:, 1].mean() for e in edades]
    return pd.DataFrame({"edad": edades, "prob": pd_})


def categorias_por_variable():
    """Categorías de cada predictora tal como las ve el modelo (con las raras ya agrupadas)."""
    from .preprocesamiento import categorias_finales
    return categorias_finales()


# ---------------------------------------------------------------------------
# 5. INSUMOS DEL EDA (no dependen de los filtros)
# ---------------------------------------------------------------------------
def auc_univariado(train: pd.DataFrame, cv) -> pd.DataFrame:
    """ROC-AUC de cada variable por separado (auditoría de fuga de datos)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
    y = train[OBJETIVO]
    filas = []
    for col in PREDICTORAS + SOSPECHOSAS + ["anio", "trimestre"]:
        X = train[[col]].copy()
        if col == "edad":
            pre = Pipeline([("i", SimpleImputer(strategy="median")),
                            ("s", SplineTransformer(n_knots=4)), ("e", StandardScaler())])
        else:
            X[col] = X[col].map(str).replace({"nan": "Faltante"})
            pre = OneHotEncoder(handle_unknown="ignore")
        m = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])
        s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        grupo = ("Sospechosa de fuga" if col in SOSPECHOSAS
                 else "Metadato" if col in ("anio", "trimestre") else "Predictora")
        filas.append([col, s.mean(), s.std(), grupo])
    return pd.DataFrame(filas, columns=["variable", "auc", "de", "grupo"]).sort_values("auc")


def informacion_mutua(train: pd.DataFrame) -> pd.Series:
    from sklearn.feature_selection import mutual_info_classif
    X = pd.DataFrame(index=train.index)
    for v in PREDICTORAS_CAT:
        X[v] = train[v].astype("category").cat.codes
    X["edad"] = train["edad"].fillna(train["edad"].median())
    mi = mutual_info_classif(X, train[OBJETIVO], discrete_features=[c != "edad" for c in X.columns],
                             random_state=SEED)
    return pd.Series(mi, index=X.columns).sort_values()


def matriz_cramer(train: pd.DataFrame) -> pd.DataFrame:
    tmp = train[PREDICTORAS_CAT].copy()
    tmp["edad"] = grupo_edad(train["edad"], (17, 34, 49, 64, 74, 85)).astype(str)
    cols = PREDICTORAS_CAT + ["edad"]
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            m.loc[a, b] = m.loc[b, a] = cramers_v(pd.crosstab(tmp[a], tmp[b]))
    return m


def pca_y_clusters(train: pd.DataFrame):
    """PCA exploratorio (como en el E1) y K-means sobre las componentes que explican el 80 %."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    X = train[PREDICTORAS_CAT].fillna(train[PREDICTORAS_CAT].mode().iloc[0])
    cats = categorias_por_variable()
    for v in PREDICTORAS_CAT:
        X[v] = pd.Categorical(X[v], categories=cats[v])
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X["edad"] = train["edad"].fillna(train["edad"].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED).fit(Xs)
    acum = np.cumsum(pca.explained_variance_ratio_)
    k80 = int(np.argmax(acum >= 0.80) + 1)
    Z = pca.transform(Xs)
    cargas = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["CP1", "CP2", "CP3"])

    Zk = Z[:, :k80]
    idx_s = np.random.RandomState(SEED).choice(len(Zk), 6000, replace=False)
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Zk)
        sil[k] = float(silhouette_score(Zk[idx_s], km.labels_[idx_s]))
    k_opt = max(sil, key=sil.get)
    km = KMeans(n_clusters=k_opt, n_init=10, random_state=SEED).fit(Zk)
    etiq = pd.Series(km.labels_, index=train.index, name="cluster")
    perfil = train.assign(cluster=etiq).groupby("cluster").agg(
        n=(OBJETIVO, "size"), prevalencia=(OBJETIVO, "mean"), edad_media=("edad", "mean"),
        hipertension=("hipertension", lambda s: (s == "Sí").mean()),
        depresion_dx=("depresion_dx", lambda s: (s == "Sí").mean()),
        mujeres=("sexo", lambda s: (s == "Mujer").mean()),
        fumador_actual=("tabaquismo", lambda s: (s == "Fumador actual").mean()),
        acv=("acv", lambda s: (s == "Sí").mean())).reset_index()

    muestra = np.random.RandomState(SEED).choice(len(Z), 6000, replace=False)
    pts = train.iloc[muestra][["edad", OBJETIVO, "sexo", "acv", "depresion_dx", "anio",
                               "frecuencia_depresion", "hipertension", "raza_etnia"]].copy()
    pts["CP1"], pts["CP2"] = Z[muestra, 0], Z[muestra, 1]
    pts["cluster"] = etiq.iloc[muestra].astype(str).values
    return {"varianza": pca.explained_variance_ratio_, "acum": acum, "k80": k80, "cargas": cargas,
            "puntos": pts.reset_index(drop=True), "silueta": sil, "k_opt": k_opt, "perfil_clusters": perfil}


def sesgo_muestral(train: pd.DataFrame) -> pd.DataFrame:
    """Muestra sin ponderar frente a la población ponderada por el peso muestral de la NHIS."""
    w = train["peso_muestral"]
    ind = {"Dificultad cognitiva": train[OBJETIVO] == 1, "65 años o más": train["edad"] >= 65,
           "Mujer": train["sexo"] == "Mujer", "Blanco no hispano": train["raza_etnia"] == "Blanco no hispano",
           "Hispano": train["raza_etnia"] == "Hispano",
           "Pregrado o posgrado": train["educacion"].isin(["Pregrado", "Posgrado"]),
           "Antecedente de ACV": train["acv"] == "Sí"}
    t = pd.DataFrame({"Muestra sin ponderar": {k: float(np.average(v)) for k, v in ind.items()},
                      "Población ponderada (EE. UU.)": {k: float(np.average(v, weights=w)) for k, v in ind.items()}})
    t["diferencia_pp"] = 100 * (t.iloc[:, 0] - t.iloc[:, 1])
    return t.reset_index().rename(columns={"index": "indicador"})


# ---------------------------------------------------------------------------
# 6. CONSTRUCCIÓN / CARGA DE LA CACHÉ
# ---------------------------------------------------------------------------
def construir_artefactos(verbose: bool = True) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                                 recall_score, precision_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.calibration import calibration_curve

    def log(msg):
        if verbose:
            print(f"[calculos] {msg}", flush=True)

    t0 = time.time()
    train, test = cargar_particiones()
    X_tr, y_tr = train[PREDICTORAS], train[OBJETIVO].values
    X_te, y_te = test[PREDICTORAS], test[OBJETIVO].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    A: dict = {"version": VERSION_CACHE, "sklearn": sklearn.__version__}

    log("Regresión logística: rejilla de hiperparámetros…")
    lr, mejores, grid_lr = entrenar_logistica(X_tr, y_tr, cv)
    A["lr"], A["lr_params"], A["grid_lr"] = lr, mejores, grid_lr
    p_oof_lr = cross_val_predict(lr, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    p_lr = lr.predict_proba(X_te)[:, 1]

    log("SVM lineal: rejilla + calibración…")
    svm, C_svm, grid_svm = entrenar_svm(X_tr, y_tr, cv)
    A["svm"], A["svm_C"], A["grid_svm"] = svm, C_svm, grid_svm
    p_oof_svm = cross_val_predict(svm, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    p_svm = svm.predict_proba(X_te)[:, 1]

    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    dummy_s = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_tr, y_tr)

    A["y_tr"], A["y_te"] = y_tr, y_te
    A["p_oof"] = {"Regresión logística": p_oof_lr, "SVM lineal": p_oof_svm}
    A["p_te"] = {"Regresión logística": p_lr, "SVM lineal": p_svm}
    A["umbral"] = {}
    for nombre in A["p_oof"]:
        um, f1, pr, rc = umbral_f1(y_tr, A["p_oof"][nombre])
        A["umbral"][nombre] = {"umbral": um, "f1": f1, "precision": pr, "recall": rc}
    A["prevalencia_tr"] = float(y_tr.mean())

    log("Bootstrap de métricas (1 000 remuestreos × 4 modelos)…")
    probs = {"Dummy (prior)": dummy.predict_proba(X_te)[:, 1],
             "Dummy (stratified)": dummy_s.predict_proba(X_te)[:, 1],
             "Regresión logística": p_lr, "SVM lineal": p_svm}
    preds = {"Dummy (prior)": dummy.predict(X_te), "Dummy (stratified)": dummy_s.predict(X_te),
             "Regresión logística": (p_lr >= A["umbral"]["Regresión logística"]["umbral"]).astype(int),
             "SVM lineal": (p_svm >= A["umbral"]["SVM lineal"]["umbral"]).astype(int)}
    puntual, boots = bootstrap_metricas(y_te, probs, preds)
    A["metricas_test"], A["boot"] = puntual, boots
    A["probs_dummy"] = {k: v for k, v in probs.items() if k.startswith("Dummy")}
    A["pred_dummy"] = {k: v for k, v in preds.items() if k.startswith("Dummy")}

    # Validación cruzada del modelo trivial (para la tabla comparativa)
    from sklearn.model_selection import cross_validate
    pun = {"PR-AUC": "average_precision", "ROC-AUC": "roc_auc", "F1": "f1", "Recall": "recall",
           "Balanced acc.": "balanced_accuracy", "Brier": "neg_brier_score"}
    r = cross_validate(DummyClassifier(strategy="prior"), X_tr, y_tr, cv=cv, scoring=pun)
    A["cv_dummy"] = {k: float(np.mean(r["test_" + k])) for k in pun}

    log("Calibración y residuos por subgrupo…")
    A["calibracion"] = {}
    for nombre, p in A["p_te"].items():
        fp, pm = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        A["calibracion"][nombre] = pd.DataFrame({"predicha": pm, "observada": fp})
        A.setdefault("brier_ref", brier_score_loss(y_te, np.full_like(p, y_tr.mean())))
    A["subgrupos"] = {n: subgrupos(test, p) for n, p in A["p_te"].items()}

    log("Curva de aprendizaje…")
    A["curva_aprendizaje"] = curva_aprendizaje(lr, X_tr, y_tr, cv)

    log("Odds ratios y dependencia parcial de la edad…")
    A["or"], A["or_corr"] = tabla_or(lr, X_tr, y_tr)
    A["dep_edad"] = dependencia_parcial_edad(lr, X_tr)
    A["prev_edad_obs"] = prevalencia_por(train.assign(g=pd.cut(train["edad"], range(15, 90, 5))), "g")
    A["prev_edad_obs"]["edad"] = [i.mid for i in A["prev_edad_obs"]["g"]]
    A["prev_edad_obs"] = A["prev_edad_obs"].drop(columns="g")

    log("Insumos del EDA (fuga, información mutua, Cramér, PCA, clusters)…")
    A["auc_uni"] = auc_univariado(train, cv)
    A["mi"] = informacion_mutua(train)
    A["cramer"] = matriz_cramer(train)
    A["pca"] = pca_y_clusters(train)
    A["sesgo"] = sesgo_muestral(train)
    A["categorias"] = categorias_por_variable()
    log(f"Listo en {time.time() - t0:.0f} s.")
    return A


def obtener_artefactos(forzar: bool = False) -> dict:
    """Carga la caché; si no existe, es de otra versión de scikit-learn o está dañada, la reconstruye."""
    if RUTA_CACHE.exists() and not forzar:
        try:
            with open(RUTA_CACHE, "rb") as f:
                A = pickle.load(f)
            if A.get("version") == VERSION_CACHE and A.get("sklearn") == sklearn.__version__:
                return A
            print("[calculos] La caché es de otra versión; se recalcula.", flush=True)
        except Exception as e:  # caché corrupta o incompatible
            print(f"[calculos] No se pudo leer la caché ({e}); se recalcula.", flush=True)
    A = construir_artefactos()
    RUTA_CACHE.parent.mkdir(exist_ok=True)
    with open(RUTA_CACHE, "wb") as f:
        pickle.dump(A, f, protocol=pickle.HIGHEST_PROTOCOL)
    return A


if __name__ == "__main__":
    obtener_artefactos(forzar=True)
