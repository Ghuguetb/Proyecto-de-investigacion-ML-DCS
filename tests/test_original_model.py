"""Pruebas unitarias del modelo original (AditivoPenalizado). Se ejecutan con: pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.original_model import AditivoPenalizado  # noqa: E402


def datos_sinteticos(n=4000, semilla=0):
    """Datos donde la solucion es conocida: logit = -1 + 0.04 (edad - 50) + 1.2 [fumador] - 0.8 [nivel B]."""
    rng = np.random.default_rng(semilla)
    edad = rng.uniform(18, 85, n)
    fuma = rng.choice(["No", "Si"], n)
    nivel = rng.choice(["A", "B", "C"], n)
    z = -1 + 0.04 * (edad - 50) + 1.2 * (fuma == "Si") - 0.8 * (nivel == "B")
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"edad": edad, "fuma": fuma, "nivel": nivel}), y, z


def test_aprende_solucion_conocida():
    X, y, z = datos_sinteticos()
    m = AditivoPenalizado(n_rondas=150, tasa=0.5, lambda_=1.0).fit(X, y)
    logit = m.decision_function(X)
    assert roc_auc_score(y, m.predict_proba(X)[:, 1]) > 0.75
    assert np.corrcoef(logit, z)[0, 1] > 0.97                      # recupera la forma verdadera
    f = m.funciones_["fuma"]                                        # niveles ordenados alfabeticamente: No, Si
    assert 1.0 < f[1] - f[0] < 1.4                                  # efecto verdadero de 1.2 en logit
    assert m.importancias().index[0] in {"edad", "fuma"}


def test_la_perdida_no_aumenta_con_ninguna_tasa_ni_penalizacion():
    X, y, _ = datos_sinteticos(1500)
    for tasa in (0.2, 0.7, 1.0):
        for kappa in (0.0, 5.0):
            m = AditivoPenalizado(n_rondas=40, tasa=tasa, lambda_=3.0, kappa=kappa, ordenes={"nivel": ["A", "B", "C"]}).fit(X, y)
            assert np.all(np.diff(m.perdida_) <= 1e-8), (tasa, kappa)


def test_reproducible_con_la_misma_semilla():
    X, y, _ = datos_sinteticos(1500)
    a = AditivoPenalizado(n_rondas=30, random_state=7).fit(X, y).predict_proba(X)
    b = AditivoPenalizado(n_rondas=30, random_state=7).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_entradas_invalidas():
    X, y, _ = datos_sinteticos(300)
    with pytest.raises(ValueError):
        AditivoPenalizado(tasa=0).fit(X, y)
    with pytest.raises(ValueError):
        AditivoPenalizado(n_rondas=0).fit(X, y)
    with pytest.raises(ValueError):
        AditivoPenalizado(lambda_=-1).fit(X, y)
    with pytest.raises(ValueError):
        AditivoPenalizado().fit(X, np.arange(len(y)) % 3)           # tres clases
    with pytest.raises(ValueError):
        AditivoPenalizado().fit(X, y[:-1])                          # largos distintos
    yn = y.astype(float); yn[0] = np.nan
    with pytest.raises(ValueError):
        AditivoPenalizado().fit(X, yn)
    m = AditivoPenalizado(n_rondas=5).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X.drop(columns="edad"))                     # falta una columna
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        AditivoPenalizado().predict(X)


def test_compatible_con_clone_y_parametros():
    m = AditivoPenalizado(n_rondas=12, tasa=0.3, lambda_=2.0, kappa=1.0, ordenes={"nivel": ["A", "B", "C"]})
    c = clone(m)
    assert c.get_params() == m.get_params()
    assert not hasattr(c, "funciones_")                             # el clon no esta ajustado


def test_faltantes_y_niveles_nuevos_no_rompen():
    X, y, _ = datos_sinteticos(800)
    X.loc[::10, "fuma"] = np.nan; X.loc[::13, "edad"] = np.nan
    m = AditivoPenalizado(n_rondas=20).fit(X, y)
    nuevo = pd.DataFrame({"edad": [np.nan, 40.0], "fuma": ["Tal vez", np.nan], "nivel": ["Z", "A"]})
    p = m.predict_proba(nuevo)
    assert p.shape == (2, 2) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)


def test_contribuciones_suman_el_logit():
    X, y, _ = datos_sinteticos(1000)
    m = AditivoPenalizado(n_rondas=30).fit(X, y)
    suma = m.contribuciones(X).sum(axis=1).to_numpy() + m.intercepto_
    assert np.allclose(suma, m.decision_function(X))
    assert np.allclose(m.contribuciones(X).mean(), 0, atol=0.15)     # curvas centradas en el entrenamiento


def test_suavizado_reduce_la_rugosidad_de_la_curva():
    X, y, _ = datos_sinteticos(1500)
    rug = lambda m: np.abs(np.diff(m.funciones_["edad"][:-1])).sum()
    sin = AditivoPenalizado(n_rondas=60, kappa=0.0).fit(X, y)
    con = AditivoPenalizado(n_rondas=60, kappa=200.0).fit(X, y)
    assert rug(con) < rug(sin)
