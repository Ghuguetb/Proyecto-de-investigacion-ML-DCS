"""Modelo original: aditivo penalizado por rondas (AditivoPenalizado).

Implementacion propia, solo con numpy y pandas. No importa ningun modelo de otra libreria.

Idea
----
La probabilidad de dificultad cognitiva se modela en la escala logit como una suma de curvas, una por variable:

    logit P(y=1 | x) = b0 + f_1(x_1) + ... + f_p(x_p)

Cada f_j es una tabla con un valor por nivel (variable categorica) o por tramo de edad (variable numerica).
Las tablas se aprenden por rondas. En cada ronda se visita cada variable (en orden aleatorio) y se corrige solo su
tabla con una regla de actualizacion derivada de una cota cuadratica de la perdida logistica.

Objetivo penalizado
-------------------
    J = sum_i log(1 + exp(F_i)) - y_i F_i + (lambda/2) sum_j ||f_j||^2 + (kappa/2) sum_{j ordenada} sum_b (f_j[b+1] - f_j[b])^2

* lambda (ridge) encoge las tablas y vuelve estable el ajuste de niveles raros.
* kappa (suavizado) obliga a que niveles vecinos de una variable ordenada (edad, educacion, frecuencia de depresion,
  etc.) tengan valores parecidos. Es la regla propia del modelo.

Regla de actualizacion
----------------------
La segunda derivada de la perdida logistica respecto de F es p(1-p) <= 1/4. Por eso, para un cambio d en la tabla de
la variable j, la perdida cumple la cota L(F + d) <= L(F) + g.d + (1/8) sum_i d_i^2, con g_i = p_i - y_i.
Minimizar esa cota mas las penalizaciones da un sistema lineal pequeno (tridiagonal si la variable es ordenada):

    (diag(n_b/4 + lambda) + kappa D) d = -(G + lambda f_j + kappa D f_j),   G_b = sum_{i en b} g_i

donde n_b es el numero de filas en el nivel b y D es el laplaciano de la cadena de niveles. Luego f_j <- f_j + tasa d.
Como la cota es convexa y vale 0 en d = 0, cualquier tasa en (0, 1] hace que J no aumente. Esa es la propiedad de
convergencia que verifica tests/test_original_model.py.

Complejidad
-----------
Entrenamiento O(R p (n + B^3)) con R rondas, p variables, n filas y B niveles por variable (B <= n_bins + 1 y
pequeno). Prediccion O(p) por fila. Memoria O(n + p B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _sigmoide(z):
    """Sigmoide estable numericamente."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class AditivoPenalizado(ClassifierMixin, BaseEstimator):
    """Clasificador binario aditivo penalizado, ajustado por rondas con cota cuadratica.

    Parameters
    ----------
    n_rondas : int
        Numero de pasadas completas sobre las variables.
    tasa : float
        Factor de paso en (0, 1]. Con tasa <= 1 el objetivo penalizado nunca aumenta.
    lambda_ : float
        Penalizacion ridge (>= 0) sobre los valores de cada tabla.
    kappa : float
        Penalizacion de suavizado (>= 0) entre niveles vecinos de las variables ordenadas. kappa = 0 apaga la regla.
    n_bins : int
        Numero maximo de tramos para variables numericas (cuantiles).
    ordenes : dict or None
        Para variables categoricas ordenadas, {columna: [niveles en orden]}. Las demas categoricas no se suavizan.
    random_state : int
        Semilla del orden aleatorio en que se visitan las variables en cada ronda.

    Attributes
    ----------
    classes_ : ndarray
    intercepto_ : float
    funciones_ : dict
        {columna: ndarray con el valor de la curva por nivel}, centradas en el conjunto de entrenamiento.
    perdida_ : ndarray (n_rondas,)
        Objetivo penalizado J despues de cada ronda (no creciente).
    """

    def __init__(self, n_rondas=100, tasa=0.5, lambda_=10.0, kappa=0.0, n_bins=32, ordenes=None, random_state=42):
        self.n_rondas = n_rondas
        self.tasa = tasa
        self.lambda_ = lambda_
        self.kappa = kappa
        self.n_bins = n_bins
        self.ordenes = ordenes
        self.random_state = random_state

    # ------------------------------------------------------------------ validacion y codificacion
    def _validar_parametros(self):
        if not (isinstance(self.n_rondas, (int, np.integer)) and self.n_rondas >= 1):
            raise ValueError("n_rondas debe ser un entero >= 1")
        if not (0 < self.tasa <= 1):
            raise ValueError("tasa debe estar en (0, 1]")
        if self.lambda_ < 0 or self.kappa < 0:
            raise ValueError("lambda_ y kappa deben ser >= 0")
        if self.n_bins < 2:
            raise ValueError("n_bins debe ser >= 2")

    @staticmethod
    def _a_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("X debe ser una tabla de dos dimensiones")
        return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])

    def _ajustar_codificador(self, X):
        """Aprende, solo con el entrenamiento, los tramos (numericas) y los niveles (categoricas)."""
        ordenes = self.ordenes or {}
        self.columnas_, self.tipos_, self.cortes_, self.niveles_, self.ordenada_ = list(X.columns), {}, {}, {}, {}
        for c in self.columnas_:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col):
                v = col.dropna().to_numpy(dtype=float)
                if len(v) == 0:
                    raise ValueError(f"La columna {c} no tiene valores observados")
                cortes = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                self.tipos_[c], self.cortes_[c], self.ordenada_[c] = "num", cortes, True
            else:
                niveles = list(ordenes[c]) if c in ordenes else sorted(col.dropna().astype(str).unique())
                self.tipos_[c], self.niveles_[c], self.ordenada_[c] = "cat", niveles, c in ordenes

    def _codificar(self, X, c):
        """Codigo entero por fila. El ultimo codigo es siempre 'faltante o nivel no visto'."""
        col = X[c]
        if self.tipos_[c] == "num":
            v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            cod = np.searchsorted(self.cortes_[c], v, side="right")
            cod[np.isnan(v)] = len(self.cortes_[c]) + 1
            return cod.astype(int)
        mapa = {n: i for i, n in enumerate(self.niveles_[c])}
        cod = col.map(lambda z: mapa.get(str(z), len(mapa)) if pd.notna(z) else len(mapa))
        return cod.to_numpy(dtype=int)

    def _n_niveles(self, c):
        return (len(self.cortes_[c]) + 2) if self.tipos_[c] == "num" else (len(self.niveles_[c]) + 1)

    @staticmethod
    def _laplaciano(B):
        """Laplaciano de la cadena de los primeros B-1 niveles (el ultimo, 'faltante', queda aislado)."""
        D = np.zeros((B, B))
        for b in range(B - 2):
            D[b, b] += 1; D[b + 1, b + 1] += 1; D[b, b + 1] -= 1; D[b + 1, b] -= 1
        return D

    # ------------------------------------------------------------------ ajuste
    def fit(self, X, y):
        """Ajusta el modelo por rondas.

        Parameters
        ----------
        X : DataFrame (n, p) con columnas numericas o categoricas (texto), admite faltantes.
        y : array (n,) con dos clases.
        """
        self._validar_parametros()
        X = self._a_dataframe(X)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X e y tienen distinto numero de filas")
        if pd.isna(y).any():
            raise ValueError("y no puede tener faltantes")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Solo se admite clasificacion binaria")
        yb = (y == self.classes_[1]).astype(float)
        n = len(yb)
        self._ajustar_codificador(X)
        cods = {c: self._codificar(X, c) for c in self.columnas_}
        cuentas = {c: np.bincount(cods[c], minlength=self._n_niveles(c)).astype(float) for c in self.columnas_}
        lap = {c: (self._laplaciano(self._n_niveles(c)) if self.ordenada_[c] else np.zeros((self._n_niveles(c),) * 2))
               for c in self.columnas_}
        f = {c: np.zeros(self._n_niveles(c)) for c in self.columnas_}
        b0 = float(np.log(yb.mean() / (1 - yb.mean()))) if 0 < yb.mean() < 1 else 0.0
        F = np.full(n, b0)
        rng = np.random.default_rng(self.random_state)
        perdida = np.empty(self.n_rondas)
        for r in range(self.n_rondas):
            p = _sigmoide(F)
            d0 = -(p - yb).sum() / (n / 4.0)               # intercepto, sin penalizacion
            b0 += self.tasa * d0; F += self.tasa * d0
            for c in rng.permutation(self.columnas_):
                p = _sigmoide(F)
                G = np.bincount(cods[c], weights=p - yb, minlength=len(f[c]))
                A = np.diag(cuentas[c] / 4.0 + self.lambda_) + self.kappa * lap[c]
                rhs = -(G + self.lambda_ * f[c] + self.kappa * lap[c] @ f[c])
                d = np.linalg.solve(A, rhs)
                f[c] += self.tasa * d
                F += self.tasa * d[cods[c]]
            perdida[r] = self._objetivo(F, yb, f, lap)
        # Se centra cada curva en el entrenamiento: la media de f_j pasa al intercepto (la prediccion no cambia)
        for c in self.columnas_:
            m = float((cuentas[c] * f[c]).sum() / n)
            f[c] = f[c] - m; b0 += m
        self.intercepto_, self.funciones_, self.perdida_ = b0, f, perdida
        self.cuentas_ = cuentas
        self.n_features_in_ = len(self.columnas_)
        self.feature_names_in_ = np.array(self.columnas_, dtype=object)
        return self

    def _objetivo(self, F, yb, f, lap):
        """Objetivo penalizado J (perdida logistica mas ridge mas suavizado)."""
        perd = np.logaddexp(0.0, F).sum() - (yb * F).sum()
        pen = sum(0.5 * self.lambda_ * (f[c] @ f[c]) + 0.5 * self.kappa * (f[c] @ lap[c] @ f[c]) for c in f)
        return float(perd + pen)

    # ------------------------------------------------------------------ prediccion
    def _logit(self, X):
        check_is_fitted(self, ["funciones_", "intercepto_"])
        X = self._a_dataframe(X)
        faltan = [c for c in self.columnas_ if c not in X.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en X: {faltan}")
        F = np.full(len(X), self.intercepto_)
        for c in self.columnas_:
            F += self.funciones_[c][self._codificar(X, c)]
        return F

    def decision_function(self, X):
        """Logit de la clase positiva."""
        return self._logit(X)

    def predict_proba(self, X):
        """Probabilidades (n, 2) en el orden de classes_."""
        p = _sigmoide(self._logit(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Clase predicha con umbral 0.5."""
        check_is_fitted(self, ["funciones_", "classes_"])
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    # ------------------------------------------------------------------ interpretabilidad
    def contribuciones(self, X):
        """Contribucion de cada variable al logit, f_j(x_j), centrada en el entrenamiento.

        En un modelo aditivo con curvas centradas, esta contribucion coincide con el valor SHAP exacto de la variable
        (fondo independiente) porque phi_j = f_j(x_j) - E[f_j(X)] y E[f_j(X)] = 0.
        """
        check_is_fitted(self, ["funciones_"])
        X = self._a_dataframe(X)
        return pd.DataFrame({c: self.funciones_[c][self._codificar(X, c)] for c in self.columnas_})

    def importancias(self):
        """Importancia global de cada variable, media del valor absoluto de su contribucion en el entrenamiento."""
        check_is_fitted(self, ["funciones_"])
        n = len(next(iter(self.cuentas_.values()))) and sum(next(iter(self.cuentas_.values())))
        return pd.Series({c: float((self.cuentas_[c] * np.abs(self.funciones_[c])).sum() / n)
                          for c in self.columnas_}).sort_values(ascending=False)
