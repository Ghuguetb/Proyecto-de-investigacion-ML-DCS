"""Pruebas estadisticas: DeLong (comparacion de AUC correlacionadas) y correccion de Holm."""
import numpy as np
from scipy import stats


def midrank(x):
    """Rangos medios de un vector (empates promediados), como los pide DeLong."""
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1; i = a
    out = np.empty(n); out[j] = t
    return out


def delong_cov(y, preds):
    """AUC y matriz de covarianza de DeLong de varios clasificadores sobre las mismas observaciones.
    
    Parameters
    ----------
    y : array of int
    preds : ndarray (K, n)
        Puntajes de K clasificadores.
    
    Returns
    -------
    aucs : ndarray (K,)
    cov : ndarray (K, K)
    """
    pos = preds[:, y == 1]; neg = preds[:, y == 0]; m, n = pos.shape[1], neg.shape[1]; K = preds.shape[0]
    tx = np.array([midrank(pos[r]) for r in range(K)]); ty = np.array([midrank(neg[r]) for r in range(K)])
    tz = np.array([midrank(preds[r]) for r in range(K)])
    aucs = np.array([(tz[r, y == 1].sum() - m * (m + 1) / 2) / (m * n) for r in range(K)])
    v01 = (tz[:, y == 1] - tx) / n; v10 = 1 - (tz[:, y == 0] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n


def delong_p(y, p1, p2):
    """Prueba de DeLong para la diferencia de AUC entre dos clasificadores correlacionados.
    
    Returns
    -------
    (diferencia de AUC, p valor bilateral)
    """
    aucs, S = delong_cov(np.asarray(y), np.vstack([p1, p2]))
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-18))
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))


def holm(pv):
    """Correccion de Holm-Bonferroni de una lista de p valores. Devuelve los p ajustados en el mismo orden."""
    pv = np.asarray(pv); o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * pv[i]); adj[i] = min(1, run)
    return adj
