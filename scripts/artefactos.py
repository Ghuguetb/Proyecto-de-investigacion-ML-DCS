"""Artefactos de los modelos: pipeline completo reentrenado sobre TODO el entrenamiento, serializado con joblib, mas
metadata.json y una prueba de ida y vuelta (las predicciones del modelo cargado deben ser identicas a las del modelo en
memoria). Un modelo por cada uno de los 7 modelos base, con su mejor combinacion de balanceo y optimizador del
Entregable 2, y el modelo original. El dashboard solo carga estos archivos, nunca reentrena."""
import json, os, platform, sys, warnings
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import joblib, numpy as np, pandas as pd
import sklearn, scipy, imblearn, xgboost
from sklearn.model_selection import train_test_split
from scripts.run_original import MEJORES, crear_original
from src import benchmark as B
from src.utils import ORDENES, OBJETIVO, PREDICTORAS, SEED, cargar_particiones
warnings.filterwarnings("ignore")
ART = RAIZ / "artifacts"; ART.mkdir(exist_ok=True); OUT = RAIZ / "resultados"
train, test = cargar_particiones()
Xtr, ytr = train[PREDICTORAS].reset_index(drop=True), train[OBJETIVO].values
Xte = test[PREDICTORAS].reset_index(drop=True)
FIJA = Xte.iloc[:300]                                     # conjunto de prueba fijo para la ida y vuelta
maestra = pd.read_csv(OUT / "tabla_maestra.csv")
esquema = {"columnas": PREDICTORAS,
           "tipos": {c: ("numerica" if c == "edad" else "categorica") for c in PREDICTORAS},
           "rango_edad": [18, 85], "categorias": {c: sorted(Xtr[c].dropna().unique().tolist()) for c in PREDICTORAS if c != "edad"},
           "admite_faltantes": True}
versiones = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
             "imbalanced-learn": imblearn.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
sub, _ = train_test_split(np.arange(len(Xtr)), train_size=B.N_SUBMUESTRA, stratify=ytr, random_state=SEED)
resumen = []
for m, (bal, opt) in {**MEJORES, "AditivoPenalizado": ("sin_balanceo", "Bayesiana")}.items():
    if m == "AditivoPenalizado":
        params = json.load(open(OUT / "hiperparametros_original.json")); modelo = crear_original(params).fit(Xtr, ytr)
        fila = pd.read_csv(OUT / "original_metricas_externas.csv").iloc[0]
        metricas = {k: [float(fila[f"{k}_media"]), float(fila[f"{k}_sd"])] for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE"]}
    else:
        ev = B.Evaluador(m, bal, Xtr.iloc[sub], ytr[sub], SEED); B.BUSCADORES[opt](ev, m, SEED); params = B.mejores_params(ev)
        modelo = B.crear_pipeline(m, bal, params).fit(Xtr, ytr)
        r = maestra[(maestra.modelo == m) & (maestra.balanceo == bal) & (maestra.optimizador == opt)].iloc[0]
        metricas = {k: [float(r[f"{k}_media"]), float(r[f"{k}_sd"])] for k in ["PR_AUC", "ROC_AUC", "Brier", "ECE"]}
    ruta = ART / f"{m}.joblib"; joblib.dump(modelo, ruta, compress=3)
    esperado = modelo.predict_proba(FIJA)[:, 1]
    cargado = joblib.load(ruta).predict_proba(FIJA)[:, 1]
    assert np.array_equal(esperado, cargado), f"La ida y vuelta de {m} no reproduce las predicciones"
    np.save(ART / f"{m}_prueba_ida_y_vuelta.npy", esperado)
    meta = {"modelo": m, "balanceo": bal, "optimizador": opt, "semilla": SEED, "hiperparametros_finales": params,
            "metricas_bucle_externo_media_sd": metricas, "versiones": versiones, "esquema_entrada": esquema,
            "entrenado_con": f"{len(Xtr)} filas (entrenamiento completo, sin la prueba)", "archivo": ruta.name,
            "tamano_MB": round(ruta.stat().st_size / 1e6, 2), "prueba_ida_y_vuelta": "identica (np.array_equal) en 300 filas fijas"}
    json.dump(meta, open(ART / f"{m}_metadata.json", "w"), indent=1, default=str, ensure_ascii=False)
    resumen.append((m, bal, opt, meta["tamano_MB"])); print(m, bal, opt, f"{meta['tamano_MB']} MB", flush=True)
pd.DataFrame(resumen, columns=["modelo", "balanceo", "optimizador", "tamano_MB"]).to_csv(OUT / "artefactos_resumen.csv", index=False)
print("FIN")
