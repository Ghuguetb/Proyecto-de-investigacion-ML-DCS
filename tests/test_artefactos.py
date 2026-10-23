"""Prueba de ida y vuelta: el modelo serializado debe dar exactamente las predicciones guardadas al serializarlo."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, pytest
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
ART = RAIZ / "artifacts"
MODELOS = sorted(p.name.replace("_metadata.json", "") for p in ART.glob("*_metadata.json"))


@pytest.mark.skipif(not MODELOS, reason="artifacts/ no existe (correr scripts/artefactos.py)")
@pytest.mark.parametrize("modelo", MODELOS)
def test_ida_y_vuelta(modelo):
    from src.utils import PREDICTORAS, cargar_particiones
    _, test = cargar_particiones()
    fija = test[PREDICTORAS].reset_index(drop=True).iloc[:300]
    p = joblib.load(ART / f"{modelo}.joblib").predict_proba(fija)[:, 1]
    assert np.array_equal(p, np.load(ART / f"{modelo}_prueba_ida_y_vuelta.npy"))
    meta = json.load(open(ART / f"{modelo}_metadata.json"))
    assert meta["semilla"] == 42 and set(meta["esquema_entrada"]["columnas"]) == set(PREDICTORAS)
    assert meta["tamano_MB"] < 90          # limite de GitHub por archivo: 100 MB
