"""Construye y ejecuta los notebooks de los capitulos 7 (modelo original) y 8 (validacion estadistica) del libro.
Uso, desde la raiz del repositorio: python scripts/construir_capitulos.py [7] [8]"""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ / "scripts"))
NB = RAIZ / "notebooks"
que = sys.argv[1:] or ["7", "8"]
if "7" in que:
    from cap7 import construir as c7
    nb = c7(); nbf.write(nb, NB / "06_modelo_original.ipynb")
    NotebookClient(nbf.read(NB / "06_modelo_original.ipynb", as_version=4), timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}})
    nb = nbf.read(NB / "06_modelo_original.ipynb", as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    nbf.write(nb, NB / "06_modelo_original.ipynb"); print("capitulo 7 listo")
if "8" in que:
    from cap8 import construir as c8
    nb = c8(); nbf.write(nb, NB / "07_validacion_estadistica.ipynb")
    nb = nbf.read(NB / "07_validacion_estadistica.ipynb", as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
    nbf.write(nb, NB / "07_validacion_estadistica.ipynb"); print("capitulo 8 listo")
