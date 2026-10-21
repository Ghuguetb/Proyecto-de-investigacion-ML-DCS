"""Uso: python scripts/run_benchmark.py [--modelos A B] [--folds 0 1 2 3 4] [--jobs 2]
Reanudable. Cada tarea guarda su resultado en resultados/tareas/."""
import argparse, itertools, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from joblib import Parallel, delayed
from src import benchmark as B

ap = argparse.ArgumentParser()
ap.add_argument("--modelos", nargs="*", default=B.MODELOS)
ap.add_argument("--balanceos", nargs="*", default=B.BALANCEOS)
ap.add_argument("--opts", nargs="*", default=B.OPTIMIZADORES)
ap.add_argument("--folds", nargs="*", type=int, default=list(range(B.N_EXTERNO)))
ap.add_argument("--jobs", type=int, default=2)
a = ap.parse_args()
carpeta = RAIZ / "resultados" / "tareas"; carpeta.mkdir(parents=True, exist_ok=True)
X, y = B.cargar_xy()

def uno(m, b, o, f):
    return B.ejecutar_tarea(m, b, o, f, X, y, carpeta)

tareas = [(m, b, o, f) for f in a.folds for m in a.modelos for b in a.balanceos for o in a.opts
          if B.balanceo_aplicable(m, b)]
print(f"{len(tareas)} tareas", flush=True)
Parallel(n_jobs=a.jobs, verbose=5)(delayed(uno)(*t) for t in tareas)
