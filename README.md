# Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia (NHIS 2022–2023)

Proyecto de Machine Learning, dificultad cognitiva subjetiva y factores de riesgo modificables de demencia (NHIS 2022-2023). Incluye la base de datos, el EDA, la auditoría de fuga, el modelo base y un benchmark de modelos con validación anidada.

## Estructura

```
├── _config.yml, _toc.yml        Configuración del Jupyter Book
├── 00_intro.md, conclusiones.md, referencias.md
├── notebooks/
│   ├── 01_base_de_datos.ipynb             Capítulo 1
│   ├── 02_eda.ipynb                       Capítulo 2
│   ├── 03_fuga_y_preprocesamiento.ipynb   Capítulo 3
│   ├── 04_modelo_base.ipynb               Capítulo 4
│   ├── 05_benchmark.ipynb                 Capítulo 5, benchmark 7x4x4, EBM, estadística, robustez
│   └── 00_proyecto_completo.ipynb         Versión previa en un solo notebook (no forma parte del libro)
├── src/
│   ├── utils.py               Carga, recodificación (codebook), diccionario, partición, SEED = 42
│   ├── preprocesamiento.py    Pipeline de scikit-learn
│   ├── benchmark.py           Modelos, balanceos, optimizadores y validación anidada
│   └── estadistica.py         DeLong y Holm
├── scripts/                   Ejecución del benchmark, análisis, evaluación final, complejidad, LIME vs SHAP
├── resultados/                Tabla maestra, resultados por tarea, figuras y tablas del capítulo 5
├── dashboard_dcs/             Dashboard interactivo (Dash) con seis pestañas
├── data/
│   ├── raw/                   adult22csv.zip y adult23csv.zip (CDC)
│   └── processed/             Generados por el notebook 01
├── models/                    Modelo base guardado por el notebook 04
├── requirements.txt           Dependencias de los capítulos 1 a 4 (Python 3.12)
└── requirements-benchmark.txt Dependencias exactas del capítulo 5 (Python 3.13)
```

## Cómo reproducir

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Ejecutar en orden: notebooks/01 → 02 → 03 → 04 (Run All en VS Code)
# Capítulo 5 (otro entorno, Python 3.13): pip install -r requirements-benchmark.txt
python scripts/run_benchmark.py && python scripts/run_ebm.py && python scripts/analisis.py 20
python scripts/finales.py && python scripts/complejidad.py && python scripts/lime_vs_shap.py
# Luego ejecutar notebooks/05_benchmark.ipynb
jupyter-book build .
# Abrir _build/html/index.html
```

## Dashboard

```bash
cd dashboard_dcs
pip install -r requirements.txt
python src/app.py        # http://127.0.0.1:8050
```
