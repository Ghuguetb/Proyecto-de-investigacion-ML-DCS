# Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia (NHIS 2022–2023)

Primer entregable del proyecto de Machine Learning, selección de la base de datos, EDA y modelo base.

## Estructura

```
proyecto_nhis_cognicion/
├── _config.yml, _toc.yml        Configuración del Jupyter Book
├── 00_intro.md, conclusiones.md, referencias.md
├── notebooks/
│   ├── 00_proyecto_completo.ipynb       Notebook único con las 4 etapas, texto final
│   ├── 01_base_de_datos.ipynb           Sección 1 de la guía
│   ├── 02_eda.ipynb                     Secciones 2.1–2.4 y 2.6–2.8
│   ├── 03_fuga_y_preprocesamiento.ipynb Secciones 2.5 y 2.9
│   └── 04_modelo_base.ipynb             Sección 3
├── src/
│   ├── utils.py               Carga, recodificación (codebook), diccionario, partición, SEED = 42
│   └── preprocesamiento.py    Pipeline de scikit-learn
├── data/
│   ├── raw/                   adult22csv.zip y adult23csv.zip (CDC)
│   └── processed/             Generados por el notebook 01
├── models/                    Modelo base guardado por el notebook 04
└── requirements.txt           Dependencias con versiones fijadas (Python 3.12)
```

## Cómo reproducir

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Ejecutar en orden: notebooks/01 → 02 → 03 → 04 (Run All en VS Code)
jupyter-book build .
# Abrir _build/html/index.html
```

La guía paso a paso detallada está en el documento Word que acompaña al proyecto.
