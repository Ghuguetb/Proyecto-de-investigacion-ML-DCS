# Dashboard DCS · Entregable 2

Dificultad cognitiva subjetiva (DCS) y factores de riesgo modificables de demencia · NHIS 2022-2023.
Autores: Santiago Díaz, Gina Huguet y Leiry Mares (Maestría en Ingeniería Biomédica, Universidad del Norte).

Dashboard en Dash Plotly, creado con `dash-tools`, con tres pestañas: **Contexto del problema**, **EDA** y **Modelos base** (regresión logística y SVM lineal).

## Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

Abre <http://127.0.0.1:8050/>. La **primera ejecución** entrena los modelos, el bootstrap y el PCA y guarda los resultados en `cache/artefactos.pkl`
(unos 2-3 minutos); las siguientes abren en segundos. La caché se regenera sola si cambia la versión de scikit-learn.

Para publicarlo (Render, Heroku, etc.) el `Procfile` ya lanza `gunicorn --timeout 600 --chdir src app:server`. El `timeout` alto cubre el primer cálculo.

## Estructura

```text
src/app.py            diseño, pestañas y callbacks (comentado por secciones)
src/assets/           style.css (tema claro crema/tierras) y bootstrap.min.css local
src/dcs/calculos.py   modelos, bootstrap, subgrupos, odds ratios, PCA, caché
src/dcs/figuras.py    todas las figuras Plotly (las usa también el JBook)
src/dcs/utils.py      rutas, diccionario de variables y carga de particiones
data/                 train.csv, test.csv y la base completa del Entregable 1
```

## Nota sobre dash-tools

`pip install dash-tools` instala el comando `dashtools` (que se usó con `dashtools init <nombre> tabs` para generar la estructura base).
Ese comando requiere `dash<3` y `setuptools<81`; para solo ejecutar este dashboard no hace falta, y `requirements.txt` no fija esas versiones.
