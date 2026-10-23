# 9. Dashboard y despliegue

El dashboard es la versión interactiva del informe. Está hecho con Dash y Plotly, está publicado en Render y recorre el mismo camino del libro en siete pestañas.

**URL pública:** <https://proyecto-de-investigacion-ml-dcs.onrender.com/>

```{admonition} Primer acceso
:class: warning
El plan gratuito de Render suspende el servicio tras un rato sin visitas, así que el primer acceso puede tardar cerca de 40 segundos. Conviene abrir la URL varios minutos antes de una demostración.
```

## 9.1 Arquitectura

```text
dashboard_dcs/
├── Procfile · runtime.txt · requirements.txt     despliegue (gunicorn, Python 3.12.7, versiones exactas)
├── data/                  datos de entrenamiento y prueba del proyecto
├── data_e3/               copias de los resultados del libro (tablas y JSON que lee el dashboard)
├── cache/artefactos.pkl   resultados pesados calculados una sola vez
├── artifacts/             modelo original ya entrenado (joblib) y su metadata.json
└── src/
    ├── app.py             diseño de las pestañas y callbacks, expone server para gunicorn
    ├── assets/            estilos y figuras
    └── dcs/               datos, figuras, cálculos y el modelo original (original_model.py)
```

La interfaz, la lógica de datos y los artefactos del modelo están separados. El dashboard **nunca reentrena ni optimiza**, solo carga modelos ya entrenados y resultados ya calculados.

## 9.2 Contenido

| Pestaña | Qué muestra |
|---|---|
| 1 Contexto del problema | Planteamiento con cifras citadas, pregunta de investigación, datos y diccionario, flujo del pipeline y limitaciones |
| 2 Análisis exploratorio | Filtros globales con callbacks, distribución del objetivo, relaciones bivariadas con explorador de ejes, correlaciones, faltantes y una frase de interpretación en cada gráfico |
| 3 Modelos base | Regresión logística y SVM con matriz de confusión, ROC, precisión-recall, calibración y un simulador de riesgo |
| 4 Benchmark | Las 108 combinaciones, Friedman y Nemenyi, tabla maestra filtrable y DeLong |
| 5 Optimización y costo | Convergencia, frente de Pareto y complejidad de cada optimizador |
| 6 Resultado final | Evaluación en la prueba, calibración, SHAP, LIME y robustez |
| 7 Modelo original | Tabla de líderes ordenable y filtrable, predicción en vivo con explicación exacta, pseudocódigo, curvas, ablación y verificación estadística con el veredicto |

## 9.3 Decisiones de diseño y de ingeniería

* **Un solo vistazo por pregunta.** Cada gráfico responde una pregunta de una persona no experta y lleva una frase de interpretación debajo.
* **Paleta limitada y tipografía única**, con contraste suficiente y sin depender solo del color (las barras de contribución también cambian de dirección y de etiqueta).
* **Rendimiento.** Los datos y los modelos se cargan una sola vez al iniciar. Las pestañas se dibujan al abrirlas para que Plotly calcule bien el tamaño, y los cálculos pesados están en `cache/artefactos.pkl`.
* **Robustez.** El formulario de predicción valida que no falten campos y que la edad esté entre 18 y 85 años, y muestra un aviso en vez de detener la aplicación.
* **Sin datos ni rutas en el código.** Las rutas salen de la ubicación del archivo, y no hay credenciales.

## 9.4 Despliegue en Render

1. La aplicación corre localmente con `python src/app.py` y expone `server` para gunicorn.
2. `requirements.txt` fija las versiones exactas, incluida `gunicorn`, y `runtime.txt` fija Python 3.12.7.
3. El repositorio de GitHub es público y Render lo despliega como Web Service con el comando de inicio `gunicorn --timeout 600 --chdir src app:server`, descrito en `render.yaml`.
4. Cada *push* a la rama principal dispara un nuevo despliegue automático.

## 9.5 Lo que no hicimos

* El dashboard usa pestañas dentro de una sola página y no la opción `use_pages` de Dash, que el documento del entregable sugiere. El contenido mínimo de cada sección se conserva.
* La predicción en vivo carga el modelo original. Los otros seis modelos base están serializados en `artifacts/` y se usan en las pruebas de ida y vuelta, pero no se cargan en el dashboard para no sobrepasar la memoria del plan gratuito, y porque XGBoost necesitaría una dependencia pesada.
