# 11. Reproducibilidad, autoría y bibliografía

## 11.1 Entorno y semillas

| Elemento | Detalle |
|---|---|
| Capítulos 1 a 4 | Python 3.12 con `requirements.txt` (versiones exactas) |
| Capítulos 5 a 8, scripts y pruebas | Python 3.13 con `requirements-benchmark.txt` (versiones exactas, incluye `pytest`) |
| Dashboard | Python 3.12.7 en Render, con `dashboard_dcs/requirements.txt` (versiones exactas) y `dashboard_dcs/runtime.txt` |
| Semilla global | `SEED = 42` en `src/utils.py` |
| Semilla por pliegue externo | 42 más el índice del pliegue (repetición 0, la del segundo entregable) |
| Semilla de las repeticiones | 42 + 1000 r para la partición externa de la repetición r, y 42 + 1000 r + índice del pliegue para la búsqueda |
| Semilla del modelo original | `random_state = 42`, que fija el orden en que se visitan las variables en cada ronda |

## 11.2 Cómo reproducir los resultados del modelo original

```bash
pip install -r requirements-benchmark.txt
pytest tests/                                    # 8 pruebas del modelo y la ida y vuelta de los artefactos
python scripts/run_original.py 0 1 2 3 4         # nested CV repetido, modelo original y siete mejores benchmarks
python scripts/run_original.py --ebm 1 2 3 4     # EBM en las repeticiones 1 a 4 (la 0 viene del segundo entregable)
python scripts/original_final.py                 # prueba, ablación, convergencia, complejidad e interpretabilidad
python scripts/validacion_original.py            # pruebas de hipótesis y veredicto R3
python scripts/artefactos.py                     # artifacts/ con joblib y metadata.json
python scripts/tabla_maestra_completa.py         # las 140 combinaciones y el modelo original
python scripts/construir_capitulos.py            # notebooks 06 y 07 del libro
jupyter-book build .
```

Las repeticiones se pueden correr por separado y en paralelo, y cada tarea se guarda al terminar, así que el proceso se puede reanudar.

## 11.3 Tabla maestra de experimentos

`resultados/tabla_maestra_completa.csv` lista las **140 combinaciones** del diseño del segundo entregable (112 de clasificación y 28 de regresión), el EBM y el modelo original. Para cada una trae el estado (evaluada, no aplica o no ejecutada), la métrica media y su desviación en el bucle externo, los tiempos y las semillas. Las 4 combinaciones de KNN con pesos de clase no aplican porque KNN no admite pesos, y las 28 de regresión no se ejecutaron por las razones del capítulo 5. Los resultados por pliegue están en `resultados/resultados_por_fold.csv` y `resultados/original_resultados_por_fold.csv`.

## 11.4 Declaración de autoría y uso de herramientas

Los autores de este trabajo somos Santiago Díaz, Gina Huguet y Leiry Mares. Declaramos lo siguiente.

* Usamos **Claude** (Anthropic) como asistente de programación y de redacción. Lo usamos para escribir y depurar código de los scripts, del dashboard y del modelo original, para proponer pruebas estadísticas y para redactar borradores del informe. Todo lo que entregamos lo revisamos, lo ejecutamos y lo podemos explicar.
* El **modelo original** (`src/original_model.py`) está programado solo con numpy y pandas, sin importar ningún modelo de otra librería. Su regla de actualización, su cota cuadrática y su propiedad de no aumento del objetivo están derivadas en el capítulo 7 y verificadas con pruebas unitarias. Cualquier parte del algoritmo la podemos explicar y modificar en vivo.
* Para la búsqueda bibliográfica usamos **Scopus AI** y **PubMed**, y un buscador web general. Las cadenas, fechas y resultados están en el capítulo 6.
* El resto del software es de código abierto y se cita en las referencias: scikit-learn, imbalanced-learn, XGBoost, Optuna, DEAP, SHAP, LIME, `interpret` (solo para el EBM del capítulo 5), Dash y Plotly.
* No copiamos implementaciones de repositorios. Las ideas de otros trabajos se citan en las referencias.
* Los resultados desfavorables se reportan tal como salieron, incluido que el modelo original no supera de forma significativa a los benchmarks (capítulo 8). No se manipularon particiones, semillas ni pruebas.

Firmas del equipo

Santiago Díaz: ____________________  Gina Huguet: ____________________  Leiry Mares: ____________________

## 11.5 Bibliografía

La bibliografía completa está en `references.bib`, y las citas del texto se resuelven con ella. La lista de referencias del proyecto sigue en el capítulo de referencias.

```{bibliography}
:all:
```
