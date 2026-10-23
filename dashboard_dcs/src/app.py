"""
app.py · Dashboard del proyecto
===================================
Dificultad cognitiva subjetiva (DCS) y factores de riesgo modificables de demencia · NHIS 2022-2023
Autores: Santiago Díaz, Gina Huguet y Leiry Mares · Maestría en Ingeniería Biomédica, Universidad del Norte

Estructura generada con dash-tools (plantilla "tabs") y ampliada para el proyecto:

    dashboard_dcs/
    ├── Procfile · runtime.txt · requirements.txt     (despliegue, creados por dashtools)
    ├── data/                  train.csv, test.csv y base completa del proyecto
    ├── cache/artefactos.pkl   modelos y resultados pesados, calculados una sola vez
    └── src/
        ├── app.py             ESTE ARCHIVO: diseño de las 3 pestañas y callbacks
        ├── assets/style.css   estilos (fondo claro, paleta de tierras)
        └── dcs/               utils, preprocesamiento, calculos y figuras (compartidos con el JBook)

Para ejecutarlo:   python src/app.py     →    http://127.0.0.1:8050/

Secciones de este archivo
-------------------------
0. Configuración y carga de datos
1. Componentes de interfaz reutilizables
2. Pestaña 1 · Contexto del problema
3. Pestaña 2 · EDA (filtros + gráficas interactivas)
4. Pestaña 3 · Modelos base (regresión logística y SVM lineal)
5. Ensamblado de la app y callbacks
"""
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, dash_table, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent))   # para importar el paquete dcs/

from dcs import benchmark_tabs as E                               # noqa: E402
from dcs import figuras as F                                   # noqa: E402
from dcs.calculos import ETIQUETAS, obtener_artefactos         # noqa: E402
from dcs import original_tab as OT                              # noqa: E402
from dcs.utils import (DICCIONARIO, OBJETIVO, ORDENES, PREDICTORAS, PREDICTORAS_CAT,  # noqa: E402
                       cargar_particiones)

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN Y CARGA DE DATOS
# ---------------------------------------------------------------------------
# Artefactos: modelos entrenados, bootstrap, odds ratios, PCA… (se calculan solo la primera vez).
A = obtener_artefactos()

# El EDA usa únicamente el conjunto de entrenamiento (el de prueba se reserva para evaluar modelos).
TRAIN, TEST = cargar_particiones()
DF = F.preparar(TRAIN)
PUNTOS_PCA = F.preparar(A["pca"]["puntos"].assign(anio=A["pca"]["puntos"]["anio"]))  # muestra de 6 000 adultos

CATS = A["categorias"]                       # categorías de cada predictora tal como las ve el modelo
CONFIG_GRAFICA = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                  "toImageButtonOptions": {"format": "png", "scale": 2}}
RAZAS = ["Blanco no hispano", "Hispano", "Negro no hispano", "Asiático no hispano", "AIAN no hispano",
         "AIAN y otro grupo", "Otro / múltiple"]


# ---------------------------------------------------------------------------
# 1. COMPONENTES DE INTERFAZ REUTILIZABLES
# ---------------------------------------------------------------------------
def n_es(x, dec=0):
    """Número con separadores en español (punto de miles, coma decimal)."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "·").replace(".", ",").replace("·", ".")


def pct_es(x, dec=1):
    return n_es(100 * x, dec) + " %"


def seccion(titulo, subtitulo=None):
    return html.Div([html.H2(titulo), html.P(subtitulo) if subtitulo else None, html.Div(className="barra")],
                    className="seccion")


def tarjeta(titulo, cuerpo, nota=None, clase=""):
    return html.Div([html.Div(titulo, className="card-h") if titulo else None, cuerpo,
                     html.P(nota, className="nota") if nota else None], className=f"card-x {clase}")


def grafica(id_, alto=None):
    return dcc.Graph(id=id_, config=CONFIG_GRAFICA, style={"height": f"{alto}px"} if alto else None)


def kpi(valor, etiqueta, sub=None, color=""):
    return html.Div([html.Div(valor, className="kpi-v"), html.Div(etiqueta, className="kpi-l"),
                     html.Div(sub, className="kpi-s") if sub else None], className=f"kpi {color}")


def estilo_tabla(**extra):
    """Estilo común de las DataTable (claro, tipografía DM Sans)."""
    base = dict(
        style_as_list_view=True, style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#F3E9D8", "fontWeight": "700", "border": "none", "color": "#3A302A",
                      "fontFamily": "DM Sans", "fontSize": "13px"},
        style_cell={"fontFamily": "DM Sans", "fontSize": "13px", "padding": "8px 10px", "textAlign": "left",
                    "border": "none", "borderBottom": "1px solid #EFE6D6", "whiteSpace": "normal", "height": "auto",
                    "color": "#3A302A", "backgroundColor": "white"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FBF7F0"}],
    )
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# 2. PESTAÑA 1 · CONTEXTO DEL PROBLEMA
# ---------------------------------------------------------------------------
def layout_contexto():
    prevalencia = DF[OBJETIVO].mean()
    return html.Div([
        # --- 2.1 Pregunta de investigación y cifras clave -------------------------------------------
        seccion("El problema en una pregunta",
                "La dificultad cognitiva subjetiva (DCS) es la percepción propia de tener problemas para recordar o concentrarse; "
                "quienes la reportan tienen cerca del doble de riesgo de desarrollar demencia."),
        dbc.Row([
            dbc.Col([html.Div([html.Small("Pregunta de investigación"),
                              "¿Qué tan bien pueden predecir la dificultad cognitiva subjetiva los factores de riesgo "
                              "modificables de demencia y el antecedente de accidente cerebrovascular?"], className="pregunta"),
                     html.Figure([html.Img(src="/assets/img/adulto_mayor_dcs.jpg",
                                           alt="Adulto mayor que se lleva la mano a la frente con gesto de preocupación, sosteniendo sus gafas",
                                           style={"width": "100%", "maxWidth": "420px", "borderRadius": "12px", "display": "block", "margin": "16px auto 4px"}),
                                  html.Figcaption("Imagen ilustrativa de la dificultad cognitiva subjetiva.",
                                                  style={"fontSize": "12px", "textAlign": "center", "color": "#6a7179"})], style={"margin": "0"})], lg=7),
            dbc.Col(dbc.Row([
                dbc.Col(kpi("56.129", "adultos analizados", "NHIS 2022-2023, CDC"), width=6, className="mb-3"),
                dbc.Col(kpi(pct_es(0.2056 if False else prevalencia), "con alguna dificultad", "prevalencia en entrenamiento", "teal"), width=6, className="mb-3"),
                dbc.Col(kpi("≈ 2×", "riesgo de demencia", "RR 2,07–2,12 en DCS", "dorado"), width=6),
                dbc.Col(kpi("≈ 45 %", "de la demencia", "atribuible a 14 factores (Lancet 2024)", "osc"), width=6),
            ]), lg=5),
        ], className="g-3"),

        # --- 2.2 Objetivos y alcance ----------------------------------------------------------------
        dbc.Row([
            dbc.Col(tarjeta("Objetivos específicos", html.Ul([
                html.Li([html.Span("1", className="n"), "Caracterizar la prevalencia de DCS y su asociación con cada factor de riesgo, con tamaños de efecto."]),
                html.Li([html.Span("2", className="n"), "Identificar y excluir las variables que constituyan fuga de información."]),
                html.Li([html.Span("3", className="n"), "Entrenar modelos base (regresión logística y SVM lineal) y compararlos con un clasificador trivial."]),
                html.Li([html.Span("4", className="n"), "Interpretar la contribución de cada factor mediante odds ratios, con énfasis en el ACV."]),
            ], className="lista-obj"), clase="acento"), lg=6),
            dbc.Col(tarjeta("Alcance y cautelas", html.Div([
                html.P("Los datos son transversales: el modelo no pronostica un deterioro futuro, estima qué tan bien el perfil de factores "
                       "de riesgo identifica a quienes hoy reportan dificultad cognitiva. Es una prueba de concepto sobre el valor predictivo "
                       "de esa información.", style={"fontSize": "14px"}),
                html.P("La variable objetivo es autorreportada, la muestra es de EE. UU. y excluye a la población institucionalizada, "
                       "así que el modelo no es una herramienta diagnóstica ni debe aplicarse a otra población sin validación local.",
                       style={"fontSize": "14px", "marginBottom": "4px"}),
            ]), clase="acento-teal"), lg=6),
        ], className="g-3"),

        # --- 2.3 Datos: flujo de la muestra y diccionario ------------------------------------------
        seccion("De dónde salen los datos", "Archivos públicos de adultos de la National Health Interview Survey (NCHS/CDC) de 2022 y 2023."),
        dbc.Row([
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_flujo_muestra(), config=CONFIG_GRAFICA),
                            "Se excluyen las entrevistas respondidas por un informante (la pregunta mide la dificultad percibida por la propia persona) "
                            "y 19 registros sin respuesta en el objetivo. La partición 80/20 es estratificada por objetivo y año, y se hizo antes de "
                            "cualquier decisión basada en los datos."), lg=7),
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_lancet_dona(), config=CONFIG_GRAFICA),
                            "De los 14 factores de riesgo modificables de la Comisión Lancet 2024, la encuesta mide 9 con las mismas preguntas en ambos años."), lg=5),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(tarjeta("Factores de riesgo del marco Lancet 2024 y su disponibilidad", html.Div([
                html.Div([html.Div("Filtrar por disponibilidad", className="ctrl-et"),
                          dcc.Dropdown(id="ctx-lancet-filtro", options=["Todos"] + list(F.TABLA_LANCET["Disponibilidad"].unique()),
                                       value="Todos", clearable=False, style={"maxWidth": "320px", "marginBottom": "8px"})]),
                dash_table.DataTable(id="ctx-lancet-tabla", columns=[{"name": c, "id": c} for c in F.TABLA_LANCET.columns],
                                     data=F.TABLA_LANCET.to_dict("records"), page_size=14,
                                     **estilo_tabla(style_data_conditional=[
                                         {"if": {"filter_query": '{Disponibilidad} = "Disponible 2022-2023"', "column_id": "Disponibilidad"}, "color": "#2E6F77", "fontWeight": "700"},
                                         {"if": {"filter_query": '{Disponibilidad} = "Solo 2022"', "column_id": "Disponibilidad"}, "color": "#B07F1F", "fontWeight": "700"},
                                         {"if": {"filter_query": '{Disponibilidad} = "No disponible"', "column_id": "Disponibilidad"}, "color": "#B9543A", "fontWeight": "700"},
                                         {"if": {"row_index": "odd"}, "backgroundColor": "#FBF7F0"}])),
            ]), "Incluir los factores de un solo año obligaría a descartar la mitad de la muestra o a imputar un año completo; por eso quedan declarados como limitación."), lg=12),
        ]),

        dbc.Row([
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_mapa_variables(), config=CONFIG_GRAFICA),
                            "Cada rectángulo es una columna de la base. Pase el cursor para ver la variable original de la NHIS y su significado; "
                            "haga clic en un grupo para acercarse."), lg=12),
        ]),

        tarjeta("Diccionario de variables", html.Div([
            html.Div([html.Div("Filtrar por rol de la variable", className="ctrl-et"),
                      dcc.Dropdown(id="ctx-dic-rol", options=["Todos"] + list(DICCIONARIO["rol"].unique()), value="Todos",
                                   clearable=False, style={"maxWidth": "320px", "marginBottom": "8px"})]),
            dash_table.DataTable(
                id="ctx-dic-tabla",
                columns=[{"name": n, "id": i} for n, i in [("Variable", "variable"), ("Variable NHIS", "variable_nhis"), ("Rol", "rol"), ("Tipo", "tipo"),
                                                          ("Valores", "unidad_categorias"), ("Significado", "significado"), ("Factor Lancet 2024", "factor_lancet_2024")]],
                data=DICCIONARIO.to_dict("records"), page_size=8, filter_action="native", sort_action="native",
                **estilo_tabla(style_cell_conditional=[{"if": {"column_id": "significado"}, "minWidth": "260px"}])),
        ]), "Use las casillas bajo cada encabezado para buscar texto. Las variables con rol «Auditoría de fuga» se descartaron del modelo."),

        # --- 2.4 Ruta del proyecto ------------------------------------------------------------------
        seccion("Ruta del proyecto"),
        tarjeta(None, dbc.Row([
            dbc.Col(html.Div([html.Div("1", className="circulo"), html.Div("Capítulos 1 a 4 del libro", className="t"),
                              html.Div("Base de datos, EDA, fuga de datos y modelo base", className="d")], className="paso"), md=4),
            dbc.Col(html.Div([html.Div("2", className="circulo"), html.Div("Capítulo 5 del libro", className="t"),
                              html.Div("Benchmark de 112 combinaciones, EBM, interpretabilidad y robustez", className="d")], className="paso"), md=4),
            dbc.Col(html.Div([html.Div("3", className="circulo"), html.Div("Este dashboard", className="t"),
                              html.Div("Recorre todo el proyecto de forma interactiva, en seis pestañas", className="d")], className="paso actual"), md=4),
        ])),
    ])


# ---------------------------------------------------------------------------
# 3. PESTAÑA 2 · EDA
# ---------------------------------------------------------------------------
OPC_VARIABLES = [{"label": ETIQUETAS[v], "value": v} for v in PREDICTORAS_CAT]
OPC_CRUCE = OPC_VARIABLES + [{"label": "Grupo de edad", "value": "grupo_edad_amplio"}, {"label": "Año", "value": "anio"}]


def panel_filtros():
    return html.Div(html.Div([
        html.Div("Filtros del análisis", className="card-h", style={"fontSize": "16px"}),
        html.P("Todas las gráficas de esta pestaña se recalculan con la población que elija.", className="nota"),
        html.Label("Año de la entrevista", className="et"),
        dcc.Checklist(id="f-anio", options=[2022, 2023], value=[2022, 2023], inline=True, inputStyle={"marginRight": "5px", "marginLeft": "10px"}),
        html.Label("Sexo", className="et"),
        dcc.Checklist(id="f-sexo", options=["Hombre", "Mujer"], value=["Hombre", "Mujer"], inline=True, inputStyle={"marginRight": "5px", "marginLeft": "10px"}),
        html.Label("Edad (años)", className="et"),
        dcc.RangeSlider(id="f-edad", min=18, max=85, step=1, value=[18, 85], marks={18: "18", 30: "30", 45: "45", 60: "60", 75: "75", 85: "85+"},
                        tooltip={"placement": "bottom", "always_visible": False}, allowCross=False),
        html.Label("Raza / etnia", className="et", style={"marginTop": "20px"}),
        dcc.Dropdown(id="f-raza", options=RAZAS, value=[], multi=True, placeholder="Todas"),
        html.Label("Antecedente de ACV", className="et"),
        dcc.RadioItems(id="f-acv", options=["Todos", "Sí", "No"], value="Todos", inline=True, inputStyle={"marginRight": "5px", "marginLeft": "10px"}),
        html.Button("Restablecer filtros", id="f-reset", className="btn-dcs sec"),
        html.Button("Descargar datos filtrados (CSV)", id="f-descarga", className="btn-dcs"),
        dcc.Download(id="f-descarga-datos"),
    ], className="card-x"), className="filtros")


def layout_eda():
    return dbc.Row([
        dbc.Col(panel_filtros(), lg=3),
        dbc.Col(html.Div([
            html.Div(id="eda-kpis", className="mt-3 mb-2"),

            # --- 3.1 Variable objetivo --------------------------------------------------------------
            seccion("Variable objetivo", "¿Cuántos adultos reportan dificultad para recordar o concentrarse y cómo se comporta en el tiempo?"),
            dbc.Row([
                dbc.Col(tarjeta(None, grafica("g-niveles"), "Los cuatro niveles de la pregunta original se reducen a dos (ninguna / alguna, mucha o no puede) porque los niveles altos son demasiado escasos para modelarse por separado."), lg=7),
                dbc.Col(tarjeta(None, grafica("g-trimestre"), "Barras de error: IC 95 % de Wilson. Si todos los intervalos cruzan la línea punteada, no hay deriva en el tiempo."), lg=5),
            ], className="g-3"),

            # --- 3.2 Edad ---------------------------------------------------------------------------
            seccion("Edad", "La edad es la única predictora numérica y tiene una relación particular con el objetivo."),
            dbc.Row([
                dbc.Col(tarjeta(None, html.Div([
                    dcc.RadioItems(id="e-modo", options=[{"label": " Conteo apilado", "value": "conteo"}, {"label": " Densidad por clase", "value": "densidad"}],
                                   value="conteo", inline=True, inputStyle={"marginLeft": "12px"}, style={"fontSize": "13px"}),
                    grafica("g-hist-edad")]), "El pico en 85 años es artificial: la edad se trunca en 85 («85 o más») para proteger la confidencialidad."), lg=6),
                dbc.Col(tarjeta(None, html.Div([
                    dcc.Dropdown(id="e-split", options=[{"label": "Sin desagregar", "value": "ninguno"}, {"label": "Por sexo", "value": "sexo"},
                                                        {"label": "Por antecedente de ACV", "value": "acv"}, {"label": "Por diagnóstico de depresión", "value": "depresion_dx"},
                                                        {"label": "Por hipertensión", "value": "hipertension"}, {"label": "Por año", "value": "anio"}],
                                 value="ninguno", clearable=False, style={"maxWidth": "300px", "fontSize": "13px"}),
                    grafica("g-prev-edad")]), "Se omiten los grupos con menos de 30 adultos. La banda es el IC 95 %; la forma de U exige modelar la edad con splines."), lg=6),
            ], className="g-3"),

            # --- 3.3 Factores de riesgo -------------------------------------------------------------
            seccion("Factores de riesgo", "Prevalencia de dificultad cognitiva en cada categoría y fuerza de asociación de cada factor."),
            dbc.Row([
                dbc.Col(tarjeta(None, html.Div([
                    html.Div("Variable a explorar", className="ctrl-et"),
                    dcc.Dropdown(id="e-var", options=OPC_VARIABLES, value="frecuencia_depresion", clearable=False, style={"maxWidth": "360px", "fontSize": "13px"}),
                    grafica("g-prev-cat")]), "Rojo: prevalencia por encima de la global del filtro; azul: por debajo. Las barras de error son el IC 95 %."), lg=6),
                dbc.Col(tarjeta(None, grafica("g-cramer"),
                                "V de Cramér: 0,1 asociación pequeña y 0,3 moderada. Con más de 40 000 adultos casi todo es «significativo», por eso se mira el tamaño del efecto."), lg=6),
            ], className="g-3"),
            tarjeta("Cruce de dos factores", html.Div([
                dbc.Row([dbc.Col([html.Div("Eje horizontal", className="ctrl-et"), dcc.Dropdown(id="e-hx", options=OPC_CRUCE, value="grupo_edad_amplio", clearable=False, style={"fontSize": "13px"})], md=4),
                         dbc.Col([html.Div("Eje vertical", className="ctrl-et"), dcc.Dropdown(id="e-hy", options=OPC_CRUCE, value="frecuencia_depresion", clearable=False, style={"fontSize": "13px"})], md=4)]),
                grafica("g-heat")]), "Cada celda muestra la prevalencia en ese cruce; se ocultan las celdas con menos de 30 adultos. Sirve para ver cómo se combinan dos factores."),

            # --- 3.4 ACV ----------------------------------------------------------------------------
            seccion("Foco en la exposición de interés: el ACV", "El antecedente de accidente cerebrovascular se asocia con la edad, y la edad con la dificultad cognitiva: hay que separar ambos efectos."),
            dbc.Row([
                dbc.Col(tarjeta(None, grafica("g-acv-edad"), "En los adultos jóvenes el ACV multiplica la prevalencia; en los mayores la base ya es alta y el ACV añade relativamente menos."), lg=6),
                dbc.Col(tarjeta(None, grafica("g-or-acv"), "El OR de Mantel-Haenszel combina los estratos de edad. Que los estratos difieran sugiere una interacción ACV × edad."), lg=6),
            ], className="g-3"),

            # --- 3.5 Estructura multivariada --------------------------------------------------------
            seccion("Relaciones entre variables y estructura de los datos", "Qué información aporta cada variable, cuánto se solapan entre sí y si los adultos forman grupos."),
            dbc.Row([
                dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_info_mutua(A), config=CONFIG_GRAFICA), "Captura relaciones no lineales como la de la edad, que la correlación simple no detecta. Calculada con todo el entrenamiento (no depende de los filtros)."), lg=6),
                dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_cramer_heatmap(A), config=CONFIG_GRAFICA), "Ningún par supera 0,6; la pareja más asociada mide el mismo constructo (diagnóstico y síntomas de depresión)."), lg=6),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(tarjeta(None, html.Div([
                    html.Div("Colorear por", className="ctrl-et"),
                    dcc.Dropdown(id="e-pca-color", options=list(F.COLOR_PCA), value="Dificultad cognitiva", clearable=False, style={"maxWidth": "300px", "fontSize": "13px"}),
                    grafica("g-pca")]), "Muestra de 6 000 adultos; responde a los filtros. Las clases se solapan mucho: no hay una separación lineal simple."), lg=7),
                dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_scree(A), config=CONFIG_GRAFICA), "La varianza está muy repartida, señal de que las predictoras son poco redundantes."), lg=5),
            ], className="g-3"),
            dbc.Row([
                dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_clusters(A), config=CONFIG_GRAFICA),
                                "K-means sobre las componentes que explican el 80 %. La silueta es baja, así que se lee como descripción de perfiles y no como grupos reales. Pase el cursor para ver el perfil de cada grupo."), lg=12),
            ]),

            # --- 3.6 Calidad de los datos y fuga ---------------------------------------------------
            seccion("Calidad de los datos y fuga de información", "Faltantes, representatividad y la auditoría que decidió qué variables pueden entrar al modelo."),
            dbc.Row([
                dbc.Col(tarjeta(None, grafica("g-faltantes"), "Ninguna predictora supera el 3 % de faltantes. El IMC numérico (8,5 %) falta por supresión de valores extremos (MNAR), por eso se usa la categoría de IMC."), lg=6),
                dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_sesgo(A), config=CONFIG_GRAFICA), "La muestra sobrerrepresenta a mayores de 65 años y a personas con estudios universitarios. El modelo se entrena sin pesos muestrales."), lg=6),
            ], className="g-3"),
            tarjeta(None, dcc.Graph(figure=F.fig_auditoria_fuga(A), config=CONFIG_GRAFICA),
                    "Una variable aislada no debería acercarse a 0,9. La pregunta de seguimiento de frecuencia alcanza 0,93 porque contiene al objetivo; «discapacidad» se construye con la propia pregunta objetivo y «demencia» es un proxy clínico; las tres se descartan."),
        ]), lg=9),
    ], className="g-3")


# ---------------------------------------------------------------------------
# 4. PESTAÑA 3 · MODELOS BASE
# ---------------------------------------------------------------------------
UM_LR = A["umbral"]["Regresión logística"]["umbral"]
PRESETS = {
    "bajo": dict(edad=35, sexo="Hombre", raza_etnia="Blanco no hispano", educacion="Posgrado", hipertension="No", colesterol_alto="No", diabetes="No",
                 acv="No", depresion_dx="No", frecuencia_depresion="Nunca", dificultad_auditiva="Ninguna", dificultad_visual="Ninguna",
                 tabaquismo="Nunca fumó", imc_categoria="Normal"),
    "medio": dict(edad=55, sexo="Mujer", raza_etnia="Hispano", educacion="Secundaria o GED", hipertension="Sí", colesterol_alto="No", diabetes="No",
                  acv="No", depresion_dx="No", frecuencia_depresion="Pocas veces al año", dificultad_auditiva="Ninguna", dificultad_visual="Alguna",
                  tabaquismo="Exfumador", imc_categoria="Sobrepeso"),
    "alto": dict(edad=62, sexo="Mujer", raza_etnia="Hispano", educacion="Menos que secundaria", hipertension="Sí", colesterol_alto="Sí", diabetes="Sí",
                 acv="Sí", depresion_dx="Sí", frecuencia_depresion="Semanal", dificultad_auditiva="Alguna", dificultad_visual="Alguna",
                 tabaquismo="Exfumador", imc_categoria="Obesidad"),
}
ORDEN_SIM = ["edad", "sexo", "raza_etnia", "educacion", "hipertension", "colesterol_alto", "diabetes", "acv", "depresion_dx", "frecuencia_depresion",
             "dificultad_auditiva", "dificultad_visual", "tabaquismo", "imc_categoria"]


def control_sim(v, pref="sim"):
    if v == "edad":
        return dbc.Col([html.Div(ETIQUETAS[v], className="ctrl-et"),
                        dcc.Slider(id=f"{pref}-{v}", min=18, max=85, step=1, value=PRESETS["medio"][v], marks={18: "18", 40: "40", 60: "60", 85: "85+"},
                                   tooltip={"placement": "bottom", "always_visible": True})], md=12, className="mb-3")
    return dbc.Col([html.Div(ETIQUETAS[v], className="ctrl-et"),
                    dcc.Dropdown(id=f"{pref}-{v}", options=CATS[v], value=PRESETS["medio"][v], clearable=False, style={"fontSize": "13px"})], md=6, className="mb-2")


def layout_modelos():
    return html.Div([
        seccion("Modelos base", "La regresión logística es el modelo principal (como en el capítulo 4 del libro); la SVM lineal se incluye como alternativa. "
                                "Ambos usan el mismo preprocesamiento y el mismo conjunto de prueba, reservado desde el inicio."),
        # --- 4.1 Controles -----------------------------------------------------------------------------
        html.Div(dbc.Row([
            dbc.Col([html.Div("Modelo", className="ctrl-et"),
                     dcc.RadioItems(id="m-modelo", options=[{"label": " Regresión logística", "value": "Regresión logística"}, {"label": " SVM lineal", "value": "SVM lineal"}],
                                    value="Regresión logística", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "4px"})], md=3),
            dbc.Col([html.Div("Datos evaluados", className="ctrl-et"),
                     dcc.RadioItems(id="m-conjunto", options=[{"label": " Prueba (reservado)", "value": "prueba"}, {"label": " Entrenamiento (validación cruzada)", "value": "entrenamiento"}],
                                    value="prueba", inputStyle={"marginRight": "6px"}, labelStyle={"display": "block", "marginBottom": "4px"})], md=3),
            dbc.Col([html.Div("Umbral de decisión", className="ctrl-et"),
                     dcc.Slider(id="m-umbral", min=0.05, max=0.7, step=0.005, value=round(UM_LR, 3), marks={0.1: "0,1", 0.25: "0,25", 0.4: "0,4", 0.55: "0,55", 0.7: "0,7"},
                                tooltip={"placement": "bottom", "always_visible": True}),
                     html.Button("Usar el umbral óptimo (máx. F1 fuera de partición)", id="m-opt", className="btn-dcs sec chico", style={"marginTop": "14px"})], md=6),
        ]), className="ctrl-fila"),
        html.Div(id="m-kpis", className="mb-3"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("m-conf"), "Cada fila suma 100 %: abajo a la derecha, la proporción de casos reales que el modelo detecta (recall); arriba a la derecha, las falsas alarmas."), lg=4),
            dbc.Col(tarjeta(None, grafica("m-rocpr"), "El rombo marca el punto de operación con el umbral elegido. La línea punteada de la derecha es la prevalencia: el desempeño de un modelo sin información."), lg=8),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("m-hist"), "Las clases se solapan bastante, lo que explica un desempeño moderado. Mover el umbral cambia cuántos casos se capturan y cuántas falsas alarmas se aceptan."), lg=6),
            dbc.Col(tarjeta(None, grafica("m-barrido"), "Calculado solo con predicciones fuera de partición del entrenamiento: el conjunto de prueba no interviene en la elección del umbral."), lg=6),
        ], className="g-3"),

        # --- 4.2 Comparación y calibración ----------------------------------------------------------
        seccion("Comparación con la línea base y calibración"),
        dbc.Row([
            dbc.Col(tarjeta(None, html.Div([
                html.Div("Métrica", className="ctrl-et"),
                dcc.Dropdown(id="m-metrica", options=["PR-AUC", "ROC-AUC", "Recall", "Precisión", "F1", "Balanced acc.", "Brier", "Accuracy"], value="PR-AUC",
                             clearable=False, style={"maxWidth": "240px", "fontSize": "13px"}),
                grafica("m-comp")]), "Intervalos de confianza por bootstrap (1 000 remuestreos del conjunto de prueba). Obsérvese la métrica Accuracy: el modelo trivial «gana» sin detectar un solo caso."), lg=7),
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_calibracion(A, F.MODELOS), config=CONFIG_GRAFICA),
                            "Si la curva sigue la diagonal, la probabilidad predicha se puede leer como riesgo real. Ambos modelos están bien calibrados."), lg=5),
        ], className="g-3"),
        tarjeta("Métricas en el conjunto de prueba · valor [IC 95 % bootstrap]", dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in F.tabla_metricas(A).columns], data=F.tabla_metricas(A).to_dict("records"),
            **estilo_tabla(style_data_conditional=[{"if": {"row_index": 2}, "backgroundColor": "#FBEDE6", "fontWeight": "600"},
                                                  {"if": {"row_index": 3}, "backgroundColor": "#E8F1F2", "fontWeight": "600"},
                                                  {"if": {"row_index": 0}, "color": "#8A7B6E"}, {"if": {"row_index": 1}, "color": "#8A7B6E"}])),
                "Recall, precisión, F1 y balanced accuracy se calculan con el umbral óptimo de cada modelo, fijado en el entrenamiento."),

        # --- 4.3 Entrenamiento ----------------------------------------------------------------------
        seccion("Entrenamiento: hiperparámetros y aprendizaje"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("m-grid"), "Cada punto es la media de una validación cruzada de 5 particiones. Las líneas punteadas (entrenamiento) casi tocan las sólidas (validación): no hay sobreajuste."), lg=6),
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_curva_aprendizaje(A), config=CONFIG_GRAFICA),
                            "La brecha se cierra y la curva se aplana: el modelo está limitado por sesgo, no por falta de datos. Más observaciones del mismo tipo casi no lo mejorarían."), lg=6),
        ], className="g-3"),
        tarjeta("Resultados de la búsqueda de hiperparámetros", html.Div(id="m-grid-tabla"), "Se elige el modelo sencillo cuando varias combinaciones empatan en PR-AUC."),

        # --- 4.4 Interpretación --------------------------------------------------------------------
        seccion("Interpretación: ¿qué factores pesan?", "Odds ratios de la regresión logística, ajustados por todas las demás variables."),
        dbc.Row([
            dbc.Col(tarjeta(None, html.Div([
                dbc.Row([dbc.Col([html.Div("Bloque de factores", className="ctrl-et"),
                                  dcc.Dropdown(id="m-or-bloque", options=["Todos"] + list(F.BLOQUES), value="Todos", clearable=False, style={"fontSize": "13px"})], md=6),
                         dbc.Col([html.Div("Mostrar", className="ctrl-et"),
                                  dcc.Checklist(id="m-or-sig", options=[{"label": " Solo significativos (Holm < 0,05)", "value": "si"}], value=[], style={"fontSize": "13px", "marginTop": "6px"})], md=6)]),
                grafica("m-or")]), "Un OR de 2 significa que las odds de reportar dificultad se duplican frente a la categoría de referencia. Los coeficientes penalizados y los no penalizados prácticamente coinciden (correlación 0,998), por lo que los IC son válidos."), lg=7),
            dbc.Col(tarjeta(None, dcc.Graph(figure=F.fig_dep_edad(A), config=CONFIG_GRAFICA),
                            "Probabilidad media predicha al fijar la edad y dejar el resto como está. Reproduce la U del EDA; en edades altas queda bajo la prevalencia observada porque el modelo atribuye parte del riesgo a otros factores propios de esa edad."), lg=5),
        ], className="g-3"),

        # --- 4.5 Diagnóstico por subgrupos --------------------------------------------------------
        seccion("Diagnóstico por subgrupos", "¿El modelo funciona igual para todos? Un residuo medio lejos de cero señala información que el modelo no está capturando."),
        tarjeta(None, html.Div([
            dbc.Row([dbc.Col([html.Div("Subgrupo", className="ctrl-et"),
                              dcc.Dropdown(id="m-sub-var", options=[{"label": ETIQUETAS.get(v, v.replace("_", " ").capitalize()), "value": v}
                                                                    for v in ["anio", "trimestre", "sexo", "grupo_edad", "raza_etnia", "educacion", "acv", "depresion_dx",
                                                                              "frecuencia_depresion", "dificultad_auditiva", "hipertension"]],
                                           value="grupo_edad", clearable=False, style={"fontSize": "13px"})], md=4)]),
            grafica("m-sub")]), "Punto rojo: el residuo medio se aleja de cero de forma significativa. En el desplegable de «Modelo» de arriba se elige cuál modelo se evalúa. El ROC-AUC de cada subgrupo aparece al pasar el cursor."),

        # --- 4.6 Simulador -------------------------------------------------------------------------
        seccion("Simulador de riesgo", "Combine factores y vea cómo responde la regresión logística. Es una demostración del modelo base, no una herramienta diagnóstica."),
        dbc.Row([
            dbc.Col(tarjeta("Perfil del adulto", html.Div([
                html.Div([html.Button("Perfil de bajo riesgo", id="sim-b", className="btn-dcs sec chico"), html.Button("Perfil intermedio", id="sim-m", className="btn-dcs sec chico"),
                          html.Button("Perfil de alto riesgo", id="sim-a", className="btn-dcs sec chico")], style={"marginBottom": "14px"}),
                dbc.Row([control_sim(v) for v in ORDEN_SIM]),
            ])), lg=5),
            dbc.Col(tarjeta(None, html.Div([
                dbc.Row([dbc.Col(grafica("sim-gauge", 260), md=5), dbc.Col(html.Div(id="sim-texto", className="resultado-sim", style={"paddingTop": "22px"}), md=7)]),
                grafica("sim-contrib")]), None), lg=7),
        ], className="g-3"),
        html.Div("Aviso: el modelo se entrenó con datos de EE. UU. (NHIS 2022-2023), de corte transversal y autorreportados. Estima la probabilidad de reportar dificultad cognitiva, no de tener demencia, y no sustituye una evaluación clínica.", className="aviso"),
    ])


# ---------------------------------------------------------------------------
# 4b. PESTAÑAS DEL BENCHMARK (benchmark, optimización, resultados finales, robustez)
# ---------------------------------------------------------------------------
def _dd(id_, opciones, valor):
    return dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False)


def _tabla(df, id_=None, tam=12, **extra):
    cols = [{"name": c, "id": c} for c in df.columns]
    kw = {"id": id_} if id_ else {}
    return dash_table.DataTable(columns=cols, data=df.to_dict("records"), page_size=tam, sort_action="native", **kw, **estilo_tabla(**extra))


def layout_benchmark():
    m = E.E3["maestra"]
    mejor = m.sort_values("PR_AUC_media", ascending=False).iloc[0]
    ebm = E.E3["ebm"].iloc[0]
    return html.Div([
        seccion("Benchmark de 112 combinaciones",
                "7 modelos × 4 técnicas de balanceo × 4 optimizadores de hiperparámetros, con validación cruzada anidada (5 folds externos, 3 internos) y preprocesamiento dentro de cada fold. "
                "La tabla muestra 108 combinaciones: KNN no admite pesos de clase, así que las 4 de KNN con class_weight no aplican. El EBM y el modelo original (aditivo penalizado) van aparte, en la pestaña 7."),
        dbc.Row([
            dbc.Col(kpi("108", "combinaciones evaluadas", "540 tareas, 5.529 evaluaciones internas, 0 errores"), md=3, className="mb-3"),
            dbc.Col(kpi(n_es(mejor["PR_AUC_media"], 3), "mejor PR-AUC externo", f"{E.NOMBRES[mejor['modelo']]} · {mejor['balanceo']} · {mejor['optimizador']}", "teal"), md=3, className="mb-3"),
            dbc.Col(kpi("≈ 0,80", "techo de ROC-AUC", "igual en todos los modelos buenos", "dorado"), md=3, className="mb-3"),
            dbc.Col(kpi("sin ganador", "entre logística, SVM y XGBoost", "DeLong con Holm no separa a ninguno", "osc"), md=3, className="mb-3"),
        ]),
        html.Div(dbc.Row([
            dbc.Col([html.Div("Métrica", className="ctrl-et"), _dd("b-metrica", [{"label": v, "value": k} for k, v in E.METRICAS.items()], "PR_AUC_media")], md=6),
            dbc.Col([html.Div("Optimizador", className="ctrl-et"), _dd("b-opt", [{"label": "Promedio de los cuatro", "value": "todos"}] + [{"label": o, "value": o} for o in E.OPT_ORDEN], "todos")], md=6),
        ]), className="ctrl-fila"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("b-mapa"), "Balancear (SMOTE, ADASYN) no mejora la discriminación. class_weight y sin balanceo quedan arriba."), lg=7),
            dbc.Col(tarjeta(None, html.Div([html.Div("Prueba de Friedman por factor", className="ctrl-et"),
                                            _dd("b-factor", [{"label": f.capitalize(), "value": f} for f in ("modelo", "balanceo", "optimizador")], "modelo"),
                                            grafica("b-rangos")]),
                            "La línea punteada es la diferencia crítica de Nemenyi. Con las 108 juntas no tiene potencia (cinco folds), por eso se analiza por factor."), lg=5),
        ], className="g-3"),
        tarjeta("Diagrama de diferencia crítica (Nemenyi) por factor", html.Img(src="/assets/e3/cd_diagrama.png", style={"width": "100%", "maxWidth": "860px", "display": "block", "margin": "0 auto"}),
                "Rango medio de cada nivel (1 es el mejor, a la derecha). Las barras gruesas unen a los niveles que Nemenyi no distingue."),
        tarjeta("Tabla maestra (filtrable y ordenable)", html.Div([
            dbc.Row([dbc.Col([html.Div("Modelo", className="ctrl-et"), dcc.Dropdown(id="b-f-modelo", options=[{"label": E.NOMBRES[k], "value": k} for k in E.NOMBRES if k != "EBM"], multi=True, placeholder="Todos")], md=4),
                     dbc.Col([html.Div("Balanceo", className="ctrl-et"), dcc.Dropdown(id="b-f-bal", options=[{"label": k, "value": k} for k in E.BAL_ORDEN], multi=True, placeholder="Todos")], md=4),
                     dbc.Col([html.Div("Optimizador", className="ctrl-et"), dcc.Dropdown(id="b-f-opt", options=[{"label": k, "value": k} for k in E.OPT_ORDEN], multi=True, placeholder="Todos")], md=4)], className="mb-3"),
            html.Div(id="b-tabla")]),
            "Incluye métricas externas con su desviación, evaluaciones, hiperparámetros finales de cada fold, semilla global (42) y semilla por fold (42 + fold), y tiempo total."),
        dbc.Row([
            dbc.Col(tarjeta("DeLong contra la mejor combinación (Holm)", _tabla(E.E3["delong"].sort_values("p_Holm").head(20).round(4), tam=10),
                            "Solo XGBoost queda detrás de la mejor, por 0,002 a 0,003 de AUC, una diferencia sin relevancia práctica."), lg=6),
            dbc.Col(tarjeta("EBM frente a las referencias", _tabla(E.E3["ebm"].round(4), tam=10),
                            "El EBM (candidato interpretable de la librería interpret) iguala a los mejores y no los supera. Su ventaja es que cada predicción se puede leer directamente."), lg=6),
        ], className="g-3"),
        html.Div("Las 28 combinaciones de regresión del enunciado no aplican porque la variable objetivo es binaria. Ridge y Lasso quedan cubiertas por la regresión logística con penalización L2 y L1.", className="aviso"),
    ])


def layout_optimizacion():
    return html.Div([
        seccion("Optimización y costo computacional",
                "Grid, Random, bayesiana (Optuna) y genética (DEAP) con un presupuesto de unas 12 evaluaciones sobre una submuestra de 8.000 filas y PR-AUC como criterio interno."),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("o-conv"), "La genética hizo 6,9 evaluaciones contra 12 por su caché y población de 4, así que su desventaja es en parte de presupuesto."), lg=5),
            dbc.Col(tarjeta(None, grafica("o-pareto"), "Cada punto es una combinación. Más tiempo no significa mejor desempeño."), lg=7),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("o-comp"), "Lineales cerca de 1, SVM con kernel RBF cerca de 2 (n al cuadrado), XGBoost hist unas 8 veces más rápido que el método exacto."), lg=7),
            dbc.Col(tarjeta("Exponente empírico de ajuste", _tabla(E.E3["expo"], tam=12), "Pendiente del tiempo de ajuste contra n en escala log-log, de 2.000 a 32.000 filas."), lg=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(tarjeta("Diversidad genética por generación", html.Img(src="/assets/e3/diversidad_genetica.png", style={"width": "100%"}),
                            "Distancia media entre los individuos. Cae a la mitad en tres generaciones, señal de convergencia rápida con poblaciones de 4."), lg=5),
            dbc.Col(tarjeta("Sensibilidad a la semilla (5 semillas)", html.Div([_tabla(E.E3["sem_mod"], tam=4), html.Div(style={"height": "14px"}), _tabla(E.E3["sem_opt"], tam=4)]),
                            "Arriba, PR-AUC de XGBoost y Random Forest con 5 semillas del modelo. Abajo, mejor PR-AUC interno que encuentra cada optimizador con 5 semillas."), lg=7),
        ], className="g-3"),
        tarjeta("Complejidad teórica, estándar frente a optimizado (32.000 filas)", _tabla(E.E3["cx"], tam=12),
                "Tiempos y memoria medidos en un proceso limpio por configuración. PR-AUC sobre 4.000 filas apartadas del entrenamiento."),
        dbc.Row([
            dbc.Col(tarjeta("Costo total por modelo (segundos)", _tabla(E.E3["costo_m"].sort_values("t_total_s", ascending=False), tam=10)), lg=6),
            dbc.Col(tarjeta("Costo total por optimizador (segundos)", _tabla(E.E3["costo_o"].sort_values("t_total_s", ascending=False), tam=10)), lg=6),
        ], className="g-3"),
    ])


def layout_final():
    t = E.E3["lime_j"]
    return html.Div([
        seccion("Resultado final, calibración, interpretabilidad y robustez",
                "Los finalistas se reajustan con toda la partición de entrenamiento y se evalúan una sola vez en la prueba reservada. Los intervalos son bootstrap percentil (1.000 remuestras)."),
        dbc.Row([
            dbc.Col(tarjeta("Conjunto de prueba (intervalo 95 %)", _tabla(E.E3["test"], tam=6), "Coincide con el capítulo 4 del libro para la logística (PR-AUC 0,538, ROC-AUC 0,804)."), lg=7),
            dbc.Col(tarjeta("DeLong entre finalistas (Holm)", _tabla(E.E3["test_delong"].round(4), tam=8), "Ninguna comparación es significativa."), lg=5),
        ], className="g-3"),
        tarjeta("Accuracy, precision, recall, F1 y AUC de los finalistas (prueba)", html.Div([_tabla(E.E3["fin"], tam=8),
                html.Img(src="/assets/e3/confusion_roc_finalistas.png", style={"width": "100%", "marginTop": "14px"})]),
                "Con el umbral 0,5 y con el 0,245 elegido en el entrenamiento. El criterio principal de validación fue la PR-AUC, porque con prevalencia de 20,6 % el accuracy engaña."),
        tarjeta("Métricas de los ocho modelos (prueba)", html.Div([_tabla(E.E3["todos"], tam=8),
                html.Img(src="/assets/e3/confusion_roc_todos.png", style={"width": "100%", "marginTop": "14px"})]),
                "Random Forest, árbol, Naive Bayes y KNN se ajustaron con la misma receta sin balanceo y se evaluaron una sola vez. Es un reporte descriptivo: ninguna decisión se tomó con estos resultados."),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("f-calib"), "SMOTE, ADASYN y class_weight suben el recall pero el modelo deja de estimar la prevalencia real."), lg=6),
            dbc.Col(tarjeta(None, html.Img(src="/assets/e3/calibracion.png", style={"width": "100%"}), "Curvas de calibración de los finalistas en la prueba."), lg=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(tarjeta(None, grafica("f-imp"), "Frecuencia de síntomas depresivos, diagnóstico de depresión y edad lideran en los tres métodos."), lg=6),
            dbc.Col(tarjeta(None, html.Div([grafica("f-lime"),
                html.P(f"Spearman global SHAP y LIME: {n_es(t['spearman_global_SHAP_LIME'], 2)} · mediana local: {n_es(t['spearman_local_mediana'], 2)} · coincidencia de las 3 principales: {n_es(t['coincidencia_top3_variables'], 2)} · estabilidad de LIME (Jaccard top 5): {n_es(t['estabilidad_LIME_jaccard_top5'], 2)}", className="nota")]),
                            "SHAP es exacto para árboles; LIME aproxima con 3.000 perturbaciones por instancia. Por eso se usa SHAP como base."), lg=6),
        ], className="g-3"),
        tarjeta("Recalibración de los modelos balanceados (Platt e isotónica)", _tabla(E.E3["recal"], tam=12),
                "Evaluado en el 30 % del entrenamiento. La recalibración baja el ECE de 0,14 a 0,22 hasta menos de 0,02 sin perder PR-AUC."),
        dbc.Row([
            dbc.Col(tarjeta("SHAP local, acierto", html.Img(src="/assets/e3/shap_waterfall_acierto.png", style={"width": "100%"}), "Probabilidad 0,90, empujan la depresión, las pérdidas sensoriales y el ACV."), lg=6),
            dbc.Col(tarjeta("SHAP local, error", html.Img(src="/assets/e3/shap_waterfall_error.png", style={"width": "100%"}), "Probabilidad 0,046, ninguna variable empuja hacia arriba."), lg=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(tarjeta(None, html.Img(src="/assets/e3/shap_xgboost.png", style={"width": "100%"}), "SHAP de XGBoost."), lg=6),
            dbc.Col(tarjeta(None, html.Img(src="/assets/e3/ebm_global.png", style={"width": "100%"}), "Funciones de forma del EBM. La edad reproduce la U del EDA."), lg=6),
        ], className="g-3"),
        seccion("Robustez", "Pesos muestrales, umbral, datos faltantes, ruido en las entradas, subgrupos y validación temporal de 2022 a 2023."),
        html.Div(dbc.Row([dbc.Col([html.Div("Análisis", className="ctrl-et"),
                                   _dd("r-an", [{"label": l, "value": v} for l, v in [("Pesos muestrales", "pesos"), ("Umbral de decisión", "umbral"), ("Datos faltantes", "faltantes"),
                                                                                   ("Ruido en las entradas", "perturbacion"), ("Subgrupos (logística)", "subgrupos"), ("Temporal 2022 a 2023", "temporal")]], "pesos")], md=6)]), className="ctrl-fila"),
        tarjeta(None, html.Div(id="r-tabla"), "Los pesos muestrales no reemplazan un análisis con diseño complejo de la NHIS, que queda declarado como limitación."),
    ])


# ---------------------------------------------------------------------------
# 4c. PESTAÑA 7 · MODELO ORIGINAL (aditivo penalizado por rondas, programado por el equipo)
# ---------------------------------------------------------------------------
def layout_original():
    d = OT.cargar()
    if not d["ok"]:
        return html.Div(dbc.Alert(f"No se pudieron leer los resultados del modelo original ({d['error']}).", color="warning"))
    v = d["veredicto"]
    r3 = v["R3_cumplida"]
    te = d["test"].set_index("modelo")
    lider = d["lider"]
    pareadas = d["pareadas"].copy()
    pareadas["referencia"] = pareadas["referencia"].map(lambda k: OT.NOMBRES.get(k, k))
    pareadas = pareadas[["referencia", "n_pares", "dif_media_PR_AUC", "IC95_BCa_inf", "IC95_BCa_sup", "wilcoxon_p", "wilcoxon_p_holm", "nadeau_bengio_p", "nadeau_bengio_p_holm", "cliffs_delta", "cohen_dz"]].round(4)
    pareadas.columns = ["Referencia", "Pares", "Dif. PR-AUC", "IC BCa inf.", "IC BCa sup.", "Wilcoxon p crudo", "Wilcoxon p Holm", "Nadeau-Bengio p crudo", "Nadeau-Bengio p Holm", "Cliff δ", "d de Cohen"]
    prueba = d["prueba"].copy(); prueba["referencia"] = prueba["referencia"].map(lambda k: OT.NOMBRES.get(k, k))
    prueba = prueba.round(4)
    return html.Div([
        seccion("Modelo original: aditivo penalizado por rondas",
                "Programado por el equipo solo con numpy (clase AditivoPenalizado, compatible con scikit-learn). La probabilidad es la suma de una curva por variable en la escala logit, "
                "y las curvas se aprenden por rondas con una actualización derivada de una cota cuadrática que garantiza que el objetivo nunca aumente."),
        dbc.Row([
            dbc.Col(kpi("Sí", "R2 · implementación propia", "sin importar ningún modelo de otra librería", "teal"), md=3, className="mb-3"),
            dbc.Col(kpi("Parcial", "R1 · novedad", "de aplicación y de implementación, no de mecanismo", "dorado"), md=3, className="mb-3"),
            dbc.Col(kpi("No" if not r3 else "Sí", "R3 · superioridad significativa", f"PR-AUC {n_es(v['PR_AUC_media_original'], 4)} frente a {n_es(max(v['PR_AUC_media_por_referencia'].values()), 4)} del mejor benchmark", "osc"), md=3, className="mb-3"),
            dbc.Col(kpi("Tope 4,0", "según la rúbrica del entregable", "cumple R2 y no cumple R3", ""), md=3, className="mb-3"),
        ]),
        html.Div("Veredicto honesto. El modelo original iguala a los mejores benchmarks y no los supera de forma estadísticamente significativa. Con 14 predictoras el rendimiento está en el techo de la información disponible, "
                 "así que no hay margen para que ningún modelo gane. Su aporte es una implementación propia, verificada con pruebas unitarias, que entrega curvas legibles y valores SHAP exactos.", className="aviso"),

        seccion("Tabla de líderes", "Mejor combinación de cada modelo base y los dos modelos aditivos, con 25 estimaciones externas pareadas (5 repeticiones × 5 pliegues, mismas particiones y semillas). Se puede ordenar y filtrar."),
        tarjeta(None, dash_table.DataTable(id="ori-lider", columns=[{"name": c, "id": c} for c in lider.columns], data=lider.to_dict("records"), page_size=12,
                                           sort_action="native", filter_action="native", **estilo_tabla()),
                "Media ± desviación estándar de la PR-AUC del bucle externo, con el intervalo bootstrap BCa de la media, y tiempos medios de entrenamiento e inferencia. El EBM tiene menos estimaciones si alguna repetición falta."),

        seccion("Predicción en vivo", "Mismo formulario de 14 variables, con la probabilidad y la explicación exacta del modelo original (valores SHAP en logit). Se carga el modelo ya entrenado, nunca se reentrena."),
        dbc.Row([
            dbc.Col(tarjeta("Perfil del adulto", html.Div([
                html.Div([html.Button("Perfil de bajo riesgo", id="ori-b", className="btn-dcs sec chico"), html.Button("Perfil intermedio", id="ori-m", className="btn-dcs sec chico"),
                          html.Button("Perfil de alto riesgo", id="ori-a", className="btn-dcs sec chico")], style={"marginBottom": "14px"}),
                dbc.Row([control_sim(x, "ori") for x in ORDEN_SIM]),
            ])), lg=5),
            dbc.Col(tarjeta(None, html.Div([
                html.Div(id="ori-alerta"),
                dbc.Row([dbc.Col(grafica("ori-gauge", 260), md=5), dbc.Col(html.Div(id="ori-texto", className="resultado-sim", style={"paddingTop": "22px"}), md=7)]),
                grafica("ori-shap")]), "Cada barra es el aporte de esa variable al logit respecto del efecto promedio. Como el modelo es aditivo, estos valores son SHAP exactos."), lg=7),
        ], className="g-3"),

        seccion("Descripción, pseudocódigo y diferencias con la literatura"),
        dbc.Row([
            dbc.Col(tarjeta("Pseudocódigo", html.Pre(OT.PSEUDOCODIGO, style={"fontSize": "12px", "whiteSpace": "pre-wrap", "background": "#FBF7F0", "padding": "12px", "borderRadius": "8px"}),
                            "El objetivo es la pérdida logística más una penalización ridge (λ) y una penalización de suavizado entre niveles vecinos de variables ordenadas (κ). Complejidad O(R p n), predicción O(p) por fila."), lg=7),
            dbc.Col(tarjeta("Qué es propio y qué no", html.Ul([
                html.Li("No es nuevo el modelo aditivo (Hastie y Tibshirani, 1990), ni la penalización de diferencias (Eilers y Marx, 1996), ni el boosting por coordenadas (Friedman, 2001)."),
                html.Li("Es propia la implementación en numpy, la regla de actualización derivada de la cota cuadrática, la propiedad de no aumento del objetivo y su verificación con pruebas unitarias."),
                html.Li("Frente al EBM (Lou y colaboradores, 2013) cambia el paso, que es un sistema lineal por variable y no árboles pequeños, y no incluye interacciones."),
                html.Li("No encontramos estudios que lo apliquen a DCS con la NHIS en PubMed, Scopus AI y un buscador web. Es una afirmación sobre esas búsquedas, no una prueba de que no exista."),
            ], style={"fontSize": "13.5px"}), None), lg=5),
        ], className="g-3"),

        seccion("Curvas aprendidas, convergencia y complejidad"),
        tarjeta(None, html.Img(src="/assets/e3/original_curvas.png", style={"width": "100%"}), "Una curva por variable. La edad reproduce la U del EDA, la depresión tiene un gradiente claro y la educación es protectora."),
        tarjeta(None, html.Img(src="/assets/e3/original_convergencia_complejidad.png", style={"width": "100%"}),
                "Izquierda: el objetivo baja de forma monótona. Centro: curva de aprendizaje en el pliegue 0. Derecha: el tiempo crece cerca de lo lineal con n y con p."),
        tarjeta("Estudio de ablación (pliegues externos, mismos hiperparámetros)", _tabla(d["ablacion"], tam=8),
                "Quitar el suavizado, el ridge o el orden de las categóricas casi no cambia la PR-AUC. El suavizado reduce la rugosidad de la curva de la edad cerca de 7 %."),

        seccion("Verificación estadística de la superioridad", f"{v['pruebas_realizadas']} pruebas, con p ajustado por Holm, tamaños del efecto e intervalos BCa. Total de comparaciones: {v['comparaciones_totales']}."),
        tarjeta("Diferencia crítica (Friedman y Nemenyi), modelo original incluido", html.Img(src="/assets/e3/original_cd.png", style={"width": "100%", "maxWidth": "860px", "display": "block", "margin": "0 auto"}),
                "Las barras gruesas unen a los modelos que no se distinguen. El original queda empatado con la SVM, la logística y XGBoost."),
        tarjeta("Pruebas pareadas por pliegue (PR-AUC, original menos referencia)", _tabla(pareadas, tam=10),
                "Hipótesis nula, igual desempeño. Supuestos revisados, datos de corte transversal e independencia aproximada, por eso se usa la corrección de Nadeau y Bengio para la validación repetida. Cliff δ y d de Cohen son los tamaños del efecto."),
        tarjeta("Pruebas en el conjunto de prueba (DeLong y McNemar con umbral 0,245)", _tabla(prueba, tam=6),
                "DeLong compara el AUC con dependencia entre curvas. McNemar compara los aciertos de cada par con el mismo umbral. Ambas con Holm."),
        tarjeta("Desempeño en la prueba reservada", _tabla(d["test"].round(4), tam=6), "Una sola evaluación. El modelo original queda en la misma zona que los finalistas del segundo entregable."),
        html.Div("Si la diferencia no es significativa, se reporta tal cual y define el tope de 4,0. Esto no invalida el trabajo, según la rúbrica del entregable.", className="aviso"),
    ])


# ---------------------------------------------------------------------------
# 5. ENSAMBLADO DE LA APP Y CALLBACKS
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="DCS y factores de riesgo de demencia · Proyecto de Machine Learning",
                suppress_callback_exceptions=True)   # Bootstrap 5 va local en assets/ (funciona sin internet)
server = app.server   # necesario para el despliegue (Procfile → gunicorn)

app.layout = html.Div([
    # Encabezado
    html.Div(dbc.Container([
        html.Div("Proyecto de Machine Learning", className="eyebrow"),
        html.H1("Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia"),
        html.P("Dashboard analítico: contexto del problema, análisis exploratorio, modelos base benchmark de 112 combinaciones, optimización, interpretabilidad y robustez sobre la National Health Interview Survey (NHIS) 2022-2023.", className="lead"),
        html.Div([html.Span("Santiago Díaz", className="chip autor"), html.Span("Gina Huguet", className="chip autor"), html.Span("Leiry Mares", className="chip autor"),
                  html.Span("Maestría en Ingeniería Biomédica · Universidad del Norte", className="chip"), html.Span("56.129 adultos · 14 predictoras", className="chip")]),
    ], fluid="xl"), className="hero"),
    dbc.Container([
        dbc.Tabs([dbc.Tab(label="1 · Contexto del problema", tab_id="tab-contexto"),
                  dbc.Tab(label="2 · Análisis exploratorio (EDA)", tab_id="tab-eda"),
                  dbc.Tab(label="3 · Modelos base", tab_id="tab-modelos"),
                  dbc.Tab(label="4 · Benchmark", tab_id="tab-bench"),
                  dbc.Tab(label="5 · Optimización y costo", tab_id="tab-opt"),
                  dbc.Tab(label="6 · Resultado final, interpretación y robustez", tab_id="tab-final"),
                  dbc.Tab(label="7 · Modelo original", tab_id="tab-original")], id="tabs", active_tab="tab-contexto"),
        html.Div(id="contenido", className="pb-4"),
        html.Div("Fuente de los datos: NCHS, National Health Interview Survey 2022 y 2023, Sample Adult Interview (CDC). Semilla aleatoria 42 · partición 80/20 estratificada. "
                 "Las asociaciones son transversales y no implican causalidad.", className="pie"),
    ], fluid="xl"),
])


# ---- 5.1 Router de pestañas (las pestañas se dibujan al abrirlas para que Plotly calcule bien el tamaño) ----
@app.callback(Output("contenido", "children"), Input("tabs", "active_tab"))
def pintar_pestana(tab):
    if tab == "tab-eda":
        return layout_eda()
    if tab == "tab-modelos":
        return layout_modelos()
    if tab == "tab-bench":
        return layout_benchmark()
    if tab == "tab-opt":
        return layout_optimizacion()
    if tab == "tab-final":
        return layout_final()
    if tab == "tab-original":
        return layout_original()
    return layout_contexto()


# ---- 5.2 Pestaña 1: filtros de las tablas ----
@app.callback(Output("ctx-lancet-tabla", "data"), Input("ctx-lancet-filtro", "value"))
def filtrar_lancet(v):
    d = F.TABLA_LANCET if v == "Todos" else F.TABLA_LANCET[F.TABLA_LANCET["Disponibilidad"] == v]
    return d.to_dict("records")


@app.callback(Output("ctx-dic-tabla", "data"), Input("ctx-dic-rol", "value"))
def filtrar_diccionario(v):
    d = DICCIONARIO if v == "Todos" else DICCIONARIO[DICCIONARIO["rol"] == v]
    return d.to_dict("records")


# ---- 5.3 Pestaña 2: filtro global y callbacks de cada gráfica ----
FILTROS = [Input("f-anio", "value"), Input("f-sexo", "value"), Input("f-edad", "value"), Input("f-raza", "value"), Input("f-acv", "value")]


def mascara(df, anios, sexos, edad, razas, acv):
    """Aplica los cinco filtros del panel lateral a cualquier DataFrame que tenga esas columnas."""
    m = df["anio"].isin(anios or []) & (df["sexo"].isin(sexos or []) | df["sexo"].isna())
    m &= df["edad"].between(edad[0], edad[1]) | df["edad"].isna()
    if razas:
        m &= df["raza_etnia"].isin(razas)
    if acv in ("Sí", "No"):
        m &= df["acv"] == acv
    return m


def filtrado(*args):
    return DF[mascara(DF, *args)]


@app.callback(Output("f-anio", "value"), Output("f-sexo", "value"), Output("f-edad", "value"), Output("f-raza", "value"), Output("f-acv", "value"),
              Input("f-reset", "n_clicks"), prevent_initial_call=True)
def restablecer(_):
    return [2022, 2023], ["Hombre", "Mujer"], [18, 85], [], "Todos"


@app.callback(Output("f-descarga-datos", "data"), Input("f-descarga", "n_clicks"), State("f-anio", "value"), State("f-sexo", "value"), State("f-edad", "value"),
              State("f-raza", "value"), State("f-acv", "value"), prevent_initial_call=True)
def descargar(_, *filtros):
    cols = [c for c in TRAIN.columns]
    return dcc.send_data_frame(filtrado(*filtros)[cols].to_csv, "adultos_filtrados_nhis.csv", index=False)


@app.callback(Output("eda-kpis", "children"), *FILTROS)
def kpis(*f):
    k = F.kpis_eda(filtrado(*f))
    if k["n"] == 0:
        return html.Div("No hay adultos con la combinación de filtros elegida.", className="aviso")
    return dbc.Row([
        dbc.Col(kpi(n_es(k["n"]), "adultos en el filtro", f"{pct_es(k['n'] / len(DF))} del entrenamiento"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(n_es(k["casos"]), "con dificultad cognitiva", "casos", "osc"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(pct_es(k["prev"]), "prevalencia", f"IC 95 %: {pct_es(k['lo'])} – {pct_es(k['hi'])}", "teal"), md=3, xs=12, className="mb-2"),
        dbc.Col(kpi(f"{k['edad']:.0f} años", "edad mediana", f"{pct_es(k['mayores'], 0)} tiene 65 años o más", "dorado"), md=3, xs=6, className="mb-2"),
        dbc.Col(kpi(pct_es(k["mujeres"], 0), "mujeres", None, "ciruela"), md=2, xs=6, className="mb-2"),
    ], className="g-2")


@app.callback(Output("g-niveles", "figure"), *FILTROS)
def g_niveles(*f):
    return F.fig_niveles(filtrado(*f))


@app.callback(Output("g-trimestre", "figure"), *FILTROS)
def g_trimestre(*f):
    return F.fig_trimestre(filtrado(*f))


@app.callback(Output("g-hist-edad", "figure"), Input("e-modo", "value"), *FILTROS)
def g_hist(modo, *f):
    return F.fig_hist_edad(filtrado(*f), modo)


@app.callback(Output("g-prev-edad", "figure"), Input("e-split", "value"), *FILTROS)
def g_prev_edad(split, *f):
    return F.fig_prev_edad(filtrado(*f), None if split == "ninguno" else split)


@app.callback(Output("g-prev-cat", "figure"), Input("e-var", "value"), *FILTROS)
def g_prev_cat(var, *f):
    return F.fig_prev_categoria(filtrado(*f), var)   # categorías originales (sin agrupar raras)


@app.callback(Output("g-cramer", "figure"), *FILTROS)
def g_cramer(*f):
    return F.fig_cramer(filtrado(*f))


@app.callback(Output("g-heat", "figure"), Input("e-hx", "value"), Input("e-hy", "value"), *FILTROS)
def g_heat(vx, vy, *f):
    d = filtrado(*f).copy()
    d["anio"] = d["anio"].astype(str)
    return F.fig_heatmap_dos(d, vx, vy)


@app.callback(Output("g-acv-edad", "figure"), *FILTROS)
def g_acv_edad(*f):
    return F.fig_acv_edad(filtrado(*f))


@app.callback(Output("g-or-acv", "figure"), *FILTROS)
def g_or_acv(*f):
    return F.fig_or_acv(filtrado(*f))


@app.callback(Output("g-pca", "figure"), Input("e-pca-color", "value"), *FILTROS)
def g_pca(color, *f):
    return F.fig_pca(A, PUNTOS_PCA[mascara(PUNTOS_PCA, *f)], color)


@app.callback(Output("g-faltantes", "figure"), *FILTROS)
def g_faltantes(*f):
    return F.fig_faltantes(filtrado(*f))


# ---- 5.4 Pestaña 3: modelos ----
@app.callback(Output("m-umbral", "value"), Input("m-opt", "n_clicks"), Input("m-modelo", "value"))
def umbral_optimo(_, modelo):
    """Al cambiar de modelo o pulsar el botón, el control vuelve al umbral que maximiza el F1 fuera de partición."""
    return round(A["umbral"][modelo]["umbral"], 3)


@app.callback(Output("m-kpis", "children"), Output("m-conf", "figure"), Output("m-rocpr", "figure"), Output("m-hist", "figure"), Output("m-barrido", "figure"),
              Input("m-modelo", "value"), Input("m-conjunto", "value"), Input("m-umbral", "value"))
def panel_modelo(modelo, conjunto, umbral):
    y, p = F.datos_modelo(A, modelo, conjunto)
    m = F.metricas_umbral(y, p, umbral)
    otro = [x for x in F.MODELOS if x != modelo][0]
    umbrales = {modelo: umbral, otro: A["umbral"][otro]["umbral"]}
    base = y.mean()
    tarjetas = dbc.Row([
        dbc.Col(kpi(n_es(m["PR-AUC"], 3), "PR-AUC", f"{n_es(m['PR-AUC'] / base, 1)}× la línea base ({n_es(base, 3)})", "osc"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(n_es(m["ROC-AUC"], 3), "ROC-AUC", "independiente del umbral", "teal"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(pct_es(m["Recall"]), "Recall", f"{n_es(m['VP'])} de {n_es(m['VP'] + m['FN'])} casos"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(pct_es(m["Precisión"]), "Precisión", f"{n_es(m['VP'])} de {n_es(m['VP'] + m['FP'])} alertas", "dorado"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(n_es(m["F1"], 3), "F1", f"balanced acc. {n_es(m['Balanced acc.'], 3)}", "salvia"), md=2, xs=6, className="mb-2"),
        dbc.Col(kpi(n_es(m["Brier"], 3), "Brier", f"especificidad {pct_es(m['Especificidad'], 0)}", "ciruela"), md=2, xs=6, className="mb-2"),
    ], className="g-2")
    return (tarjetas, F.fig_confusion(y, p, umbral), F.fig_roc_pr(A, F.MODELOS, conjunto, umbrales),
            F.fig_hist_prob(y, p, umbral), F.fig_barrido_umbral(A, modelo, umbral))


@app.callback(Output("m-comp", "figure"), Input("m-metrica", "value"))
def comparacion(metrica):
    return F.fig_comparacion(A, metrica)


@app.callback(Output("m-grid", "figure"), Output("m-grid-tabla", "children"), Input("m-modelo", "value"))
def grid(modelo):
    t = (A["grid_lr"] if modelo == "Regresión logística" else A["grid_svm"]).copy()
    cols = [{"name": c, "id": c, "type": "numeric", "format": dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)} if t[c].dtype.kind == "f"
            else {"name": c, "id": c} for c in t.columns]
    tabla = dash_table.DataTable(columns=cols, data=t.to_dict("records"), page_size=10, sort_action="native",
                                 **estilo_tabla(style_data_conditional=[{"if": {"row_index": 0}, "backgroundColor": "#FBEDE6", "fontWeight": "700"}]))
    return F.fig_grid(A, modelo), tabla


@app.callback(Output("m-or", "figure"), Input("m-or-bloque", "value"), Input("m-or-sig", "value"))
def odds(bloque, sig):
    return F.fig_or(A, bloque, bool(sig))


@app.callback(Output("m-sub", "figure"), Input("m-modelo", "value"), Input("m-sub-var", "value"))
def subgrupos(modelo, var):
    return F.fig_subgrupos(A, modelo, var)


# ---- 5.5 Simulador de riesgo ----
@app.callback([Output(f"sim-{v}", "value") for v in ORDEN_SIM], Input("sim-b", "n_clicks"), Input("sim-m", "n_clicks"), Input("sim-a", "n_clicks"), prevent_initial_call=True)
def preajustes(*_):
    clave = {"sim-b": "bajo", "sim-m": "medio", "sim-a": "alto"}[ctx.triggered_id]
    return [PRESETS[clave][v] for v in ORDEN_SIM]


@app.callback(Output("sim-gauge", "figure"), Output("sim-texto", "children"), Output("sim-contrib", "figure"), [Input(f"sim-{v}", "value") for v in ORDEN_SIM])
def simulador(*valores):
    perfil = dict(zip(ORDEN_SIM, valores))
    if any(v is None for v in perfil.values()):
        return dash.no_update, dash.no_update, dash.no_update
    perfil["edad"] = float(perfil["edad"])
    contrib, prob = F.contribuciones(A, perfil)
    prev = A["prevalencia_tr"]
    supera = prob >= UM_LR
    texto = [html.P([html.B(f"{n_es(prob * 100, 1)} %"), f" de probabilidad estimada de reportar dificultad cognitiva, es decir {n_es(prob / prev, 1)} veces la prevalencia de la muestra ({pct_es(prev)})."]),
             html.P([f"Con el umbral óptimo del modelo ({n_es(UM_LR, 3)}), este perfil ", html.B("sí" if supera else "no"),
                     " sería marcado como de mayor riesgo." if supera else " sería marcado."])]
    return F.fig_gauge(prob, UM_LR), texto, F.fig_contrib(contrib)


# ---- 5.6 Modelo original: predicción en vivo con el artefacto ya entrenado ----
@app.callback([Output(f"ori-{v}", "value") for v in ORDEN_SIM], Input("ori-b", "n_clicks"), Input("ori-m", "n_clicks"), Input("ori-a", "n_clicks"), prevent_initial_call=True)
def preajustes_original(*_):
    clave = {"ori-b": "bajo", "ori-m": "medio", "ori-a": "alto"}[ctx.triggered_id]
    return [PRESETS[clave][v] for v in ORDEN_SIM]


@app.callback(Output("ori-gauge", "figure"), Output("ori-texto", "children"), Output("ori-shap", "figure"), Output("ori-alerta", "children"),
              [Input(f"ori-{v}", "value") for v in ORDEN_SIM])
def prediccion_original(*valores):
    perfil = dict(zip(ORDEN_SIM, valores))
    faltan = [ETIQUETAS[k] for k, x in perfil.items() if x is None]
    if faltan:
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("Complete todos los campos antes de predecir. Falta: " + ", ".join(faltan), color="warning")
    try:
        perfil["edad"] = float(perfil["edad"])
        if not 18 <= perfil["edad"] <= 85:
            raise ValueError("La edad debe estar entre 18 y 85 años.")
        prob, d = OT.predecir(perfil)
    except Exception as e:   # entrada inválida o artefacto no disponible: se avisa sin detener la app
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(f"No se pudo calcular la predicción. {e}", color="danger")
    prev = A["prevalencia_tr"]
    supera = prob >= OT.UMBRAL
    texto = [html.P([html.B(f"{n_es(prob * 100, 1)} %"), f" de probabilidad estimada de reportar dificultad cognitiva, es decir {n_es(prob / prev, 1)} veces la prevalencia de la muestra ({pct_es(prev)})."]),
             html.P([f"Con el umbral de {n_es(OT.UMBRAL, 3)}, este perfil ", html.B("sí" if supera else "no"), " sería marcado como de mayor riesgo."])]
    return F.fig_gauge(prob, OT.UMBRAL), texto, OT.fig_shap(d), None


# ---- 5.x Pestañas del benchmark ----
@app.callback(Output("b-mapa", "figure"), Input("b-metrica", "value"), Input("b-opt", "value"))
def b_mapa(metrica, opt):
    return E.fig_mapa(metrica, opt)


@app.callback(Output("b-rangos", "figure"), Input("b-factor", "value"))
def b_rangos(factor):
    return E.fig_rangos(factor)


@app.callback(Output("b-tabla", "children"), Input("b-f-modelo", "value"), Input("b-f-bal", "value"), Input("b-f-opt", "value"))
def b_tabla(mod, bal, opt):
    m = E.E3["maestra"]
    if mod: m = m[m["modelo"].isin(mod)]
    if bal: m = m[m["balanceo"].isin(bal)]
    if opt: m = m[m["optimizador"].isin(opt)]
    cols = ["modelo", "balanceo", "optimizador", "PR_AUC_media", "PR_AUC_sd", "ROC_AUC_media", "Brier_media", "ECE_media", "Recall_05_media",
            "evals", "semilla_global", "semillas_por_fold", "t_total_min", "hiperparametros_finales_por_fold"]
    d = m[cols].sort_values("PR_AUC_media", ascending=False).round(4)
    t = _tabla(d, tam=12, style_cell_conditional=[{"if": {"column_id": "hiperparametros_finales_por_fold"}, "maxWidth": "260px", "whiteSpace": "nowrap",
                                                    "overflow": "hidden", "textOverflow": "ellipsis"}])
    t.tooltip_data = [{"hiperparametros_finales_por_fold": {"value": str(r["hiperparametros_finales_por_fold"]), "type": "text"}} for r in d.to_dict("records")]
    t.tooltip_duration = None
    return t


@app.callback(Output("o-conv", "figure"), Output("o-pareto", "figure"), Output("o-comp", "figure"), Input("tabs", "active_tab"))
def o_figs(_):
    return E.fig_convergencia(), E.fig_pareto(), E.fig_complejidad()


@app.callback(Output("f-calib", "figure"), Output("f-imp", "figure"), Output("f-lime", "figure"), Input("tabs", "active_tab"))
def f_figs(_):
    return E.fig_calibracion_balanceo(), E.fig_importancia(), E.fig_lime()


@app.callback(Output("r-tabla", "children"), Input("r-an", "value"))
def r_tabla(an):
    return _tabla(E.tabla_robustez(an), tam=14)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
