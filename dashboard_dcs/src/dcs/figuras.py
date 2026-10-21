"""
figuras.py
==========
Todas las figuras Plotly del dashboard. El JBook importa este mismo archivo, de modo que
cada gráfica del informe es idéntica a la que se ve en el dashboard.

Convenciones
------------
* Fondo claro (crema / blanco), tipografía DM Sans y una paleta de tierras con acento teal.
* Cada función recibe datos ya filtrados (``df``) o los artefactos de ``calculos.py`` (``A``)
  y devuelve un ``plotly.graph_objects.Figure``.
* ``separators=",."`` hace que los decimales se vean con coma, como en el Entregable 1.

Secciones
---------
1. Estilo común y utilidades.
2. Pestaña 1 · Contexto del problema.
3. Pestaña 2 · EDA.
4. Pestaña 3 · Modelos base.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calculos import (BLOQUE_DE, BLOQUES, ETIQUETAS, GRUPOS_EDAD_CORTES, cramers_v, grupo_edad,
                       prevalencia_por, wilson)
from .utils import (CATEGORIAS, DICCIONARIO, OBJETIVO, ORDENES, PREDICTORAS, PREDICTORAS_CAT)

# ---------------------------------------------------------------------------
# 1. ESTILO COMÚN Y UTILIDADES
# ---------------------------------------------------------------------------
C = {
    "terracota": "#B9543A", "terracota_claro": "#E7B7A5", "teal": "#2E6F77", "teal_claro": "#A9CBCF",
    "arena": "#E9DDC9", "dorado": "#C99A3B", "salvia": "#7E9C84", "ciruela": "#7A4B6B",
    "texto": "#3A302A", "suave": "#8A7B6E", "rejilla": "#EEE5D6", "fondo": "#FFFDF9",
    "sin": "#5E9AA6", "con": "#C0583F",
}
PALETA_BLOQUES = {"Salud mental": C["ciruela"], "Sensorial": C["dorado"], "Cardiometabólico": C["terracota"],
                  "Neurovascular": "#8C2F39", "Estilo de vida": C["salvia"], "Sociodemográfico": C["teal"]}
FUENTE = "DM Sans, Helvetica, Arial, sans-serif"

GRUPO_EDAD_LBL = [f"{a + 1}-{b}" for a, b in zip(GRUPOS_EDAD_CORTES[:-1], GRUPOS_EDAD_CORTES[1:])]


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas que usan varias figuras (grupos de edad)."""
    d = df.copy()
    d["grupo_edad_q"] = pd.cut(d["edad"], GRUPOS_EDAD_CORTES, labels=GRUPO_EDAD_LBL).astype(str)
    d["grupo_edad_amplio"] = pd.cut(d["edad"], [17, 34, 49, 64, 74, 85],
                                    labels=["18-34", "35-49", "50-64", "65-74", "75-85"]).astype(str)
    d["grupo_edad_acv"] = pd.cut(d["edad"], [17, 44, 54, 64, 74, 85],
                                 labels=["18-44", "45-54", "55-64", "65-74", "75+"]).astype(str)
    d["anio"] = d["anio"].astype(int)
    return d


def _estilo(fig: go.Figure, titulo: str | None = None, alto: int = 380, margen=None) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, x=0.01, xanchor="left", font=dict(size=14, color=C["texto"])) if titulo else None,
        height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["fondo"],
        font=dict(family=FUENTE, size=12, color=C["texto"]), separators=",.",
        margin=margen or dict(l=55, r=20, t=80 if titulo else 20, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=C["rejilla"], borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FUENTE, size=12, color=C["texto"]),
                        bordercolor=C["arena"]),
    )
    fig.update_xaxes(gridcolor=C["rejilla"], zerolinecolor=C["rejilla"], linecolor=C["arena"], automargin=True)
    fig.update_yaxes(gridcolor=C["rejilla"], zerolinecolor=C["rejilla"], linecolor=C["arena"], automargin=True)
    # Leyenda: por defecto arriba, a la izquierda y bajo el título; si la figura la puso debajo, se ancla por arriba
    lg = fig.layout.legend
    if lg.y is None:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    elif lg.y < 0:
        fig.update_layout(legend_yanchor="top")
    return fig


def fig_vacia(mensaje: str = "No hay datos con los filtros actuales", alto: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=C["suave"]))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _estilo(fig, alto=alto)


def _err(p, lo, hi):
    return dict(type="data", symmetric=False, array=np.nan_to_num(hi - p), arrayminus=np.nan_to_num(p - lo),
                color=C["suave"], thickness=1.2, width=3)


def orden_categorias(var: str, cats_modelo: dict | None = None):
    if var in ("grupo_edad_q", "grupo_edad_amplio", "grupo_edad_acv"):
        return None
    base = CATEGORIAS.get(var) or ORDENES.get(var)
    if cats_modelo and var in cats_modelo:
        return cats_modelo[var]
    return base


# ---------------------------------------------------------------------------
# 2. PESTAÑA 1 · CONTEXTO DEL PROBLEMA
# ---------------------------------------------------------------------------
def fig_flujo_muestra() -> go.Figure:
    """Diagrama de Sankey con el recorrido de la muestra (cifras del capítulo 1 del Entregable 1)."""
    nodos = ["Adultos NHIS 2022-2023 · 57.173", "Respondió un informante · 1.025", "Responden por sí mismos · 56.148",
             "Objetivo sin respuesta · 19", "Muestra analítica · 56.129", "Entrenamiento · 44.903", "Prueba reservada · 11.226"]
    colores = [C["teal"], C["terracota_claro"], C["teal_claro"], C["terracota_claro"], C["teal"], C["salvia"], C["dorado"]]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=nodos, color=colores, pad=22, thickness=18, line=dict(color="white", width=1),
                  x=[0.001, 0.30, 0.30, 0.62, 0.62, 0.999, 0.999], y=[0.50, 0.04, 0.55, 0.22, 0.62, 0.45, 0.93]),
        link=dict(source=[0, 0, 2, 2, 4, 4], target=[1, 2, 3, 4, 5, 6], value=[1025, 56148, 19, 56129, 44903, 11226],
                  color=["rgba(185,84,58,.25)", "rgba(46,111,119,.25)", "rgba(185,84,58,.25)",
                         "rgba(46,111,119,.25)", "rgba(126,156,132,.35)", "rgba(201,154,59,.35)"],
                  hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f} adultos<extra></extra>")))
    return _estilo(fig, "Del archivo del CDC a la muestra de trabajo", 360, dict(l=10, r=10, t=60, b=10))


TABLA_LANCET = pd.DataFrame([
    ("Baja escolaridad", "Disponible 2022-2023", "educacion", "Máximo nivel educativo (5 categorías)"),
    ("Hipertensión", "Disponible 2022-2023", "hipertension", "Diagnóstico alguna vez"),
    ("Colesterol LDL alto", "Disponible 2022-2023", "colesterol_alto", "Colesterol alto diagnosticado alguna vez"),
    ("Diabetes", "Disponible 2022-2023", "diabetes", "Diagnóstico alguna vez"),
    ("Depresión", "Disponible 2022-2023", "depresion_dx, frecuencia_depresion", "Diagnóstico y frecuencia de síntomas"),
    ("Pérdida auditiva", "Disponible 2022-2023", "dificultad_auditiva", "Dificultad para oír, aun con audífonos"),
    ("Pérdida visual", "Disponible 2022-2023", "dificultad_visual", "Dificultad para ver, aun con gafas"),
    ("Tabaquismo", "Disponible 2022-2023", "tabaquismo", "Nunca, exfumador o fumador actual"),
    ("Obesidad", "Disponible 2022-2023", "imc_categoria", "Categoría de IMC (peso y talla autorreportados)"),
    ("Consumo excesivo de alcohol", "Solo 2022", "—", "Se preguntó solo en 2022"),
    ("Inactividad física", "Solo 2022", "—", "Se preguntó solo en 2022"),
    ("Aislamiento social", "No disponible", "—", "Sin medición comparable"),
    ("Traumatismo craneoencefálico", "No disponible", "—", "Sin medición comparable"),
    ("Contaminación del aire", "No disponible", "—", "No se mide en la encuesta"),
], columns=["Factor de riesgo (Lancet 2024)", "Disponibilidad", "Variable(s) en la base", "Cómo se mide"])


def fig_lancet_dona() -> go.Figure:
    cuenta = TABLA_LANCET["Disponibilidad"].value_counts()
    colores = {"Disponible 2022-2023": C["teal"], "Solo 2022": C["dorado"], "No disponible": C["terracota_claro"]}
    fig = go.Figure(go.Pie(labels=cuenta.index, values=cuenta.values, hole=0.62, sort=False,
                           marker=dict(colors=[colores[k] for k in cuenta.index], line=dict(color="white", width=2)),
                           textinfo="value", textfont=dict(size=15, color="white"),
                           hovertemplate="%{label}: %{value} de 14 factores<extra></extra>"))
    fig.add_annotation(text="<b>9</b> de 14", x=0.5, y=0.52, showarrow=False, font=dict(size=26, color=C["texto"]))
    fig.add_annotation(text="factores medibles", x=0.5, y=0.40, showarrow=False, font=dict(size=12, color=C["suave"]))
    fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return _estilo(fig, "Cobertura del marco Lancet 2024 en la NHIS", 340, dict(l=10, r=10, t=50, b=40))


def fig_mapa_variables() -> go.Figure:
    """Treemap del diccionario de variables: cada rectángulo es una variable de la base."""
    d = DICCIONARIO.copy()
    def grupo(r):
        if r["rol"] == "Predictora":
            return BLOQUE_DE.get(r["variable"], "Sociodemográfico")
        return {"Objetivo": "Variable objetivo", "Solo EDA": "Solo para el EDA", "Auditoría de fuga": "Descartadas por fuga",
                "Estratificación": "Metadatos", "Identificador": "Metadatos"}.get(r["rol"], r["rol"])
    d["grupo"] = d.apply(grupo, axis=1)
    d["etiqueta"] = d["variable"].map(lambda v: ETIQUETAS.get(v, v.replace("_", " ").capitalize()))
    colores = {**PALETA_BLOQUES, "Variable objetivo": C["con"], "Solo para el EDA": C["suave"],
               "Descartadas por fuga": "#B9A99A", "Metadatos": "#D8CCBA"}
    ids, labels, parents, colors, hover = ["Base"], ["Base NHIS<br>(25 columnas)"], [""], ["#F4EBDD"], [""]
    for g, sub in d.groupby("grupo", sort=False):
        ids.append(g); labels.append(g); parents.append("Base"); colors.append(colores[g]); hover.append("")
        for _, r in sub.iterrows():
            ids.append(f"{g}/{r['variable']}"); labels.append(r["etiqueta"]); parents.append(g)
            colors.append(colores[g])
            hover.append(f"<b>{r['variable']}</b> ({r['variable_nhis']})<br>{r['significado']}<br>"
                          f"<i>Tipo: {r['tipo']}</i>")
    fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, marker=dict(colors=colors, line=dict(color="white", width=2)),
                               hovertext=hover, hoverinfo="text", textfont=dict(color="white", size=13),
                               branchvalues="remainder", pathbar=dict(visible=True), root_color="#F4EBDD"))
    fig.update_traces(textinfo="label")
    return _estilo(fig, "Mapa de variables: qué entra al modelo y qué se queda fuera", 430, dict(l=5, r=5, t=50, b=5))


def fig_auditoria_fuga(A) -> go.Figure:
    """ROC-AUC de cada variable por separado: una sola variable no debería acercarse a 0,9."""
    d = A["auc_uni"].copy()
    d["etiqueta"] = d["variable"].map(lambda v: ETIQUETAS.get(v, v))
    col = {"Predictora": C["teal"], "Sospechosa de fuga": C["con"], "Metadato": "#B9A99A"}
    fig = go.Figure()
    for g in ["Predictora", "Sospechosa de fuga", "Metadato"]:
        s = d[d["grupo"] == g]
        fig.add_trace(go.Bar(y=s["etiqueta"], x=s["auc"], orientation="h", name=g, marker_color=col[g],
                             error_x=dict(type="data", array=s["de"], color=C["suave"], thickness=1),
                             hovertemplate="<b>%{y}</b><br>AUC = %{x:.3f}<extra></extra>"))
    for x, t in [(0.5, "azar"), (0.9, "zona de alerta")]:
        fig.add_vline(x=x, line_dash="dash", line_color=C["suave"], annotation_text=t, annotation_position="top",
                      annotation_font=dict(size=10, color=C["suave"]))
    fig.update_xaxes(range=[0.45, 1.0], title="ROC-AUC con una sola variable (validación cruzada de 5 particiones)")
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    return _estilo(fig, "Auditoría de fuga de datos: desempeño de cada variable por separado", 520,
                   dict(l=10, r=20, t=50, b=70))


# ---------------------------------------------------------------------------
# 3. PESTAÑA 2 · EDA   (reciben el DataFrame ya filtrado)
# ---------------------------------------------------------------------------
def kpis_eda(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "casos": 0, "prev": np.nan, "lo": np.nan, "hi": np.nan, "edad": np.nan, "mujeres": np.nan, "mayores": np.nan}
    k, n = int(df[OBJETIVO].sum()), len(df)
    lo, hi = wilson(k, n)
    return {"n": n, "casos": k, "prev": k / n, "lo": float(lo), "hi": float(hi), "edad": float(df["edad"].median()),
            "mujeres": float((df["sexo"] == "Mujer").mean()), "mayores": float((df["edad"] >= 65).mean())}


def fig_niveles(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    niv = df["nivel_dificultad_cognitiva"].value_counts().reindex(ORDENES["nivel_dificultad_cognitiva"]).fillna(0)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], specs=[[{"type": "xy"}, {"type": "domain"}]],
                        subplot_titles=("Pregunta original (4 niveles)", "Variable binaria del proyecto"))
    fig.add_trace(go.Bar(x=niv.index, y=niv.values, marker_color=[C["sin"], C["terracota_claro"], C["con"], "#8C2F39"],
                         text=[f"{v / niv.sum():.1%}".replace(".", ",") for v in niv.values], textposition="outside",
                         hovertemplate="%{x}: %{y:,.0f} adultos<extra></extra>", showlegend=False), 1, 1)
    b = df[OBJETIVO].value_counts().reindex([0, 1]).fillna(0)
    fig.add_trace(go.Pie(labels=["Sin dificultad (0)", "Con dificultad (1)"], values=b.values, hole=0.55,
                         marker=dict(colors=[C["sin"], C["con"]], line=dict(color="white", width=2)), sort=False,
                         textinfo="percent", hovertemplate="%{label}: %{value:,.0f}<extra></extra>"), 1, 2)
    fig.update_yaxes(title="adultos", row=1, col=1)
    fig.update_annotations(font=dict(size=12, color=C["suave"]))
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.75, xanchor="center"))
    return _estilo(fig, "Variable objetivo: ¿tiene dificultad para recordar o concentrarse?", 380, dict(l=55, r=20, t=80, b=70))


def fig_trimestre(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    d = df.assign(periodo=df["anio"].astype(str) + "-T" + df["trimestre"].astype(str))
    p = prevalencia_por(d, "periodo").sort_values("periodo")
    fig = go.Figure(go.Scatter(x=p["periodo"], y=p["prevalencia"], mode="lines+markers",
                               line=dict(color=C["con"], width=2.5), marker=dict(size=9, color=C["con"]),
                               error_y=_err(p["prevalencia"], p["ic_inf"], p["ic_sup"]), customdata=np.c_[p["n"], p["casos"]],
                               hovertemplate="<b>%{x}</b><br>Prevalencia %{y:.1%}<br>n = %{customdata[0]:,.0f} · casos = %{customdata[1]:,.0f}<extra></extra>"))
    fig.add_hline(y=df[OBJETIVO].mean(), line_dash="dash", line_color=C["suave"],
                  annotation_text="prevalencia global del filtro", annotation_font=dict(size=10, color=C["suave"]))
    fig.update_yaxes(tickformat=".0%", title="prevalencia (IC 95 % de Wilson)")
    return _estilo(fig, "¿Cambia la prevalencia entre años y trimestres?", 340)


def fig_hist_edad(df: pd.DataFrame, modo: str = "conteo") -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    fig = go.Figure()
    for val, nom, col in [(0, "Sin dificultad", C["sin"]), (1, "Con dificultad", C["con"])]:
        s = df.loc[df[OBJETIVO] == val, "edad"].dropna()
        fig.add_trace(go.Histogram(x=s, name=nom, marker_color=col, xbins=dict(start=17.5, end=85.5, size=2),
                                   histnorm="probability density" if modo == "densidad" else "",
                                   opacity=0.85 if modo == "conteo" else 0.6,
                                   hovertemplate="%{x} años<br>%{y}<extra>" + nom + "</extra>"))
    fig.update_layout(barmode="stack" if modo == "conteo" else "overlay", bargap=0.04)
    fig.update_xaxes(title="edad (años; 85 = 85 o más)")
    fig.update_yaxes(title="adultos" if modo == "conteo" else "densidad")
    return _estilo(fig, "Distribución de la edad según la variable objetivo", 360)


def fig_prev_edad(df: pd.DataFrame, split: str | None = None) -> go.Figure:
    """Prevalencia por grupo de edad quinquenal con banda de IC; permite desagregar por otra variable."""
    if len(df) == 0:
        return fig_vacia()
    fig = go.Figure()
    niveles = [(None, df)] if not split else [(c, g) for c, g in df.groupby(split, observed=True)]
    paleta = [C["con"], C["teal"], C["dorado"], C["ciruela"], C["salvia"]]
    mid = {l: (a + 1 + b) / 2 for l, a, b in zip(GRUPO_EDAD_LBL, GRUPOS_EDAD_CORTES[:-1], GRUPOS_EDAD_CORTES[1:])}
    for i, (nivel, g) in enumerate(niveles):
        p = prevalencia_por(g, "grupo_edad_q", GRUPO_EDAD_LBL)
        p = p[p["n"] >= 30]
        if p.empty:
            continue
        col = paleta[i % len(paleta)]
        x = p["grupo_edad_q"].map(mid)
        nombre = "Todos" if nivel is None else str(nivel)
        rgb = tuple(int(col[j:j + 2], 16) for j in (1, 3, 5))
        fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(p["ic_sup"]) + list(p["ic_inf"])[::-1], fill="toself", mode="lines",
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=p["prevalencia"], mode="lines+markers", name=nombre, line=dict(color=col, width=2.5),
                                 marker=dict(size=7), customdata=np.c_[p["grupo_edad_q"], p["n"]],
                                 hovertemplate="<b>%{customdata[0]} años</b><br>Prevalencia %{y:.1%}<br>n = %{customdata[1]:,.0f}<extra>" + nombre + "</extra>"))
    fig.update_xaxes(title="edad (centro del grupo de 5 años)")
    fig.update_yaxes(tickformat=".0%", title="prevalencia (IC 95 %)", rangemode="tozero")
    return _estilo(fig, "La relación con la edad tiene forma de U", 360)


def fig_prev_categoria(df: pd.DataFrame, var: str, cats_modelo: dict | None = None) -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    orden = orden_categorias(var, cats_modelo)
    d = df.copy()
    d[var] = d[var].astype(object).where(d[var].notna(), "Faltante")
    p = prevalencia_por(d, var, (list(orden) + ["Faltante"]) if orden else None)
    p = p[p["n"] > 0]
    glob = df[OBJETIVO].mean()
    colores = [C["con"] if v > glob else C["sin"] for v in p["prevalencia"]]
    fig = go.Figure(go.Bar(y=p[var].astype(str), x=p["prevalencia"], orientation="h", marker_color=colores,
                           error_x=_err(p["prevalencia"], p["ic_inf"], p["ic_sup"]), customdata=np.c_[p["n"], p["casos"]],
                           text=[f"{v:.1%}".replace(".", ",") for v in p["prevalencia"]], textposition="inside",
                           insidetextanchor="start", textfont=dict(color="white", size=12),
                           hovertemplate="<b>%{y}</b><br>Prevalencia %{x:.1%}<br>n = %{customdata[0]:,.0f} · casos = %{customdata[1]:,.0f}<extra></extra>"))
    fig.add_vline(x=glob, line_dash="dash", line_color=C["texto"], annotation_text=f"global {glob:.1%}".replace(".", ","),
                  annotation_font=dict(size=10), annotation_position="top")
    fig.update_xaxes(tickformat=".0%", title="prevalencia de dificultad cognitiva (IC 95 %)", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _estilo(fig, f"Prevalencia según: {ETIQUETAS.get(var, var)}", max(300, 90 + 42 * len(p)))


def tabla_cramer(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for v in PREDICTORAS_CAT:
        t = pd.crosstab(df[v], df[OBJETIVO])
        if t.shape[0] < 2 or t.shape[1] < 2:
            continue
        filas.append([v, cramers_v(t.values)])
    t = pd.DataFrame(filas, columns=["variable", "V"])
    if len(df) > 0 and df["edad"].notna().any():
        g = pd.crosstab(df["grupo_edad_amplio"], df[OBJETIVO])
        if g.shape[0] >= 2 and g.shape[1] >= 2:
            t.loc[len(t)] = ["edad", cramers_v(g.values)]
    return t


def fig_cramer(df: pd.DataFrame) -> go.Figure:
    if len(df) < 50:
        return fig_vacia()
    t = tabla_cramer(df).sort_values("V")
    if t.empty:
        return fig_vacia("Con estos filtros ninguna variable tiene variación suficiente")
    t["etq"] = t["variable"].map(ETIQUETAS) + np.where(t["variable"] == "edad", " (5 grupos)", "")
    t["bloque"] = t["variable"].map(BLOQUE_DE)
    fig = go.Figure()
    for b in BLOQUES:
        s = t[t["bloque"] == b]
        if s.empty:
            continue
        fig.add_trace(go.Bar(y=s["etq"], x=s["V"], orientation="h", name=b, marker_color=PALETA_BLOQUES[b],
                             hovertemplate="<b>%{y}</b><br>V de Cramér = %{x:.3f}<extra></extra>"))
    for x, tx in [(0.1, "pequeña"), (0.3, "moderada")]:
        fig.add_vline(x=x, line_dash="dot", line_color=C["suave"], annotation_text=tx, annotation_position="top",
                      annotation_font=dict(size=10, color=C["suave"]))
    fig.update_xaxes(title="V de Cramér (asociación con la variable objetivo)", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _estilo(fig, "¿Qué tan fuerte se asocia cada factor con la dificultad cognitiva?", 470, dict(l=10, r=20, t=50, b=80))


def fig_heatmap_dos(df: pd.DataFrame, vx: str, vy: str, cats_modelo: dict | None = None, n_min: int = 30) -> go.Figure:
    """Prevalencia en el cruce de dos variables; las celdas con menos de n_min adultos se omiten."""
    if len(df) == 0:
        return fig_vacia()
    if vx == vy:
        return fig_vacia("Elija dos variables distintas")
    ox = orden_categorias(vx, cats_modelo) or sorted(df[vx].dropna().unique(), key=str)
    oy = orden_categorias(vy, cats_modelo) or sorted(df[vy].dropna().unique(), key=str)
    g = df.groupby([vy, vx], observed=True)[OBJETIVO].agg(["mean", "count"]).reset_index()
    z = g.pivot(index=vy, columns=vx, values="mean").reindex(index=[o for o in oy if o in set(g[vy])],
                                                              columns=[o for o in ox if o in set(g[vx])])
    n = g.pivot(index=vy, columns=vx, values="count").reindex(index=z.index, columns=z.columns)
    z = z.where(n >= n_min)
    txt = z.map(lambda v: "" if pd.isna(v) else f"{v:.0%}") if hasattr(z, "map") else z.applymap(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                               text=txt.values, texttemplate="%{text}", customdata=n.values,
                               colorscale=[[0, "#F6EFE2"], [0.5, "#E7B7A5"], [1, "#8C2F39"]], zmin=0,
                               colorbar=dict(title="prevalencia", tickformat=".0%", thickness=12),
                               hovertemplate=f"{ETIQUETAS.get(vx, vx)}: %{{x}}<br>{ETIQUETAS.get(vy, vy)}: %{{y}}<br>"
                                             "Prevalencia %{z:.1%}<br>n = %{customdata:,.0f}<extra></extra>"))
    fig.update_xaxes(title=ETIQUETAS.get(vx, vx.replace("_", " ")), side="bottom")
    fig.update_yaxes(title=ETIQUETAS.get(vy, vy.replace("_", " ")), autorange="reversed")
    return _estilo(fig, "Cruce de dos factores: dónde se concentra la dificultad cognitiva", 420)


def fig_acv_edad(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    d = df.dropna(subset=["acv"])
    orden = ["18-44", "45-54", "55-64", "65-74", "75+"]
    fig = go.Figure()
    for acv, col in [("No", C["teal"]), ("Sí", C["con"])]:
        s = d[d["acv"] == acv]
        if s.empty:
            continue
        p = prevalencia_por(s, "grupo_edad_acv", orden)
        fig.add_trace(go.Scatter(x=p["grupo_edad_acv"], y=p["prevalencia"], mode="lines+markers", name=f"ACV: {acv}",
                                 line=dict(color=col, width=2.5), marker=dict(size=9), error_y=_err(p["prevalencia"], p["ic_inf"], p["ic_sup"]),
                                 customdata=p["n"], hovertemplate="<b>%{x}</b><br>Prevalencia %{y:.1%}<br>n = %{customdata:,.0f}<extra>ACV " + acv + "</extra>"))
    fig.update_xaxes(title="grupo de edad"); fig.update_yaxes(tickformat=".0%", title="prevalencia (IC 95 %)", rangemode="tozero")
    return _estilo(fig, "Dificultad cognitiva según antecedente de ACV, por edad", 360)


def tabla_or_acv(df: pd.DataFrame):
    """OR del ACV por estrato de edad, OR crudo y OR de Mantel-Haenszel (con IC 95 %)."""
    from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
    d = df.dropna(subset=["acv"]).copy()
    d["acv_si"] = (d["acv"] == "Sí").astype(int)
    orden = ["18-44", "45-54", "55-64", "65-74", "75+"]
    filas, tablas = [], []
    def or_ic(t):
        a, b, c, dd = [float(x) + 0.5 for x in t.ravel()]
        o = (a * dd) / (b * c); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
        return o, np.exp(np.log(o) - 1.96 * se), np.exp(np.log(o) + 1.96 * se)
    for g in orden:
        s = d[d["grupo_edad_acv"] == g]
        t = pd.crosstab(s["acv_si"], s[OBJETIVO]).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
        if t[0].sum() < 5 or t[1].sum() < 5:
            continue
        o, lo, hi = or_ic(t); tablas.append(t)
        filas.append([f"Edad {g}", o, lo, hi, int(t[0].sum())])
    t_all = pd.crosstab(d["acv_si"], d[OBJETIVO]).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    if t_all[0].sum() >= 5 and t_all[1].sum() >= 5:
        try:
            cr = Table2x2(t_all + 0.0); lo, hi = cr.oddsratio_confint()
            filas.append(["OR crudo (todas las edades)", cr.oddsratio, lo, hi, int(t_all[0].sum())])
        except Exception:
            pass
    if len(tablas) >= 2:
        try:
            mh = StratifiedTable([t + 0.5 for t in tablas]); lo, hi = mh.oddsratio_pooled_confint()
            filas.append(["OR ajustado por edad (Mantel-Haenszel)", mh.oddsratio_pooled, lo, hi, int(t_all[0].sum())])
        except Exception:
            pass
    return pd.DataFrame(filas, columns=["grupo", "OR", "ic_inf", "ic_sup", "n_acv"])


def fig_or_acv(df: pd.DataFrame) -> go.Figure:
    t = tabla_or_acv(df)
    if t.empty:
        return fig_vacia("Con los filtros actuales no hay suficientes adultos con ACV")
    colores = [C["teal"] if g.startswith("Edad") else (C["dorado"] if "crudo" in g else C["con"]) for g in t["grupo"]]
    fig = go.Figure()
    for i, r in t.iterrows():
        fig.add_trace(go.Scatter(x=[r["ic_inf"], r["ic_sup"]], y=[r["grupo"]] * 2, mode="lines", line=dict(color=colores[i], width=3),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=t["OR"], y=t["grupo"], mode="markers+text", marker=dict(size=13, color=colores, line=dict(color="white", width=1.5)),
                             text=[f"{o:.2f}".replace(".", ",") for o in t["OR"]], textposition="top center", showlegend=False,
                             customdata=np.c_[t["ic_inf"], t["ic_sup"], t["n_acv"]],
                             hovertemplate="<b>%{y}</b><br>OR = %{x:.2f} (IC 95 %: %{customdata[0]:.2f}–%{customdata[1]:.2f})<br>adultos con ACV: %{customdata[2]:,.0f}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash", line_color=C["texto"])
    rango = [float(np.log10(min(t["ic_inf"].min() * 0.85, 0.9))), float(np.log10(t["ic_sup"].max() * 1.2))]
    fig.update_xaxes(type="log", range=rango, title="odds ratio del ACV (escala logarítmica)", tickvals=[0.5, 1, 2, 3, 5, 8, 12], ticktext=["0,5", "1", "2", "3", "5", "8", "12"])
    fig.update_yaxes(autorange="reversed")
    return _estilo(fig, "El efecto del ACV es mayor en los más jóvenes", 360, dict(l=10, r=20, t=50, b=50))


def fig_info_mutua(A) -> go.Figure:
    mi = A["mi"].sort_values()
    fig = go.Figure()
    for b in BLOQUES:
        s_ = mi[[BLOQUE_DE[v] == b for v in mi.index]]
        if s_.empty:
            continue
        fig.add_trace(go.Bar(y=[ETIQUETAS[v] for v in s_.index], x=s_.values, orientation="h", name=b, marker_color=PALETA_BLOQUES[b],
                             hovertemplate="<b>%{y}</b><br>información mutua = %{x:.4f} nats<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[ETIQUETAS[v] for v in mi.index])
    fig.update_xaxes(title="información mutua con la variable objetivo (nats)")
    return _estilo(fig, "Información mutua: cualquier tipo de dependencia", 470, dict(l=10, r=20, t=90, b=50))


def fig_cramer_heatmap(A) -> go.Figure:
    m = A["cramer"].copy()
    nombres = [ETIQUETAS.get(c, c) + (" (grupos)" if c == "edad" else "") for c in m.columns]
    z = m.values.copy(); z[np.triu_indices_from(z, 1)] = np.nan
    fig = go.Figure(go.Heatmap(z=z, x=nombres, y=nombres, zmin=0, zmax=0.6, colorscale=[[0, "#FBF6EC"], [1, "#8C2F39"]],
                               text=[["" if np.isnan(v) else f"{v:.2f}".replace(".", ",") for v in fila] for fila in z], texttemplate="%{text}", textfont=dict(size=9),
                               colorbar=dict(title="V de Cramér", thickness=12),
                               hovertemplate="%{y} × %{x}<br>V = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(tickangle=-45)
    return _estilo(fig, "Asociación entre las propias predictoras", 520, dict(l=10, r=10, t=50, b=140))


COLOR_PCA = {"Dificultad cognitiva": OBJETIVO, "Sexo": "sexo", "Antecedente de ACV": "acv", "Diagnóstico de depresión": "depresion_dx",
             "Hipertensión": "hipertension", "Año": "anio", "Edad": "edad", "Grupo (K-means)": "cluster"}


def fig_pca(A, puntos: pd.DataFrame, color: str = "Dificultad cognitiva") -> go.Figure:
    """Proyección de los adultos en las dos primeras componentes. ``puntos`` es la muestra de
    ``A['pca']['puntos']`` ya filtrada con los controles del EDA."""
    p = puntos
    if p.empty:
        return fig_vacia()
    col = COLOR_PCA[color]
    fig = go.Figure()
    if col == "edad":
        fig.add_trace(go.Scattergl(x=p["CP1"], y=p["CP2"], mode="markers", marker=dict(size=5, color=p["edad"], colorscale=[[0, C["teal_claro"]], [1, "#8C2F39"]],
                                                                                          opacity=0.65, colorbar=dict(title="edad", thickness=12)),
                                   hovertemplate="CP1 %{x:.2f} · CP2 %{y:.2f}<br>edad %{marker.color}<extra></extra>", showlegend=False))
    else:
        valores = p[col].astype(str).map({"0": "Sin dificultad", "1": "Con dificultad"}) if col == OBJETIVO else p[col].astype(object).where(p[col].notna(), "Sin dato").astype(str)
        for i, v in enumerate(sorted(valores.unique())):
            s = p[valores == v]
            c = {"Sin dificultad": C["sin"], "Con dificultad": C["con"]}.get(v, [C["teal"], C["dorado"], C["ciruela"], C["salvia"], C["terracota"], "#8C2F39"][i % 6])
            fig.add_trace(go.Scattergl(x=s["CP1"], y=s["CP2"], mode="markers", name=v, marker=dict(size=5, color=c, opacity=0.55),
                                       hovertemplate="CP1 %{x:.2f} · CP2 %{y:.2f}<extra>" + v + "</extra>"))
    ev = A["pca"]["varianza"]
    fig.update_xaxes(title=f"CP1 ({ev[0]:.1%} de la varianza)".replace(".", ","), zeroline=True)
    fig.update_yaxes(title=f"CP2 ({ev[1]:.1%} de la varianza)".replace(".", ","), zeroline=True)
    return _estilo(fig, f"Proyección PCA coloreada por: {color.lower()}", 430)


def fig_scree(A) -> go.Figure:
    v, ac = A["pca"]["varianza"], A["pca"]["acum"]
    k = np.arange(1, len(v) + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=v, name="individual", marker_color=C["teal_claro"], hovertemplate="CP%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=ac, name="acumulada", mode="lines+markers", line=dict(color=C["con"], width=2.5), marker=dict(size=5),
                             hovertemplate="hasta CP%{x}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=0.8, line_dash="dash", line_color=C["suave"], annotation_text="80 %", annotation_font=dict(size=10))
    fig.update_yaxes(tickformat=".0%", title="varianza explicada"); fig.update_xaxes(title="componente principal")
    return _estilo(fig, f"Se necesitan {A['pca']['k80']} de {len(v)} componentes para llegar al 80 %", 340)


def fig_clusters(A) -> go.Figure:
    t = A["pca"]["perfil_clusters"]
    fig = go.Figure(go.Bar(x=[f"Grupo {c}<br>(n = {n:,})".replace(",", ".") for c, n in zip(t["cluster"], t["n"])], y=t["prevalencia"],
                           marker_color=[C["con"] if v > 0.4 else (C["terracota_claro"] if v > 0.2 else C["sin"]) for v in t["prevalencia"]],
                           text=[f"{v:.0%}" for v in t["prevalencia"]], textposition="outside",
                           customdata=np.c_[t["edad_media"], t["depresion_dx"], t["hipertension"], t["fumador_actual"], t["acv"]],
                           hovertemplate="<b>%{x}</b><br>Prevalencia %{y:.1%}<br>edad media %{customdata[0]:.1f}<br>con depresión %{customdata[1]:.0%}"
                                         "<br>con hipertensión %{customdata[2]:.0%}<br>fumadores actuales %{customdata[3]:.0%}<br>con ACV %{customdata[4]:.0%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="prevalencia", range=[0, max(t["prevalencia"]) * 1.2])
    return _estilo(fig, "Grupos exploratorios (K-means) y su prevalencia", 340)


def fig_faltantes(df: pd.DataFrame) -> go.Figure:
    if len(df) == 0:
        return fig_vacia()
    cols = PREDICTORAS + ["imc"]
    f = df[cols].isna().mean().sort_values()
    f = f[f > 0] if (f > 0).any() else f
    colores = [C["terracota"] if c == "imc" else C["teal"] for c in f.index]
    fig = go.Figure(go.Bar(y=[ETIQUETAS.get(c, "IMC numérico (solo EDA)") for c in f.index], x=f.values, orientation="h", marker_color=colores,
                           text=[f"{v:.1%}".replace(".", ",") for v in f.values], textposition="outside",
                           hovertemplate="<b>%{y}</b><br>%{x:.2%} de faltantes<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="proporción de datos faltantes", range=[0, max(f.max() * 1.25, 0.01)])
    return _estilo(fig, "Datos faltantes por variable", 380, dict(l=10, r=40, t=50, b=50))


def fig_sesgo(A) -> go.Figure:
    t = A["sesgo"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t["indicador"], x=t["Muestra sin ponderar"], orientation="h", name="Muestra (sin ponderar)", marker_color=C["terracota"],
                         hovertemplate="%{y}<br>muestra: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(y=t["indicador"], x=t["Población ponderada (EE. UU.)"], orientation="h", name="Población (ponderada)", marker_color=C["teal"],
                         hovertemplate="%{y}<br>población: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="proporción"); fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return _estilo(fig, "Representatividad: la muestra frente a la población de EE. UU.", 400, dict(l=10, r=20, t=50, b=80))


# ---------------------------------------------------------------------------
# 4. PESTAÑA 3 · MODELOS BASE
# ---------------------------------------------------------------------------
MODELOS = ["Regresión logística", "SVM lineal"]
COLOR_MODELO = {"Regresión logística": C["con"], "SVM lineal": C["teal"], "Dummy (prior)": "#B9A99A", "Dummy (stratified)": "#CFC3B2"}


def datos_modelo(A, modelo: str, conjunto: str):
    """Devuelve (y, p): prueba real o predicciones out-of-fold del entrenamiento."""
    if conjunto == "prueba":
        return A["y_te"], A["p_te"][modelo]
    return A["y_tr"], A["p_oof"][modelo]


def metricas_umbral(y, p, umbral: float) -> dict:
    from .calculos import metricas
    yhat = (p >= umbral).astype(int)
    m = metricas(y, p, yhat)
    vp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); vn = int(((yhat == 0) & (y == 0)).sum())
    m.update({"VP": vp, "FP": fp, "FN": fn, "VN": vn, "Especificidad": vn / max(vn + fp, 1), "VPN": vn / max(vn + fn, 1),
              "Marcados": (vp + fp) / len(y)})
    return m


def fig_confusion(y, p, umbral: float) -> go.Figure:
    m = metricas_umbral(y, p, umbral)
    z = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
    fila = z / z.sum(axis=1, keepdims=True)
    # texto de cada celda con separadores en estilo español (miles con punto, decimales con coma)
    txt = [[f"<b>{z[i, j]:,.0f}</b><br>({fila[i, j] * 100:.1f} %)".replace(",", "·").replace(".", ",").replace("·", ".") for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=fila, x=["Predice: sin dificultad", "Predice: con dificultad"], y=["Real: sin dificultad", "Real: con dificultad"],
                               text=txt, texttemplate="%{text}", colorscale=[[0, "#FBF6EC"], [1, "#2E6F77"]], showscale=False, zmin=0, zmax=1,
                               hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>", textfont=dict(size=14)))
    fig.update_yaxes(autorange="reversed")
    return _estilo(fig, f"Matriz de confusión con umbral {umbral:.2f}".replace(".", ","), 340, dict(l=10, r=10, t=50, b=40))


def fig_roc_pr(A, modelos: list[str], conjunto: str, umbrales: dict | None = None) -> go.Figure:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva precisión-recall"), horizontal_spacing=0.12)
    y0 = A["y_te"] if conjunto == "prueba" else A["y_tr"]
    for nombre in modelos:
        y, p = datos_modelo(A, nombre, conjunto)
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nombre} (AUC = {roc_auc_score(y, p):.3f})".replace(".", ","),
                                 line=dict(color=COLOR_MODELO[nombre], width=2.5), legendgroup=nombre,
                                 hovertemplate="FPR %{x:.2f} · recall %{y:.2f}<extra></extra>"), 1, 1)
        pr, rc, _ = precision_recall_curve(y, p)
        paso = max(1, len(pr) // 600)
        fig.add_trace(go.Scatter(x=rc[::paso], y=pr[::paso], mode="lines", name=f"{nombre} (AP = {average_precision_score(y, p):.3f})".replace(".", ","),
                                 line=dict(color=COLOR_MODELO[nombre], width=2.5), legendgroup=nombre,
                                 hovertemplate="recall %{x:.2f} · precisión %{y:.2f}<extra></extra>"), 1, 2)
        if umbrales and nombre in umbrales:
            t = umbrales[nombre]; yh = (p >= t).astype(int)
            tp = ((yh == 1) & (y == 1)).sum(); fp_ = ((yh == 1) & (y == 0)).sum(); fn_ = ((yh == 0) & (y == 1)).sum(); tn_ = ((yh == 0) & (y == 0)).sum()
            fig.add_trace(go.Scatter(x=[fp_ / max(fp_ + tn_, 1)], y=[tp / max(tp + fn_, 1)], mode="markers", showlegend=False, legendgroup=nombre,
                                     marker=dict(size=13, color=COLOR_MODELO[nombre], line=dict(color="white", width=2.5), symbol="diamond"),
                                     hovertemplate=f"Umbral {t:.2f}<extra></extra>"), 1, 1)
            fig.add_trace(go.Scatter(x=[tp / max(tp + fn_, 1)], y=[tp / max(tp + fp_, 1)], mode="markers", showlegend=False, legendgroup=nombre,
                                     marker=dict(size=13, color=COLOR_MODELO[nombre], line=dict(color="white", width=2.5), symbol="diamond"),
                                     hovertemplate=f"Umbral {t:.2f}<extra></extra>"), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["suave"], dash="dot"), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[y0.mean()] * 2, mode="lines", line=dict(color=C["suave"], dash="dot"), name=f"azar (prevalencia {y0.mean():.3f})".replace(".", ","),
                             hoverinfo="skip"), 1, 2)
    fig.update_xaxes(title="tasa de falsos positivos", row=1, col=1); fig.update_yaxes(title="recall (sensibilidad)", row=1, col=1)
    fig.update_xaxes(title="recall", row=1, col=2); fig.update_yaxes(title="precisión", range=[0, 1], row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=C["suave"]))
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))
    return _estilo(fig, None, 420, dict(l=55, r=20, t=40, b=100))


def fig_hist_prob(y, p, umbral: float) -> go.Figure:
    fig = go.Figure()
    for val, nom, col in [(0, "Sin dificultad", C["sin"]), (1, "Con dificultad", C["con"])]:
        fig.add_trace(go.Histogram(x=p[y == val], name=nom, marker_color=col, opacity=0.62, histnorm="probability density",
                                   xbins=dict(start=0, end=1, size=0.025), hovertemplate="prob. %{x:.2f}<br>densidad %{y:.2f}<extra>" + nom + "</extra>"))
    fig.add_vline(x=umbral, line_dash="dash", line_color=C["texto"], annotation_text=f"umbral {umbral:.2f}".replace(".", ","), annotation_font=dict(size=11))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="probabilidad predicha", range=[0, 1]); fig.update_yaxes(title="densidad")
    return _estilo(fig, "Cómo se separan las dos clases", 340)


def fig_barrido_umbral(A, modelo: str, umbral: float) -> go.Figure:
    from sklearn.metrics import precision_recall_curve
    y, p = A["y_tr"], A["p_oof"][modelo]
    pr, rc, um = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    paso = max(1, len(um) // 500)
    fig = go.Figure()
    for nom, v, col, ancho in [("Precisión", pr[:-1], C["teal"], 2), ("Recall", rc[:-1], C["con"], 2), ("F1", f1, C["texto"], 3)]:
        fig.add_trace(go.Scatter(x=um[::paso], y=v[::paso], name=nom, mode="lines", line=dict(color=col, width=ancho),
                                 hovertemplate="umbral %{x:.2f}<br>" + nom + " %{y:.2f}<extra></extra>"))
    opt = A["umbral"][modelo]["umbral"]
    fig.add_vline(x=opt, line_dash="dot", line_color=C["dorado"], annotation_text=f"óptimo {opt:.3f}".replace(".", ","), annotation_position="top left", annotation_font=dict(size=10))
    fig.add_vline(x=umbral, line_dash="dash", line_color=C["texto"], annotation_text="elegido", annotation_position="top right", annotation_font=dict(size=10))
    fig.update_xaxes(title="umbral de probabilidad (predicciones fuera de partición del entrenamiento)", range=[0, 0.9])
    fig.update_yaxes(range=[0, 1.02])
    return _estilo(fig, "El compromiso entre detectar casos y evitar falsas alarmas", 340)


def fig_calibracion(A, modelos: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.7], y=[0, 0.7], mode="lines", line=dict(color=C["suave"], dash="dot"), name="calibración perfecta", hoverinfo="skip"))
    for nombre in modelos:
        c = A["calibracion"][nombre]
        fig.add_trace(go.Scatter(x=c["predicha"], y=c["observada"], mode="lines+markers", name=nombre, line=dict(color=COLOR_MODELO[nombre], width=2.5),
                                 marker=dict(size=8), hovertemplate="predicha %{x:.1%}<br>observada %{y:.1%}<extra>" + nombre + "</extra>"))
    fig.update_xaxes(title="probabilidad predicha (media por decil)", tickformat=".0%"); fig.update_yaxes(title="proporción observada", tickformat=".0%")
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    return _estilo(fig, "Calibración en el conjunto de prueba", 360, dict(l=55, r=20, t=50, b=90))


def fig_comparacion(A, metrica: str = "PR-AUC") -> go.Figure:
    orden = ["Dummy (prior)", "Dummy (stratified)", "Regresión logística", "SVM lineal"]
    pun = [A["metricas_test"][m][metrica] for m in orden]
    lo = [A["boot"][m][metrica].quantile(0.025) for m in orden]; hi = [A["boot"][m][metrica].quantile(0.975) for m in orden]
    fig = go.Figure(go.Bar(x=orden, y=pun, marker_color=[COLOR_MODELO[m] for m in orden], error_y=_err(np.array(pun), np.array(lo), np.array(hi)),
                           text=[f"{v:.3f}".replace(".", ",") for v in pun], textposition="outside", textfont=dict(size=12),
                           customdata=np.c_[lo, hi], hovertemplate="<b>%{x}</b><br>" + metrica + " = %{y:.3f}<br>IC 95 %: %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra></extra>"))
    ymax = max(hi) * 1.18
    fig.update_yaxes(title=f"{metrica} en prueba (IC 95 % bootstrap)", range=[0, min(ymax, 1.05) if metrica != "Brier" else ymax])
    return _estilo(fig, f"Comparación con la línea base trivial: {metrica}", 380)


def tabla_metricas(A) -> pd.DataFrame:
    orden = ["Dummy (prior)", "Dummy (stratified)", "Regresión logística", "SVM lineal"]
    mets = ["PR-AUC", "ROC-AUC", "Recall", "Precisión", "F1", "Balanced acc.", "Brier", "Accuracy"]
    filas = []
    for m in orden:
        f = {"Modelo": m}
        for k in mets:
            b = A["boot"][m][k]
            f[k] = f"{A['metricas_test'][m][k]:.3f} [{b.quantile(.025):.3f}–{b.quantile(.975):.3f}]".replace(".", ",")
        filas.append(f)
    return pd.DataFrame(filas)


def fig_grid(A, modelo: str) -> go.Figure:
    fig = go.Figure()
    if modelo == "Regresión logística":
        g = A["grid_lr"]
        for cw, col in [("ninguno", C["con"]), ("balanced", C["teal"])]:
            s = g[g["class_weight"] == cw].sort_values("C")
            fig.add_trace(go.Scatter(x=s["C"], y=s["PR-AUC (val.)"], mode="lines+markers", name=f"validación · class_weight = {cw}", line=dict(color=col, width=2.5),
                                     error_y=dict(type="data", array=s["de PR-AUC"], color=col, thickness=1), marker=dict(size=8),
                                     hovertemplate="C = %{x}<br>PR-AUC val. %{y:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=s["C"], y=s["PR-AUC (entr.)"], mode="lines", name=f"entrenamiento · {cw}", line=dict(color=col, width=1.5, dash="dot"),
                                     hovertemplate="C = %{x}<br>PR-AUC entr. %{y:.4f}<extra></extra>"))
    else:
        s = A["grid_svm"].sort_values("C")
        fig.add_trace(go.Scatter(x=s["C"], y=s["PR-AUC (val.)"], mode="lines+markers", name="validación", line=dict(color=C["teal"], width=2.5), marker=dict(size=8),
                                 error_y=dict(type="data", array=s["de PR-AUC"], color=C["teal"], thickness=1), hovertemplate="C = %{x}<br>PR-AUC val. %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=s["C"], y=s["PR-AUC (entr.)"], mode="lines", name="entrenamiento", line=dict(color=C["teal"], width=1.5, dash="dot"),
                                 hovertemplate="C = %{x}<br>PR-AUC entr. %{y:.4f}<extra></extra>"))
    fig.update_xaxes(type="log", title="C (inverso de la regularización, escala log)", tickvals=[0.001, 0.01, 0.1, 1, 10], ticktext=["0,001", "0,01", "0,1", "1", "10"]); fig.update_yaxes(title="PR-AUC", tickformat=".3f")
    fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center", font=dict(size=10)))
    return _estilo(fig, f"Búsqueda de hiperparámetros · {modelo}", 360, dict(l=60, r=20, t=50, b=110))


def fig_curva_aprendizaje(A) -> go.Figure:
    d = A["curva_aprendizaje"]
    fig = go.Figure()
    for col, sd, nom, color in [("train", "train_sd", "entrenamiento", C["teal"]), ("val", "val_sd", "validación", C["con"])]:
        rgb = tuple(int(color[j:j + 2], 16) for j in (1, 3, 5))
        fig.add_trace(go.Scatter(x=list(d["n"]) + list(d["n"])[::-1], y=list(d[col] + d[sd]) + list(d[col] - d[sd])[::-1], fill="toself", mode="lines",
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d["n"], y=d[col], mode="lines+markers", name=nom, line=dict(color=color, width=2.5), marker=dict(size=7),
                                 hovertemplate="n = %{x:,.0f}<br>PR-AUC %{y:.4f}<extra>" + nom + "</extra>"))
    fig.add_hline(y=A["prevalencia_tr"], line_dash="dot", line_color=C["suave"], annotation_text="línea base trivial", annotation_font=dict(size=10))
    fig.update_xaxes(title="adultos usados para entrenar"); fig.update_yaxes(title="PR-AUC", range=[0.15, 0.62])
    return _estilo(fig, "Curva de aprendizaje de la regresión logística", 360)


def fig_or(A, bloque: str = "Todos", solo_sig: bool = False) -> go.Figure:
    t = A["or"].copy()
    if bloque != "Todos":
        t = t[t["bloque"] == bloque]
    if solo_sig:
        t = t[t["p_holm"] < 0.05]
    if t.empty:
        return fig_vacia("Ningún término cumple el filtro")
    t = t.sort_values("OR")
    sig = t["p_holm"] < 0.05
    fig = go.Figure()
    for _, r in t.iterrows():
        col = PALETA_BLOQUES.get(r["bloque"], C["suave"]) if r["p_holm"] < 0.05 else "#C9BDAE"
        fig.add_trace(go.Scatter(x=[r["ic_inf"], r["ic_sup"]], y=[r["etiqueta"]] * 2, mode="lines", line=dict(color=col, width=2.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=t["OR"], y=t["etiqueta"], mode="markers", showlegend=False,
                             marker=dict(size=10, color=[PALETA_BLOQUES.get(b, C["suave"]) if s else "#C9BDAE" for b, s in zip(t["bloque"], sig)], line=dict(color="white", width=1.2)),
                             customdata=np.c_[t["ic_inf"], t["ic_sup"], t["p_holm"]],
                             hovertemplate="<b>%{y}</b><br>OR = %{x:.2f} (IC 95 %: %{customdata[0]:.2f}–%{customdata[1]:.2f})<br>p (Holm) = %{customdata[2]:.3g}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash", line_color=C["texto"])
    fig.update_xaxes(type="log", title="odds ratio ajustado (escala logarítmica, IC 95 %)", tickvals=[0.5, 0.75, 1, 1.5, 2, 3, 5], ticktext=["0,5", "0,75", "1", "1,5", "2", "3", "5"])
    return _estilo(fig, "Odds ratios ajustados (gris: no significativo tras Holm)", max(320, 90 + 24 * len(t)), dict(l=10, r=20, t=50, b=50))


def fig_dep_edad(A) -> go.Figure:
    d, o = A["dep_edad"], A["prev_edad_obs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=o["edad"], y=o["prevalencia"], mode="markers", name="prevalencia observada (quinquenios)", marker=dict(size=9, color=C["suave"]),
                             hovertemplate="edad %{x:.0f}<br>observada %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["edad"], y=d["prob"], mode="lines", name="modelo (dependencia parcial)", line=dict(color=C["con"], width=3),
                             hovertemplate="edad %{x}<br>modelo %{y:.1%}<extra></extra>"))
    i = int(d["prob"].idxmin())
    fig.add_annotation(x=d.loc[i, "edad"], y=d.loc[i, "prob"], text=f"mínimo: {d.loc[i, 'prob']:.1%} a los {int(d.loc[i, 'edad'])}".replace(".", ","), showarrow=True, arrowcolor=C["suave"], ay=40, font=dict(size=11))
    fig.update_xaxes(title="edad (años)"); fig.update_yaxes(tickformat=".0%", title="probabilidad de dificultad cognitiva", rangemode="tozero")
    return _estilo(fig, "Efecto de la edad modelado con splines", 360)


def fig_subgrupos(A, modelo: str, var: str) -> go.Figure:
    t = A["subgrupos"][modelo]
    t = t[t["variable"] == var].copy()
    if t.empty:
        return fig_vacia()
    orden = ORDENES.get(var) or CATEGORIAS.get(var)
    if orden:
        t["nivel"] = pd.Categorical(t["nivel"], categories=[o for o in orden if o in set(t["nivel"])] + [o for o in t["nivel"] if o not in orden], ordered=True)
        t = t.sort_values("nivel")
    else:
        t = t.sort_values("nivel")
    t["nivel"] = t["nivel"].astype(str)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=("Prevalencia observada y predicha", "Residuo medio (observado − predicho)"), horizontal_spacing=0.14)
    fig.add_trace(go.Bar(x=t["nivel"], y=t["observado"], name="observada", marker_color=C["sin"], hovertemplate="%{x}<br>observada %{y:.1%}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=t["nivel"], y=t["predicho"], name="predicha", marker_color=C["con"], hovertemplate="%{x}<br>predicha %{y:.1%}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t["nivel"], y=t["residuo"], mode="markers", showlegend=False, marker=dict(size=11, color=[C["con"] if abs(r) > e else C["teal"] for r, e in zip(t["residuo"], t["ic95"])]),
                             error_y=dict(type="data", array=t["ic95"], color=C["suave"], thickness=1.2, width=4), customdata=np.c_[t["n"], t["roc_auc"]],
                             hovertemplate="<b>%{x}</b><br>residuo %{y:+.3f}<br>n = %{customdata[0]:,.0f}<br>ROC-AUC en el subgrupo: %{customdata[1]:.3f}<extra></extra>"), 1, 2)
    fig.add_hline(y=0, line_dash="dash", line_color=C["texto"], row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1); fig.update_yaxes(tickformat="+.2f", row=1, col=2)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.28, x=0.2, xanchor="center"))
    fig.update_annotations(font=dict(size=12, color=C["suave"]))
    return _estilo(fig, f"Diagnóstico por subgrupos · {ETIQUETAS.get(var, var.replace('_', ' ').capitalize())} · {modelo}", 400, dict(l=55, r=20, t=70, b=110))


# -- Simulador de riesgo ------------------------------------------------------
def perfil_df(perfil: dict) -> pd.DataFrame:
    return pd.DataFrame([{v: perfil[v] for v in PREDICTORAS}])


def contribuciones(A, perfil: dict) -> pd.DataFrame:
    """Cuánto cambia la probabilidad si cada factor pasara a su categoría de referencia (el resto igual)."""
    lr = A["lr"]
    base = float(lr.predict_proba(perfil_df(perfil))[0, 1])
    filas = []
    for v in PREDICTORAS:
        ref = 50.0 if v == "edad" else A["categorias"][v][0]
        alt = dict(perfil); alt[v] = ref
        p_ref = float(lr.predict_proba(perfil_df(alt))[0, 1])
        filas.append([v, ETIQUETAS[v], str(perfil[v]) if v != "edad" else f"{int(perfil[v])} años", str(ref) if v != "edad" else "50 años", base - p_ref])
    return pd.DataFrame(filas, columns=["variable", "factor", "valor", "referencia", "delta"]), base


def fig_gauge(prob: float, umbral: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix=" %", valueformat=".1f", font=dict(size=40, color=C["texto"])),
                                 gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor=C["suave"]), bar=dict(color=C["con"] if prob >= umbral else C["teal"], thickness=0.32),
                                            bgcolor="white", borderwidth=0,
                                            steps=[dict(range=[0, umbral * 100], color="#DCEBEC"), dict(range=[umbral * 100, 100], color="#F5DCD2")],
                                            threshold=dict(line=dict(color=C["texto"], width=3), thickness=0.8, value=umbral * 100))))
    return _estilo(fig, "Probabilidad estimada", 260, dict(l=25, r=25, t=50, b=10))


def fig_contrib(df_c: pd.DataFrame) -> go.Figure:
    d = df_c.reindex(df_c["delta"].abs().sort_values().index)
    fig = go.Figure(go.Bar(y=d["factor"], x=d["delta"] * 100, orientation="h", marker_color=[C["con"] if v > 0.0005 else (C["teal"] if v < -0.0005 else "#C9BDAE") for v in d["delta"]],
                           customdata=np.c_[d["valor"], d["referencia"]],
                           hovertemplate="<b>%{y}</b><br>Valor elegido: %{customdata[0]}<br>Referencia: %{customdata[1]}<br>Cambio en la probabilidad: %{x:+.1f} puntos<extra></extra>"))
    fig.add_vline(x=0, line_color=C["texto"])
    fig.update_xaxes(title="puntos porcentuales de probabilidad frente a la categoría de referencia", ticksuffix=" pp")
    return _estilo(fig, "¿Qué factores empujan el riesgo de este perfil?", 460, dict(l=10, r=20, t=50, b=60))
