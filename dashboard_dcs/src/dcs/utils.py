"""
Utilidades compartidas del proyecto (copiadas del Entregable 1, con la ruta de datos ajustada):
Dificultad cognitiva subjetiva y factores de riesgo modificables de demencia (NHIS 2022-2023).

Todas las recodificaciones de este módulo provienen del codebook oficial de la NHIS
(no de los datos), por lo que pueden aplicarse antes de la partición train/test sin
generar fuga de información.
"""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]   # carpeta dashboard_dcs/
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data"   # aquí viven train.csv, test.csv y la base completa
ARCHIVOS_RAW = {2022: "adult22csv.zip", 2023: "adult23csv.zip"}

# ---------------------------------------------------------------------------
# Órdenes de las variables ordinales (de menor a mayor)
# ---------------------------------------------------------------------------
ORDENES = {
    "educacion": ["Menos que secundaria", "Secundaria o GED", "Técnico o universidad incompleta",
                  "Pregrado", "Posgrado"],
    "frecuencia_depresion": ["Nunca", "Pocas veces al año", "Mensual", "Semanal", "Diaria"],
    "dificultad_auditiva": ["Ninguna", "Alguna", "Mucha", "No puede"],
    "dificultad_visual": ["Ninguna", "Alguna", "Mucha", "No puede"],
    "imc_categoria": ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"],
    "nivel_dificultad_cognitiva": ["Ninguna", "Alguna", "Mucha", "No puede"],
}

# Categorías de las nominales/binarias; la primera es la categoría de referencia
CATEGORIAS = {
    "sexo": ["Hombre", "Mujer"],
    "raza_etnia": ["Blanco no hispano", "Hispano", "Negro no hispano", "Asiático no hispano",
                   "AIAN no hispano", "AIAN y otro grupo", "Otro / múltiple"],
    "tabaquismo": ["Nunca fumó", "Exfumador", "Fumador actual"],
    "hipertension": ["No", "Sí"],
    "colesterol_alto": ["No", "Sí"],
    "diabetes": ["No", "Sí"],
    "acv": ["No", "Sí"],
    "depresion_dx": ["No", "Sí"],
    "discapacidad": ["No", "Sí"],
    "demencia_dx": ["No", "Sí"],
}

# ---------------------------------------------------------------------------
# Diccionario de variables
# ---------------------------------------------------------------------------
_DIC = [
    # nombre, variable NHIS, rol, tipo, unidad/categorías, significado, factor Lancet 2024
    ("dificultad_cognitiva", "COGMEMDFF_A", "Objetivo", "Binaria", "0 = ninguna; 1 = alguna, mucha o no puede",
     "¿Tiene dificultad para recordar o concentrarse? (Washington Group)", "—"),
    ("edad", "AGEP_A", "Predictora", "Numérica discreta", "Años (18-84; 85 = 85 o más)",
     "Edad del adulto entrevistado", "No (confusor)"),
    ("sexo", "SEX_A", "Predictora", "Nominal binaria", "Hombre / Mujer", "Sexo del adulto", "No (confusor)"),
    ("raza_etnia", "HISPALLP_A", "Predictora", "Nominal (7)", "Grupos raciales y origen hispano",
     "Raza/etnia autodeclarada", "No (confusor)"),
    ("educacion", "EDUCP_A", "Predictora", "Ordinal (5)", "Menos que secundaria … Posgrado",
     "Máximo nivel educativo alcanzado (agrupado a partir de 10 niveles)", "Sí: baja escolaridad"),
    ("hipertension", "HYPEV_A", "Predictora", "Binaria", "Sí / No", "Alguna vez diagnosticado con hipertensión",
     "Sí: hipertensión"),
    ("colesterol_alto", "CHLEV_A", "Predictora", "Binaria", "Sí / No",
     "Alguna vez diagnosticado con colesterol alto", "Sí: colesterol LDL alto"),
    ("diabetes", "DIBEV_A", "Predictora", "Binaria", "Sí / No", "Alguna vez diagnosticado con diabetes",
     "Sí: diabetes"),
    ("acv", "STREV_A", "Predictora", "Binaria", "Sí / No", "Alguna vez le dijeron que tuvo un ACV",
     "No (exposición de interés)"),
    ("depresion_dx", "DEPEV_A", "Predictora", "Binaria", "Sí / No", "Alguna vez diagnosticado con depresión",
     "Sí: depresión"),
    ("frecuencia_depresion", "DEPFREQ_A", "Predictora", "Ordinal (5)", "Nunca … Diaria",
     "Frecuencia con que se siente deprimido", "Sí: depresión (síntomas)"),
    ("dificultad_auditiva", "HEARINGDF_A", "Predictora", "Ordinal (4)", "Ninguna … No puede",
     "Dificultad para oír, aun con audífonos", "Sí: pérdida auditiva"),
    ("dificultad_visual", "VISIONDF_A", "Predictora", "Ordinal (4)", "Ninguna … No puede",
     "Dificultad para ver, aun con gafas", "Sí: pérdida visual"),
    ("tabaquismo", "SMKCIGST_A", "Predictora", "Nominal (3)", "Nunca / Exfumador / Actual",
     "Estado de tabaquismo de cigarrillo", "Sí: tabaquismo"),
    ("imc_categoria", "BMICAT_A", "Predictora", "Ordinal (4)", "Bajo peso / Normal / Sobrepeso / Obesidad",
     "Categoría de índice de masa corporal", "Sí: obesidad"),
    ("imc", "WEIGHTLBTC_A, HEIGHTTC_A", "Solo EDA", "Numérica continua", "kg/m²",
     "IMC calculado con peso (lb) y talla (in) autorreportados", "Sí: obesidad"),
    ("nivel_dificultad_cognitiva", "COGMEMDFF_A", "Solo EDA", "Ordinal (4)", "Ninguna … No puede",
     "Nivel original de la variable objetivo", "—"),
    ("frecuencia_dificultad_cog", "COGFRQDFF_A", "Auditoría de fuga", "Ordinal", "Nunca … Siempre / No aplica",
     "Con qué frecuencia tiene dificultad para recordar (pregunta de seguimiento del objetivo)", "—"),
    ("discapacidad", "DISAB3_A", "Auditoría de fuga", "Binaria", "Sí / No",
     "Indicador de discapacidad del Washington Group (se construye con la pregunta de cognición)", "—"),
    ("demencia_dx", "DEMENEV_A", "Auditoría de fuga", "Binaria", "Sí / No",
     "Alguna vez diagnosticado con demencia", "—"),
    ("anio", "SRVY_YR", "Estratificación", "Discreta", "2022 / 2023", "Año de la encuesta", "—"),
    ("trimestre", "INTV_QRT", "Solo EDA", "Discreta", "1 a 4", "Trimestre de la entrevista", "—"),
    ("peso_muestral", "WTFA_A", "Solo EDA", "Numérica", "Personas representadas",
     "Peso muestral final anual (solo para evaluar representatividad)", "—"),
    ("id_hogar", "HHX", "Identificador", "Texto", "—", "Identificador aleatorio del hogar", "—"),
]
DICCIONARIO = pd.DataFrame(_DIC, columns=["variable", "variable_nhis", "rol", "tipo",
                                          "unidad_categorias", "significado", "factor_lancet_2024"])

PREDICTORAS_NUM = ["edad"]
PREDICTORAS_CAT = ["sexo", "raza_etnia", "educacion", "hipertension", "colesterol_alto", "diabetes", "acv",
                   "depresion_dx", "frecuencia_depresion", "dificultad_auditiva", "dificultad_visual",
                   "tabaquismo", "imc_categoria"]
PREDICTORAS = PREDICTORAS_NUM + PREDICTORAS_CAT
OBJETIVO = "dificultad_cognitiva"
SOSPECHOSAS = ["frecuencia_dificultad_cog", "discapacidad", "demencia_dx"]

_COLS_NHIS = ["HHX", "SRVY_YR", "INTV_QRT", "WTFA_A", "AVAIL_A", "PROXY_A", "COGMEMDFF_A", "COGFRQDFF_A",
              "AGEP_A", "SEX_A", "HISPALLP_A", "EDUCP_A", "HYPEV_A", "CHLEV_A", "DIBEV_A", "STREV_A",
              "DEPEV_A", "DEPFREQ_A", "HEARINGDF_A", "VISIONDF_A", "SMKCIGST_A", "BMICAT_A",
              "WEIGHTLBTC_A", "HEIGHTTC_A", "DISAB3_A", "DEMENEV_A"]


def cargar_crudos(dir_raw=DIR_RAW) -> pd.DataFrame:
    """Lee los CSV de adultos 2022 y 2023 directamente desde los .zip del CDC."""
    partes = []
    for anio, nombre in ARCHIVOS_RAW.items():
        ruta = Path(dir_raw) / nombre
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}. Descárguelo del sitio del CDC y colóquelo en data/raw/.")
        with zipfile.ZipFile(ruta) as z:
            csv = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(csv) as f:
                df = pd.read_csv(f, usecols=_COLS_NHIS, low_memory=False)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def _si_no(s):
    return s.map({1: "Sí", 2: "No"})


def construir_dataset(crudo: pd.DataFrame):
    """Aplica los criterios de inclusión y recodifica según el codebook.

    Devuelve (dataset, flujo), donde flujo registra cuántos registros quedan en cada paso.
    """
    flujo = [("Adultos en los archivos 2022 + 2023", len(crudo))]
    proxy = (crudo["AVAIL_A"] == 3) & (crudo["PROXY_A"] == 1)
    d = crudo.loc[~proxy].copy()
    flujo.append(("Excluye entrevistas respondidas por un informante (proxy)", len(d)))
    d = d.loc[d["COGMEMDFF_A"].isin([1, 2, 3, 4])].copy()
    flujo.append(("Excluye objetivo con código de no respuesta (7 = se negó, 9 = no sabe)", len(d)))

    out = pd.DataFrame(index=d.index)
    out["id_hogar"] = d["HHX"]
    out["anio"] = d["SRVY_YR"].astype(int)
    out["trimestre"] = d["INTV_QRT"].astype(int)
    out["peso_muestral"] = d["WTFA_A"].astype(float)

    # Objetivo
    niv = {1: "Ninguna", 2: "Alguna", 3: "Mucha", 4: "No puede"}
    out["nivel_dificultad_cognitiva"] = d["COGMEMDFF_A"].map(niv)
    out[OBJETIVO] = (d["COGMEMDFF_A"] > 1).astype(int)

    # Edad: 85 = 85 o más (truncada para uso público); 97/98/99 = no respuesta
    out["edad"] = d["AGEP_A"].where(d["AGEP_A"] <= 85).astype(float)
    out["sexo"] = d["SEX_A"].map({1: "Hombre", 2: "Mujer"})
    out["raza_etnia"] = d["HISPALLP_A"].map({1: "Hispano", 2: "Blanco no hispano", 3: "Negro no hispano",
                                             4: "Asiático no hispano", 5: "AIAN no hispano",
                                             6: "AIAN y otro grupo", 7: "Otro / múltiple"})
    edu = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 4, 10: 4}
    out["educacion"] = d["EDUCP_A"].map(edu).map(dict(enumerate(ORDENES["educacion"])))
    for nuevo, orig in [("hipertension", "HYPEV_A"), ("colesterol_alto", "CHLEV_A"), ("diabetes", "DIBEV_A"),
                        ("acv", "STREV_A"), ("depresion_dx", "DEPEV_A"), ("discapacidad", "DISAB3_A"),
                        ("demencia_dx", "DEMENEV_A")]:
        out[nuevo] = _si_no(d[orig])
    out["frecuencia_depresion"] = d["DEPFREQ_A"].map({5: "Nunca", 4: "Pocas veces al año", 3: "Mensual",
                                                      2: "Semanal", 1: "Diaria"})
    for nuevo, orig in [("dificultad_auditiva", "HEARINGDF_A"), ("dificultad_visual", "VISIONDF_A")]:
        out[nuevo] = d[orig].map(niv)
    # 5 = fumador con estado actual desconocido y 9 = no sabe -> faltante
    out["tabaquismo"] = d["SMKCIGST_A"].map({1: "Fumador actual", 2: "Fumador actual", 3: "Exfumador",
                                             4: "Nunca fumó"})
    out["imc_categoria"] = d["BMICAT_A"].map({1: "Bajo peso", 2: "Normal", 3: "Sobrepeso", 4: "Obesidad"})

    # IMC numérico (solo EDA). 996/96 = valor suprimido por excepcionalmente bajo o alto
    peso = d["WEIGHTLBTC_A"].where(d["WEIGHTLBTC_A"].between(100, 299))
    talla = d["HEIGHTTC_A"].where(d["HEIGHTTC_A"].between(59, 76))
    out["imc"] = (703 * peso / talla ** 2).round(2)
    out["peso_talla_suprimidos"] = ((d["WEIGHTLBTC_A"] == 996) | (d["HEIGHTTC_A"] == 96)).astype(int)

    # Pregunta de seguimiento del objetivo (solo se hace si COGMEMDFF_A > 1)
    frq = {1: "Nunca", 2: "Pocas veces", 3: "Algunas veces", 4: "Muchas veces", 5: "Siempre"}
    out["frecuencia_dificultad_cog"] = d["COGFRQDFF_A"].map(frq)
    out.loc[d["COGFRQDFF_A"].isna(), "frecuencia_dificultad_cog"] = "No aplica"

    out = out.reset_index(drop=True)
    return out, pd.DataFrame(flujo, columns=["paso", "registros"])


def dividir(df: pd.DataFrame, test_size: float = 0.20, seed: int = SEED):
    """Reserva del conjunto de prueba: partición estratificada por objetivo y año."""
    estrato = df[OBJETIVO].astype(str) + "_" + df["anio"].astype(str)
    train, test = train_test_split(df, test_size=test_size, stratify=estrato, random_state=seed)
    return train.copy(), test.copy()


def cargar_particiones():
    """Carga las particiones guardadas por el notebook 01."""
    train = pd.read_csv(DIR_PROC / "train.csv")
    test = pd.read_csv(DIR_PROC / "test.csv")
    return train, test


def a_ordinal(serie: pd.Series, variable: str) -> pd.Series:
    """Convierte una variable con etiquetas en su posición numérica (0, 1, 2, ...)."""
    orden = ORDENES.get(variable) or CATEGORIAS.get(variable)
    return serie.map({c: i for i, c in enumerate(orden)})


def cramers_v(tabla) -> float:
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    from scipy.stats import chi2_contingency
    tabla = np.asarray(tabla)
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1))))


def estilo_graficos():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 11})
