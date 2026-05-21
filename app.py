import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Inteligencia Territorial",
    layout="wide"
)

# =========================================================
# FUNCIÓN LECTURA SEGURA CSV
# =========================================================

def leer_csv_seguro(ruta):

    codificaciones = [

        'utf-8',

        'latin1',

        'cp1252',

        'ISO-8859-1'
    ]

    for cod in codificaciones:

        try:

            df = pd.read_csv(
                ruta,
                encoding=cod
            )

            return df

        except Exception:

            continue

    return pd.DataFrame()

# =========================================================
# LIMPIEZA COLUMNAS
# =========================================================

def limpiar_columnas(df):

    # quitar espacios

    df.columns = df.columns.str.strip()

    # reparar caracteres dañados

    reemplazos = {

        'Ã‘':'Ñ',
        'Ã“':'Ó',
        'Ã‰':'É',
        'Ã':'Í',
        'Ã':'Á',
        'Ãš':'Ú'
    }

    nuevas = []

    for col in df.columns:

        for malo, bueno in reemplazos.items():

            col = col.replace(
                malo,
                bueno
            )

        col = (
            col.upper()
            .replace(" ", "_")
        )

        nuevas.append(col)

    df.columns = nuevas

    return df

# =========================================================
# CARGAR BASES
# =========================================================

@st.cache_data
def cargar_predictiva():

    df = leer_csv_seguro(
        "riesgo_municipal.csv"
    )

    return limpiar_columnas(df)

@st.cache_data
def cargar_operacional():

    df = leer_csv_seguro(
        "eventos_operacionales.csv"
    )

    return limpiar_columnas(df)

df = cargar_predictiva()

df_op = cargar_operacional()

# =========================================================
# VALIDACIÓN
# =========================================================

if df.empty:

    st.error(
        "No se pudo cargar riesgo_municipal.csv"
    )

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Filtros Estratégicos"
)

with st.sidebar.expander(
    "Columnas Base Predictiva"
):

    st.write(df.columns.tolist())

with st.sidebar.expander(
    "Columnas Base Operacional"
):

    st.write(df_op.columns.tolist())

# =========================================================
# FILTROS
# =========================================================

dep_sel = []
mun_sel = []
anio_sel = []

if 'DEPARTAMENTO' in df.columns:

    dep_sel = st.sidebar.multiselect(

        "Departamento",

        sorted(
            df['DEPARTAMENTO']
            .dropna()
            .unique()
        )
    )

if 'MUNICIPIO' in df.columns:

    mun_sel = st.sidebar.multiselect(

        "Municipio",

        sorted(
            df['MUNICIPIO']
            .dropna()
            .unique()
        )
    )

if 'AÑO' in df.columns:

    anio_sel = st.sidebar.multiselect(

        "Año",

        sorted(
            df['AÑO']
            .dropna()
            .unique()
        )
    )

# =========================================================
# FILTRAR
# =========================================================

df_filtrado = df.copy()

if dep_sel:

    df_filtrado = df_filtrado[
        df_filtrado['DEPARTAMENTO']
        .isin(dep_sel)
    ]

if mun_sel:

    df_filtrado = df_filtrado[
        df_filtrado['MUNICIPIO']
        .isin(mun_sel)
    ]

if anio_sel:

    df_filtrado = df_filtrado[
        df_filtrado['AÑO']
        .isin(anio_sel)
    ]

# =========================================================
# TÍTULO
# =========================================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

st.markdown("""
Plataforma estratégica para análisis territorial,
inteligencia operacional y prospectiva avanzada.
""")

# =========================================================
# KPIs
# =========================================================

st.subheader(
    "Indicadores Estratégicos"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Municipios",
    df_filtrado['MUNICIPIO'].nunique()
)

col2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

col3.metric(
    "Riesgo Promedio",
    round(df_filtrado['RIESGO'].mean(), 2)
)

col4.metric(
    "GAO Presentes",
    int(df_filtrado['GAO_PRESENTES'].mean())
)

# =========================================================
# MUNICIPIOS CRÍTICOS
# =========================================================

st.subheader(
    "Municipios de Mayor Riesgo"
)

top = df_filtrado.groupby(
    'MUNICIPIO'
)['RIESGO'].mean().reset_index()

top = top.sort_values(
    by='RIESGO',
    ascending=False
).head(20)

fig1 = px.bar(

    top,

    x='MUNICIPIO',

    y='RIESGO',

    color='RIESGO',

    color_continuous_scale='Reds',

    title='Top Municipios Críticos'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =========================================================
# MAPA ESTRATÉGICO
# =========================================================

st.subheader(
    "Mapa Estratégico Territorial"
)

mapa = df_filtrado.groupby(
    [
        'MUNICIPIO',
        'LATITUD',
        'LONGITUD'
    ]
).agg({

    'RIESGO':'mean',

    'EVENTOS':'sum'

}).reset_index()

fig_map = px.scatter_mapbox(

    mapa,

    lat='LATITUD',

    lon='LONGITUD',

    size='EVENTOS',

    color='RIESGO',

    hover_name='MUNICIPIO',

    zoom=4,

    height=700,

    color_continuous_scale='Reds',

    size_max=30
)

fig_map.update_layout(

    mapbox_style='carto-darkmatter',

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    )
)

st.plotly_chart(
    fig_map,
    use_container_width=True
)

# =========================================================
# ALERTAS
# =========================================================

st.subheader(
    "Alertas Estratégicas"
)

df_alertas = df_filtrado.copy()

df_alertas['IPT'] = (

    df_alertas['RIESGO'] * 0.5

    +

    df_alertas['IET'] * 0.3

    +

    df_alertas['EVENTOS'] * 0.2
)

alertas = df_alertas.groupby(
    'MUNICIPIO'
)['IPT'].mean().reset_index()

alertas = alertas.sort_values(
    by='IPT',
    ascending=False
)

fig_alertas = px.bar(

    alertas.head(20),

    x='MUNICIPIO',

    y='IPT',

    color='IPT',

    color_continuous_scale='Reds',

    title='Presión Territorial'
)

st.plotly_chart(
    fig_alertas,
    use_container_width=True
)

# =========================================================
# DETECTAR COLUMNA ORGANIZACIONAL
# =========================================================

st.subheader(
    "Inteligencia Organizacional"
)

columna_gao = None

candidatas = [

    'GAO',

    'FG/BLOQUE/ESTRUCTURA',

    'FG_BLOQUE_ESTRUCTURA',

    'ESTRUCTURA',

    'SUBESTRUCTURA'
]

for col in candidatas:

    if col in df_op.columns:

        columna_gao = col

        break

# =========================================================
# INTELIGENCIA ORGANIZACIONAL
# =========================================================

if columna_gao:

    st.success(
        f"Columna organizacional detectada: {columna_gao}"
    )

    gao_intel = df_op.groupby(
        columna_gao
    ).agg({

        'MUNICIPIO':'nunique',

        'ACTIVIDAD':'nunique'

    }).reset_index()

    gao_intel.columns = [

        'ORGANIZACION',

        'EXPANSION_TERRITORIAL',

        'DIVERSIDAD_TACTICA'
    ]

    gao_intel['INDICE_ADAPTACION'] = (

        gao_intel['EXPANSION_TERRITORIAL'] * 0.5

        +

        gao_intel['DIVERSIDAD_TACTICA'] * 0.5
    )

    gao_intel = gao_intel.sort_values(

        by='INDICE_ADAPTACION',

        ascending=False
    )

    st.dataframe(
        gao_intel,
        use_container_width=True
    )

    fig_org = px.scatter(

        gao_intel,

        x='EXPANSION_TERRITORIAL',

        y='DIVERSIDAD_TACTICA',

        size='INDICE_ADAPTACION',

        color='INDICE_ADAPTACION',

        hover_name='ORGANIZACION',

        title='Adaptación y Expansión Organizacional',

        color_continuous_scale='Reds'
    )

    st.plotly_chart(
        fig_org,
        use_container_width=True
    )

else:

    st.warning(
        "No se detectó columna organizacional"
    )

# =========================================================
# TABLA FINAL
# =========================================================

st.subheader(
    "Vista Estratégica Consolidada"
)

st.dataframe(
    df_filtrado,
    use_container_width=True
)
