import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Inteligencia Territorial",
    layout="wide"
)

# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data
def cargar_datos():

    df = pd.read_csv(
        "riesgo_municipal.csv"
    )

    return df

@st.cache_data
def cargar_operacional():

    df_op = pd.read_csv(
        "eventos_operacionales.csv"
    )

    return df_op

# =========================================================
# CARGAR DATASETS
# =========================================================

df = cargar_datos()
df_op = cargar_operacional()

# =========================================================
# NORMALIZAR COLUMNAS
# =========================================================

df.columns = df.columns.str.strip()

df.columns = [
    col.upper().replace(" ", "_")
    for col in df.columns
]

df_op.columns = df_op.columns.str.strip()

df_op.columns = [
    col.upper().replace(" ", "_")
    for col in df_op.columns
]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Filtros Estratégicos")

with st.sidebar.expander("Columnas Base Predictiva"):

    st.write(df.columns.tolist())

with st.sidebar.expander("Columnas Base Operacional"):

    st.write(df_op.columns.tolist())

# =========================================================
# FILTROS BASE PREDICTIVA
# =========================================================

# -----------------------------------------
# DEPARTAMENTO
# -----------------------------------------

if 'DEPARTAMENTO' in df.columns:

    departamentos = sorted(
        df['DEPARTAMENTO'].dropna().unique()
    )

    dep_sel = st.sidebar.multiselect(
        "Departamento",
        departamentos
    )

else:

    dep_sel = []

# -----------------------------------------
# MUNICIPIO
# -----------------------------------------

if 'MUNICIPIO' in df.columns:

    municipios = sorted(
        df['MUNICIPIO'].dropna().unique()
    )

    mun_sel = st.sidebar.multiselect(
        "Municipio",
        municipios
    )

else:

    mun_sel = []

# -----------------------------------------
# AÑO
# -----------------------------------------

if 'AÑO' in df.columns:

    anios = sorted(
        df['AÑO'].dropna().unique()
    )

    anio_sel = st.sidebar.multiselect(
        "Año",
        anios
    )

else:

    anio_sel = []

# =========================================================
# APLICAR FILTROS
# =========================================================

df_filtrado = df.copy()

if dep_sel and 'DEPARTAMENTO' in df_filtrado.columns:

    df_filtrado = df_filtrado[
        df_filtrado['DEPARTAMENTO'].isin(dep_sel)
    ]

if mun_sel and 'MUNICIPIO' in df_filtrado.columns:

    df_filtrado = df_filtrado[
        df_filtrado['MUNICIPIO'].isin(mun_sel)
    ]

if anio_sel and 'AÑO' in df_filtrado.columns:

    df_filtrado = df_filtrado[
        df_filtrado['AÑO'].isin(anio_sel)
    ]

# =========================================================
# TÍTULO
# =========================================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

st.markdown("""
Plataforma estratégica para análisis predictivo,
riesgo territorial, inteligencia operacional
y prospectiva de amenazas.
""")

# =========================================================
# KPIs
# =========================================================

st.subheader("Indicadores Estratégicos")

col1, col2, col3, col4 = st.columns(4)

if 'MUNICIPIO' in df_filtrado.columns:

    col1.metric(
        "Municipios",
        df_filtrado['MUNICIPIO'].nunique()
    )

if 'EVENTOS' in df_filtrado.columns:

    col2.metric(
        "Eventos",
        int(df_filtrado['EVENTOS'].sum())
    )

if 'RIESGO' in df_filtrado.columns:

    col3.metric(
        "Riesgo Promedio",
        round(df_filtrado['RIESGO'].mean(), 2)
    )

if 'GAO_PRESENTES' in df_filtrado.columns:

    col4.metric(
        "GAO Presentes",
        int(df_filtrado['GAO_PRESENTES'].mean())
    )

# =========================================================
# MUNICIPIOS CRÍTICOS
# =========================================================

if (
    'MUNICIPIO' in df_filtrado.columns
    and
    'RIESGO' in df_filtrado.columns
):

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

        title='Top 20 Municipios Críticos'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =========================================================
# EVOLUCIÓN TEMPORAL
# =========================================================

if (
    'AÑO' in df_filtrado.columns
    and
    'SEMANA' in df_filtrado.columns
):

    st.subheader(
        "Evolución Temporal"
    )

    temporal = df_filtrado.groupby(
        ['AÑO', 'SEMANA']
    ).size().reset_index(name='EVENTOS')

    temporal['PERIODO'] = (

        temporal['AÑO'].astype(str)
        + '-'
        + temporal['SEMANA'].astype(str)
    )

    fig2 = px.line(

        temporal,

        x='PERIODO',

        y='EVENTOS',

        markers=True,

        title='Comportamiento Temporal'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =========================================================
# ESCALAMIENTO TERRITORIAL
# =========================================================

if (
    'IET' in df_filtrado.columns
    and
    'MUNICIPIO' in df_filtrado.columns
):

    st.subheader(
        "Escalamiento Territorial"
    )

    esc = df_filtrado.groupby(
        'MUNICIPIO'
    )['IET'].mean().reset_index()

    esc = esc.sort_values(
        by='IET',
        ascending=False
    ).head(20)

    fig3 = px.bar(

        esc,

        x='MUNICIPIO',

        y='IET',

        color='IET',

        color_continuous_scale='Oranges',

        title='Índice Escalamiento Territorial'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================================================
# EXPOSICIÓN OPERACIONAL
# =========================================================

if 'UNIDADES_EXPUESTAS' in df_filtrado.columns:

    st.subheader(
        "Exposición Operacional"
    )

    expo = df_filtrado.groupby(
        'MUNICIPIO'
    )['UNIDADES_EXPUESTAS'].mean().reset_index()

    expo = expo.sort_values(
        by='UNIDADES_EXPUESTAS',
        ascending=False
    ).head(20)

    fig4 = px.bar(

        expo,

        x='MUNICIPIO',

        y='UNIDADES_EXPUESTAS',

        color='UNIDADES_EXPUESTAS',

        color_continuous_scale='Blues',

        title='Municipios con Mayor Exposición'
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =========================================================
# MAPA ESTRATÉGICO
# =========================================================

if (
    'LATITUD' in df_filtrado.columns
    and
    'LONGITUD' in df_filtrado.columns
):

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

        'RIESGO': 'mean',

        'EVENTOS': 'sum'

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
# ALERTAS TEMPRANAS
# =========================================================

if 'RIESGO' in df_filtrado.columns:

    st.subheader(
        "Alertas Estratégicas"
    )

    df_alertas = df_filtrado.copy()

    if 'IET' in df_alertas.columns:

        df_alertas['IPT'] = (

            df_alertas['RIESGO'] * 0.5

            +

            df_alertas['IET'] * 0.3

            +

            df_alertas['EVENTOS'] * 0.2
        )

    else:

        df_alertas['IPT'] = (

            df_alertas['RIESGO'] * 0.7

            +

            df_alertas['EVENTOS'] * 0.3
        )

    alertas = df_alertas.groupby(
        'MUNICIPIO'
    )['IPT'].mean().reset_index()

    alertas = alertas.sort_values(
        by='IPT',
        ascending=False
    )

    def clasificar_alerta(valor):

        if valor >= 15:

            return "🔴 Crítico"

        elif valor >= 8:

            return "🟠 Alto"

        elif valor >= 4:

            return "🟡 Atención"

        else:

            return "🟢 Estable"

    alertas['NIVEL_ALERTA'] = alertas[
        'IPT'
    ].apply(clasificar_alerta)

    st.dataframe(
        alertas.head(30),
        use_container_width=True
    )

    fig_alertas = px.bar(

        alertas.head(20),

        x='MUNICIPIO',

        y='IPT',

        color='NIVEL_ALERTA',

        title='Municipios Bajo Mayor Presión Territorial'
    )

    st.plotly_chart(
        fig_alertas,
        use_container_width=True
    )

# =========================================================
# INTELIGENCIA ORGANIZACIONAL
# =========================================================

st.subheader(
    "Inteligencia Organizacional"
)

# ---------------------------------------------------------
# DETECTAR COLUMNA ORGANIZACIONAL
# ---------------------------------------------------------

columna_gao = None

candidatas = [

    'GAO',

    'FG/BLOQUE/ESTRUCTURA',

    'FG_BLOQUE_ESTRUCTURA',

    'ESTRUCTURA',

    'SUBESTRUCTURA'
]

for col in candidatas:

    col_normalizada = col.upper().replace(" ", "_")

    if col_normalizada in df_op.columns:

        columna_gao = col_normalizada

        break

# ---------------------------------------------------------
# VALIDAR
# ---------------------------------------------------------

if columna_gao is not None:

    st.success(
        f"Columna organizacional detectada: {columna_gao}"
    )

    # ---------------------------------------------
    # MÉTRICAS ORGANIZACIONALES
    # ---------------------------------------------

    columnas_agg = {}

    if 'MUNICIPIO' in df_op.columns:
        columnas_agg['MUNICIPIO'] = 'nunique'

    if 'ACTIVIDAD' in df_op.columns:
        columnas_agg['ACTIVIDAD'] = 'nunique'

    gao_intel = df_op.groupby(
        columna_gao
    ).agg(columnas_agg).reset_index()

    # ---------------------------------------------
    # RENOMBRAR
    # ---------------------------------------------

    nombres = [columna_gao]

    if 'MUNICIPIO' in columnas_agg:
        nombres.append('EXPANSION_TERRITORIAL')

    if 'ACTIVIDAD' in columnas_agg:
        nombres.append('DIVERSIDAD_TACTICA')

    gao_intel.columns = nombres

    # ---------------------------------------------
    # ÍNDICE ADAPTACIÓN
    # ---------------------------------------------

    if (
        'EXPANSION_TERRITORIAL' in gao_intel.columns
        and
        'DIVERSIDAD_TACTICA' in gao_intel.columns
    ):

        gao_intel['INDICE_ADAPTACION'] = (

            gao_intel['EXPANSION_TERRITORIAL'] * 0.5

            +

            gao_intel['DIVERSIDAD_TACTICA'] * 0.5
        )

    elif 'EXPANSION_TERRITORIAL' in gao_intel.columns:

        gao_intel['INDICE_ADAPTACION'] = (
            gao_intel['EXPANSION_TERRITORIAL']
        )

    else:

        gao_intel['INDICE_ADAPTACION'] = 1

    # ---------------------------------------------
    # ORDENAR
    # ---------------------------------------------

    gao_intel = gao_intel.sort_values(

        by='INDICE_ADAPTACION',

        ascending=False
    )

    # ---------------------------------------------
    # TABLA
    # ---------------------------------------------

    st.dataframe(
        gao_intel,
        use_container_width=True
    )

    # ---------------------------------------------
    # GRÁFICO
    # ---------------------------------------------

    if (
        'EXPANSION_TERRITORIAL' in gao_intel.columns
        and
        'DIVERSIDAD_TACTICA' in gao_intel.columns
    ):

        fig_org = px.scatter(

            gao_intel,

            x='EXPANSION_TERRITORIAL',

            y='DIVERSIDAD_TACTICA',

            size='INDICE_ADAPTACION',

            color='INDICE_ADAPTACION',

            hover_name=columna_gao,

            title='Adaptación y Expansión Organizacional',

            color_continuous_scale='Reds'
        )

        st.plotly_chart(
            fig_org,
            use_container_width=True
        )

else:

    st.warning(
        "No se detectó columna organizacional válida"
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
