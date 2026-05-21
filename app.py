import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================

st.set_page_config(
    page_title="Inteligencia Territorial",
    layout="wide"
)

# ==================================================
# CARGA DE DATOS
# ==================================================

@st.cache_data
def cargar_datos():

    df = pd.read_csv(
        "riesgo_municipal.csv"
    )

    return df

df = cargar_datos()

# ==================================================
# FILTROS LATERALES
# ==================================================

st.sidebar.title("Filtros Estratégicos")

# -------------------------
# Departamento
# -------------------------

departamentos = sorted(
    df['DEPARTAMENTO'].dropna().unique()
)

dep_sel = st.sidebar.multiselect(
    "Departamento",
    departamentos
)

# -------------------------
# Municipio
# -------------------------

municipios = sorted(
    df['MUNICIPIO'].dropna().unique()
)

mun_sel = st.sidebar.multiselect(
    "Municipio",
    municipios
)

# -------------------------
# Año
# -------------------------

anios = sorted(
    df['AÑO'].dropna().unique()
)

anio_sel = st.sidebar.multiselect(
    "Año",
    anios
)

# ==================================================
# APLICAR FILTROS
# ==================================================

df_filtrado = df.copy()

if dep_sel:

    df_filtrado = df_filtrado[
        df_filtrado['DEPARTAMENTO'].isin(dep_sel)
    ]

if mun_sel:

    df_filtrado = df_filtrado[
        df_filtrado['MUNICIPIO'].isin(mun_sel)
    ]

if anio_sel:

    df_filtrado = df_filtrado[
        df_filtrado['AÑO'].isin(anio_sel)
    ]

# ==================================================
# TÍTULO
# ==================================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

st.markdown("""
Plataforma estratégica para análisis de riesgo,
escalamiento territorial y comportamiento operacional.
""")

# ==================================================
# KPIs
# ==================================================

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

# ==================================================
# MUNICIPIOS CRÍTICOS
# ==================================================

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

# ==================================================
# EVOLUCIÓN TEMPORAL
# ==================================================

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

# ==================================================
# ESCALAMIENTO TERRITORIAL
# ==================================================

if 'IET' in df_filtrado.columns:

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

# ==================================================
# EXPOSICIÓN OPERACIONAL
# ==================================================

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

# ==================================================
# MAPA ESTRATÉGICO
# ==================================================

if 'LATITUD' in df_filtrado.columns and 'LONGITUD' in df_filtrado.columns:

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
# ==================================================
# ALERTAS TEMPRANAS
# ==================================================

st.subheader(
    "Alertas Estratégicas"
)

# --------------------------------------
# CREAR ÍNDICE PRESIÓN TERRITORIAL
# --------------------------------------

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

# --------------------------------------
# AGRUPAR MUNICIPIOS
# --------------------------------------

alertas = df_alertas.groupby(
    'MUNICIPIO'
)['IPT'].mean().reset_index()

alertas = alertas.sort_values(
    by='IPT',
    ascending=False
)

# --------------------------------------
# CLASIFICAR ALERTAS
# --------------------------------------

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

# --------------------------------------
# MOSTRAR ALERTAS
# --------------------------------------

st.dataframe(
    alertas.head(30),
    use_container_width=True
)

# --------------------------------------
# GRÁFICO ALERTAS
# --------------------------------------

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
# ==================================================
# ANÁLISIS AVANZADO GAO
# ==================================================

st.subheader(
    "Análisis Estratégico GAO"
)

# --------------------------------------
# MÉTRICAS GAO
# --------------------------------------

gao_stats = df_filtrado.groupby(
    'GAO'
).agg({

    'EVENTOS': 'sum',

    'MUNICIPIO': 'nunique',

    'ACTIVIDAD': 'nunique',

    'RIESGO': 'mean'

}).reset_index()

# RENOMBRAR

gao_stats.columns = [

    'GAO',

    'EVENTOS',

    'MUNICIPIOS_OCUPADOS',

    'DIVERSIDAD_TACTICA',

    'RIESGO_PROMEDIO'
]

# --------------------------------------
# ÍNDICE ADAPTACIÓN
# --------------------------------------

gao_stats['INDICE_ADAPTACION'] = (

    gao_stats['DIVERSIDAD_TACTICA'] * 0.6

    +

    gao_stats['MUNICIPIOS_OCUPADOS'] * 0.4
)

# --------------------------------------
# ORDENAR
# --------------------------------------

gao_stats = gao_stats.sort_values(

    by='INDICE_ADAPTACION',

    ascending=False
)

# --------------------------------------
# TABLA
# --------------------------------------

st.dataframe(
    gao_stats,
    use_container_width=True
)

# --------------------------------------
# GRÁFICO EXPANSIÓN
# --------------------------------------

fig_gao = px.bar(

    gao_stats,

    x='GAO',

    y='MUNICIPIOS_OCUPADOS',

    color='INDICE_ADAPTACION',

    title='Expansión Territorial y Adaptación GAO',

    color_continuous_scale='Viridis'
)

st.plotly_chart(
    fig_gao,
    use_container_width=True
)

# --------------------------------------
# DIVERSIDAD TÁCTICA
# --------------------------------------

fig_tactica = px.scatter(

    gao_stats,

    x='MUNICIPIOS_OCUPADOS',

    y='DIVERSIDAD_TACTICA',

    size='EVENTOS',

    color='RIESGO_PROMEDIO',

    hover_name='GAO',

    title='Capacidad Adaptativa y Expansión Territorial',

    color_continuous_scale='Reds'
)

st.plotly_chart(
    fig_tactica,
    use_container_width=True
)
# ==================================================
# TABLA FINAL
# ==================================================

st.subheader(
    "Vista Estratégica Consolidada"
)

st.dataframe(
    df_filtrado,
    use_container_width=True
)
