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
# TABLA FINAL
# ==================================================

st.subheader(
    "Vista Estratégica Consolidada"
)

st.dataframe(
    df_filtrado,
    use_container_width=True
)
