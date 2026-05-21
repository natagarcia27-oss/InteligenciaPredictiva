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
# FUNCIÓN SEGURA CSV
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

    st.error(
        f"No fue posible leer el archivo: {ruta}"
    )

    return pd.DataFrame()

# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data
def cargar_datos():

    return leer_csv_seguro(
        "riesgo_municipal.csv"
    )

@st.cache_data
def cargar_operacional():

    return leer_csv_seguro(
        "eventos_operacionales.csv"
    )

# =========================================================
# CARGAR BASES
# =========================================================

df = cargar_datos()

df_op = cargar_operacional()

# =========================================================
# VALIDAR
# =========================================================

if df.empty:

    st.error(
        "No se pudo cargar riesgo_municipal.csv"
    )

    st.stop()

if df_op.empty:

    st.warning(
        "No se pudo cargar eventos_operacionales.csv"
    )

# =========================================================
# NORMALIZAR COLUMNAS
# =========================================================

df.columns = df.columns.str.strip()

df.columns = [

    col.upper().replace(" ", "_")

    for col in df.columns
]

if not df_op.empty:

    df_op.columns = df_op.columns.str.strip()

    df_op.columns = [

        col.upper().replace(" ", "_")

        for col in df_op.columns
    ]

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

if not df_op.empty:

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
# FILTRAR DATASET
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
Plataforma estratégica para análisis predictivo,
riesgo territorial e inteligencia operacional.
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

    title='Top 20 Municipios Críticos'
)

st.plotly_chart(
    fig1,
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
# ALERTAS TEMPRANAS
# =========================================================

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
# INTELIGENCIA ORGANIZACIONAL
# =========================================================

if not df_op.empty:

    st.subheader(
        "Inteligencia Organizacional"
    )

    st.write(
        "Columnas detectadas en base operacional:"
    )

    st.write(
        df_op.columns.tolist()
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
