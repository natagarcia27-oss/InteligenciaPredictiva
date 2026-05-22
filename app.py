# =========================================================
# CENTRO DE FUSIÓN GEOESPACIAL E INTELIGENCIA TERRITORIAL
# VERSION ENTERPRISE ESTABLE FINAL
# CORRECCION GLOBAL:
# - EXPORTACION EXCEL
# - RECUPERACION GRAFICOS
# - RECUPERACION MAPAS
# - ESTABILIDAD STREAMLIT CLOUD
# - VALIDACION COLUMNAS
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans

from xgboost import XGBClassifier

import networkx as nx

from io import BytesIO

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(

    page_title="Fusion Territorial Estratégica",
    layout="wide"

)

# =========================================================
# ESTILO
# =========================================================

st.markdown("""

<style>

.main {
    background-color: #0d1117;
    color: white;
}

h1,h2,h3 {
    color: #58a6ff;
}

[data-testid="stMetricValue"] {
    color: #58a6ff;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# FUNCIONES
# =========================================================

def leer_csv_seguro(ruta):

    codificaciones = [

        "utf-8",
        "latin1",
        "cp1252",
        "ISO-8859-1"
    ]

    for cod in codificaciones:

        try:

            return pd.read_csv(
                ruta,
                encoding=cod
            )

        except:
            pass

    return pd.DataFrame()

def limpiar_columnas(df):

    nuevas = []

    for col in df.columns:

        nuevas.append(

            str(col)
            .strip()
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

    df.columns = nuevas

    return df

def existe(df_temp, col):

    return col in df_temp.columns

def detectar_columna(lista, dataframe):

    for col in lista:

        if col in dataframe.columns:

            return col

    return None

# =========================================================
# CARGA DATOS
# =========================================================

@st.cache_data
def cargar_predictiva():

    return limpiar_columnas(

        leer_csv_seguro(
            "riesgo_municipal.csv"
        )
    )

@st.cache_data
def cargar_operacional():

    return limpiar_columnas(

        leer_csv_seguro(
            "eventos_operacionales.csv"
        )
    )

df = cargar_predictiva()
df_op = cargar_operacional()

# =========================================================
# VALIDACION
# =========================================================

if df.empty:

    st.error(
        "No se pudo cargar riesgo_municipal.csv"
    )

    st.stop()

# =========================================================
# DETECCION COLUMNAS
# =========================================================

COL_GAO = detectar_columna(

    [
        "GAO",
        "FG/BLOQUE/ESTRUCTURA",
        "FG_BLOQUE_ESTRUCTURA",
        "ESTRUCTURA"
    ],

    df_op
)

# =========================================================
# FILTROS
# =========================================================

st.sidebar.title(
    "Filtros Estratégicos"
)

dep_sel = []
mun_sel = []
anio_sel = []

if existe(df, 'DEPARTAMENTO'):

    dep_sel = st.sidebar.multiselect(

        "Departamento",

        sorted(df['DEPARTAMENTO'].dropna().unique())
    )

if existe(df, 'MUNICIPIO'):

    mun_sel = st.sidebar.multiselect(

        "Municipio",

        sorted(df['MUNICIPIO'].dropna().unique())
    )

if existe(df, 'AÑO'):

    anio_sel = st.sidebar.multiselect(

        "Año",

        sorted(df['AÑO'].dropna().unique())
    )

# =========================================================
# FILTRADO
# =========================================================

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

# =========================================================
# CONVERSION NUMERICA
# =========================================================

numericas = [

    'RIESGO',
    'IET',
    'EVENTOS',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS',
    'FREQ_HISTORICA',
    'DIV_TACTICA',
    'EVENTO_FUTURO'
]

for col in numericas:

    if existe(df_filtrado, col):

        df_filtrado[col] = pd.to_numeric(

            df_filtrado[col],
            errors='coerce'

        ).fillna(0)

# =========================================================
# INDICES
# =========================================================

df_filtrado['IPE'] = (

    df_filtrado['RIESGO'] * 0.30 +

    df_filtrado['IET'] * 0.25 +

    df_filtrado['EVENTOS'] * 0.20 +

    df_filtrado['UNIDADES_EXPUESTAS'] * 0.15 +

    df_filtrado['GAO_PRESENTES'] * 0.10
)

df_filtrado['IPO'] = (

    df_filtrado['EVENTOS'] * 0.35 +

    df_filtrado['GAO_PRESENTES'] * 0.25 +

    df_filtrado['UNIDADES_EXPUESTAS'] * 0.20 +

    df_filtrado['RIESGO'] * 0.20
)

df_filtrado['IGE'] = (

    df_filtrado['IPE'] * 0.60 +

    df_filtrado['IPO'] * 0.40
)

df_filtrado['RIESGO_EVOLUTIVO'] = (

    df_filtrado['IGE'] * 0.50 +

    df_filtrado['EVENTOS'] * 0.30 +

    df_filtrado['GAO_PRESENTES'] * 0.20
)

df_filtrado['PRIORIDAD_ESTRATEGICA'] = (

    df_filtrado['RIESGO_EVOLUTIVO'] * 0.5 +

    df_filtrado['UNIDADES_EXPUESTAS'] * 0.3 +

    df_filtrado['GAO_PRESENTES'] * 0.2
)

# =========================================================
# ALERTAS
# =========================================================

def alerta(v):

    if v >= 40:
        return "CRITICO"

    elif v >= 30:
        return "ALTO"

    elif v >= 20:
        return "MEDIO"

    else:
        return "BAJO"

df_filtrado['ALERTA'] = df_filtrado[
    'PRIORIDAD_ESTRATEGICA'
].apply(alerta)

# =========================================================
# TITULO
# =========================================================

st.title(
    "Centro de Fusión Geoespacial e Inteligencia Territorial"
)

# =========================================================
# KPIS
# =========================================================

st.subheader(
    "Indicadores Estratégicos"
)

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)

c1.metric(
    "Municipios",
    int(df_filtrado['MUNICIPIO'].nunique())
)

c2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

c3.metric(
    "IGE",
    round(df_filtrado['IGE'].mean(),2)
)

c4.metric(
    "Riesgo Evolutivo",
    round(df_filtrado['RIESGO_EVOLUTIVO'].mean(),2)
)

c5.metric(
    "Prioridad Estratégica",
    round(df_filtrado['PRIORIDAD_ESTRATEGICA'].mean(),2)
)

c6.metric(
    "Alertas Criticas",
    int(
        (df_filtrado['ALERTA'] == 'CRITICO').sum()
    )
)

c7.metric(
    "GAO Promedio",
    round(df_filtrado['GAO_PRESENTES'].mean(),2)
)

c8.metric(
    "Registros",
    len(df_filtrado)
)

# =========================================================
# MAPA GEOESTRATEGICO
# =========================================================

if all(existe(df_filtrado,c) for c in [

    'LATITUD',
    'LONGITUD'
]):

    st.subheader(
        "Mapa Geoestratégico"
    )

    fig_map = px.scatter_mapbox(

        df_filtrado,

        lat='LATITUD',

        lon='LONGITUD',

        color='PRIORIDAD_ESTRATEGICA',

        size='EVENTOS',

        hover_name='MUNICIPIO',

        zoom=4,

        height=700,

        color_continuous_scale='Turbo'
    )

    fig_map.update_layout(
        mapbox_style='carto-darkmatter'
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# =========================================================
# MAPA TEMPORAL
# =========================================================

if existe(df_filtrado,'SEMANA') and all(

    existe(df_filtrado,c)

    for c in ['LATITUD','LONGITUD']

):

    st.subheader(
        "Mapa Temporal Dinámico"
    )

    fig_time = px.scatter_mapbox(

        df_filtrado,

        lat='LATITUD',

        lon='LONGITUD',

        color='RIESGO_EVOLUTIVO',

        size='EVENTOS',

        animation_frame='SEMANA',

        hover_name='MUNICIPIO',

        zoom=4,

        height=700,

        color_continuous_scale='Turbo'
    )

    fig_time.update_layout(
        mapbox_style='carto-darkmatter'
    )

    st.plotly_chart(
        fig_time,
        use_container_width=True
    )

# =========================================================
# DENSIDAD CRIMINAL
# =========================================================

if all(existe(df_filtrado,c) for c in [

    'LATITUD',
    'LONGITUD'
]):

    st.subheader(
        "Mapa de Densidad Criminal"
    )

    fig_density = px.density_mapbox(

        df_filtrado,

        lat='LATITUD',

        lon='LONGITUD',

        z='RIESGO_EVOLUTIVO',

        radius=45,

        zoom=4,

        height=700,

        mapbox_style='carto-darkmatter'
    )

    st.plotly_chart(
        fig_density,
        use_container_width=True
    )

# =========================================================
# MATRIZ ESTRATEGICA
# =========================================================

st.subheader(
    "Matriz Estratégica"
)

fig_matrix = px.scatter(

    df_filtrado,

    x='RIESGO_EVOLUTIVO',

    y='PRIORIDAD_ESTRATEGICA',

    size='EVENTOS',

    color='PRIORIDAD_ESTRATEGICA',

    hover_name='MUNICIPIO',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_matrix,
    use_container_width=True
)

# =========================================================
# CAPACIDAD TRANSFORMACION GAO
# =========================================================

if COL_GAO:

    st.subheader(
        "Capacidad de Innovación y Transformación GAO"
    )

    gao_stats = df_op.groupby(
        COL_GAO,
        as_index=False
    ).size()

    gao_stats.columns = [
        'GAO',
        'ACTIVIDAD_OPERACIONAL'
    ]

    gao_stats['CAPACIDAD_TRANSFORMACION'] = (

        gao_stats['ACTIVIDAD_OPERACIONAL']
        /
        gao_stats['ACTIVIDAD_OPERACIONAL'].max()

    ) * 100

    fig_gao = px.bar(

        gao_stats.sort_values(
            by='CAPACIDAD_TRANSFORMACION',
            ascending=False
        ).head(15),

        x='GAO',

        y='CAPACIDAD_TRANSFORMACION',

        color='CAPACIDAD_TRANSFORMACION',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_gao,
        use_container_width=True
    )

# =========================================================
# ESCENARIOS PROSPECTIVOS
# =========================================================

st.subheader(
    "Escenarios Prospectivos"
)

prospectiva = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
).agg({

    'EVENTOS':'sum',

    'RIESGO':'mean',

    'GAO_PRESENTES':'mean',

    'UNIDADES_EXPUESTAS':'mean'
})

prospectiva['ESCENARIO_CRITICO'] = (

    prospectiva['EVENTOS'] * 0.35 +

    prospectiva['RIESGO'] * 0.35 +

    prospectiva['GAO_PRESENTES'] * 0.20 +

    prospectiva['UNIDADES_EXPUESTAS'] * 0.10
) * 1.40

fig_prosp = px.bar(

    prospectiva.sort_values(
        by='ESCENARIO_CRITICO',
        ascending=False
    ).head(20),

    x='MUNICIPIO',

    y='ESCENARIO_CRITICO',

    color='ESCENARIO_CRITICO',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_prosp,
    use_container_width=True
)

# =========================================================
# IA PREDICTIVA
# =========================================================

features = [

    'FREQ_HISTORICA',
    'DIV_TACTICA',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS',
    'IET',
    'RIESGO'
]

target = 'EVENTO_FUTURO'

if all(existe(df_filtrado,c) for c in features+[target]):

    st.subheader(
        "Motor IA Predictiva"
    )

    modelo_df = df_filtrado[
        features+[target]
    ].dropna()

    if len(modelo_df) > 50:

        X = modelo_df[features]
        y = modelo_df[target]

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )

        modelo = XGBClassifier(

            n_estimators=100,

            max_depth=4,

            learning_rate=0.1,

            eval_metric='logloss'
        )

        modelo.fit(
            X_train,
            y_train
        )

        pred = modelo.predict(X_test)

        acc = accuracy_score(
            y_test,
            pred
        )

        st.metric(
            "Precisión IA",
            round(acc,3)
        )

# =========================================================
# CLUSTERING TERRITORIAL
# =========================================================

cluster_cols = [

    'RIESGO',
    'EVENTOS',
    'IET',
    'GAO_PRESENTES'
]

if all(existe(df_filtrado,c) for c in cluster_cols):

    st.subheader(
        "Clustering Territorial"
    )

    cluster_df = df_filtrado[
        ['MUNICIPIO'] + cluster_cols
    ].dropna()

    modelo_cluster = KMeans(

        n_clusters=4,

        random_state=42,

        n_init=10
    )

    cluster_df['CLUSTER'] = modelo_cluster.fit_predict(

        cluster_df[cluster_cols]
    )

    fig_cluster = px.scatter(

        cluster_df,

        x='RIESGO',

        y='EVENTOS',

        size='IET',

        color='CLUSTER',

        hover_name='MUNICIPIO'
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

# =========================================================
# REDES RELACIONALES
# =========================================================

if COL_GAO and existe(df_op,'MUNICIPIO'):

    st.subheader(
        "Grafos Relacionales"
    )

    G = nx.Graph()

    red = df_op[
        [COL_GAO,'MUNICIPIO']
    ].dropna()

    for _, row in red.iterrows():

        G.add_edge(
            row[COL_GAO],
            row['MUNICIPIO']
        )

    pos = nx.spring_layout(
        G,
        seed=42
    )

    edge_x = []
    edge_y = []

    for edge in G.edges():

        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(

        x=edge_x,
        y=edge_y,

        line=dict(width=0.5),

        hoverinfo='none',

        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    node_trace = go.Scatter(

        x=node_x,
        y=node_y,

        mode='markers+text',

        text=node_text,

        textposition="top center",

        hoverinfo='text',

        marker=dict(
            size=10,
            color='cyan'
        )
    )

    fig_grafo = go.Figure(

        data=[edge_trace, node_trace]
    )

    fig_grafo.update_layout(

        showlegend=False,

        height=700,

        template='plotly_dark'
    )

    st.plotly_chart(
        fig_grafo,
        use_container_width=True
    )

# =========================================================
# ALERTAS
# =========================================================

st.subheader(
    "Alertas Inteligentes"
)

alertas = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
).agg({

    'PRIORIDAD_ESTRATEGICA':'mean',

    'EVENTOS':'sum',

    'GAO_PRESENTES':'mean'
})

alertas['TIPO_ALERTA'] = np.where(

    (
        (alertas['PRIORIDAD_ESTRATEGICA'] >= 30) &
        (alertas['GAO_PRESENTES'] >= 2)
    ),

    "CONVERGENCIA CRITICA",

    np.where(

        alertas['EVENTOS'] >= alertas['EVENTOS'].quantile(0.90),

        "ESCALAMIENTO",

        "VIGILANCIA"
    )
)

st.dataframe(

    alertas.sort_values(
        by='PRIORIDAD_ESTRATEGICA',
        ascending=False
    ),

    use_container_width=True
)

# =========================================================
# MOTOR NARRATIVO IA
# =========================================================

st.subheader(
    "Narrativa Estratégica Automatizada"
)

top = alertas.sort_values(

    by='PRIORIDAD_ESTRATEGICA',

    ascending=False

).head(5)

for _, row in top.iterrows():

    narrativa = f"""

    El municipio de {row['MUNICIPIO']}
    presenta una condición operacional
    de tipo {row['TIPO_ALERTA']}.

    Se identifica incremento de presión
    territorial y probabilidad de
    expansión operacional.

    La evolución prospectiva indica
    necesidad de monitoreo prioritario.

    """

    st.markdown(
        narrativa
    )

# =========================================================
# EXPORTACION EJECUTIVA
# =========================================================

st.subheader(
    "Exportación Ejecutiva"
)

excel_buffer = BytesIO()

with pd.ExcelWriter(
    excel_buffer
) as writer:

    df_filtrado.to_excel(

        writer,

        index=False,

        sheet_name='Fusion_Territorial'
    )

st.download_button(

    label="Descargar Reporte Estratégico Excel",

    data=excel_buffer.getvalue(),

    file_name="Reporte_Fusion_Territorial.xlsx",

    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
