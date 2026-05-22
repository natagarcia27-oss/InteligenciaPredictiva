# =========================================================
# CENTRO DE FUSIÓN GEOESPACIAL E INTELIGENCIA TERRITORIAL
# VERSION FASE AVANZADA
# FASE:
# GRAFOS + EXPANSION TEMPORAL + CORREDORES REALES
# + PRESION PERIFERICA + ALERTAS DINAMICAS
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

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

import networkx as nx

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(

    page_title="Fusion Territorial",

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
    background-color: #161b22;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# FUNCIONES
# =========================================================

def leer_csv_seguro(ruta):

    cods = [

        "utf-8",
        "latin1",
        "cp1252",
        "ISO-8859-1"
    ]

    for cod in cods:

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
# CARGA
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
# DETECTAR COLUMNAS
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

COL_ACTIVIDAD = detectar_columna(

    [
        "ACTIVIDAD",
        "TIPO_DE_ACTIVIDAD"
    ],

    df_op
)

COL_AFECTACION = detectar_columna(

    [
        "TIPO_DE_AFECTACION",
        "TIPO_DE_AFECTACIÓN"
    ],

    df_op
)

# =========================================================
# SIDEBAR
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
# NUMERICOS
# =========================================================

numericas = [

    'RIESGO',
    'IET',
    'EVENTOS',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS',
    'FREQ_HISTORICA',
    'DIV_TACTICA'
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

# =========================================================
# ALERTAS
# =========================================================

def alerta(v):

    if v >= 30:
        return "CRITICO"

    elif v >= 20:
        return "ALTO"

    elif v >= 10:
        return "MEDIO"

    else:
        return "BAJO"

df_filtrado['ALERTA'] = df_filtrado[
    'IGE'
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

c1,c2,c3,c4,c5,c6 = st.columns(6)

c1.metric(
    "Municipios",
    int(df_filtrado['MUNICIPIO'].nunique())
)

c2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

c3.metric(
    "IPE",
    round(df_filtrado['IPE'].mean(),2)
)

c4.metric(
    "IPO",
    round(df_filtrado['IPO'].mean(),2)
)

c5.metric(
    "IGE",
    round(df_filtrado['IGE'].mean(),2)
)

c6.metric(
    "GAO",
    round(df_filtrado['GAO_PRESENTES'].mean(),2)
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

    mapa = df_filtrado.groupby(

        [
            'MUNICIPIO',
            'LATITUD',
            'LONGITUD'
        ],

        as_index=False

    ).agg({

        'IGE':'mean',

        'EVENTOS':'sum'
    })

    fig_map = px.scatter_mapbox(

        mapa,

        lat='LATITUD',

        lon='LONGITUD',

        size='EVENTOS',

        color='IGE',

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
# MAPA CALOR
# =========================================================

if all(existe(df_filtrado,c) for c in [

    'LATITUD',
    'LONGITUD'
]):

    st.subheader(
        "Mapa Calor Territorial"
    )

    fig_heat = px.density_mapbox(

        df_filtrado,

        lat='LATITUD',

        lon='LONGITUD',

        z='EVENTOS',

        radius=30,

        zoom=4,

        height=700,

        mapbox_style='carto-darkmatter'
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True
    )

# =========================================================
# EXPANSION TEMPORAL
# =========================================================

if existe(df_filtrado,'SEMANA'):

    st.subheader(
        "Expansión Temporal"
    )

    exp_temp = df_filtrado.groupby(
        'SEMANA',
        as_index=False
    ).agg({

        'EVENTOS':'sum',

        'RIESGO':'mean'
    })

    fig_exp = px.line(

        exp_temp,

        x='SEMANA',

        y='EVENTOS',

        markers=True
    )

    st.plotly_chart(
        fig_exp,
        use_container_width=True
    )

# =========================================================
# PROPAGACION REGIONAL
# =========================================================

st.subheader(
    "Presión y Propagación Regional"
)

prop = df_filtrado.groupby(
    'DEPARTAMENTO',
    as_index=False
).agg({

    'EVENTOS':'sum',

    'RIESGO':'mean',

    'GAO_PRESENTES':'mean'
})

prop['PRESION'] = (

    prop['EVENTOS'] * 0.5 +

    prop['RIESGO'] * 0.3 +

    prop['GAO_PRESENTES'] * 0.2
)

fig_prop = px.bar(

    prop.sort_values(
        by='PRESION',
        ascending=False
    ),

    x='DEPARTAMENTO',

    y='PRESION',

    color='PRESION',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_prop,
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
# CLUSTERING
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

        color='CLUSTER',

        size='IET',

        hover_name='MUNICIPIO'
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

# =========================================================
# CENTRALIDAD OPERACIONAL
# =========================================================

if COL_GAO and existe(df_op,'MUNICIPIO'):

    st.subheader(
        "Centralidad Operacional"
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

    centralidad = nx.degree_centrality(G)

    cent_df = pd.DataFrame({

        'NODO': list(centralidad.keys()),

        'CENTRALIDAD': list(centralidad.values())
    })

    fig_cent = px.bar(

        cent_df.sort_values(
            by='CENTRALIDAD',
            ascending=False
        ).head(20),

        x='NODO',

        y='CENTRALIDAD',

        color='CENTRALIDAD',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_cent,
        use_container_width=True
    )

# =========================================================
# GRAFO OPERACIONAL
# =========================================================

if COL_GAO and existe(df_op,'MUNICIPIO'):

    st.subheader(
        "Grafo Operacional"
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

        edge_x.extend(
            [x0, x1, None]
        )

        edge_y.extend(
            [y0, y1, None]
        )

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
# CONVERGENCIA CRIMINAL
# =========================================================

if COL_GAO and existe(df_op,'MUNICIPIO'):

    st.subheader(
        "Convergencia Criminal"
    )

    conv = df_op.groupby(
        'MUNICIPIO'
    )[COL_GAO].nunique().reset_index()

    conv.columns = [

        'MUNICIPIO',
        'GAO_CONVERGENTES'
    ]

    fig_conv = px.bar(

        conv.sort_values(
            by='GAO_CONVERGENTES',
            ascending=False
        ).head(20),

        x='MUNICIPIO',

        y='GAO_CONVERGENTES',

        color='GAO_CONVERGENTES',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_conv,
        use_container_width=True
    )

# =========================================================
# ALERTAS AVANZADAS
# =========================================================

st.subheader(
    "Alertas Avanzadas"
)

alertas = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
).agg({

    'IGE':'mean',

    'EVENTOS':'sum',

    'GAO_PRESENTES':'mean'
})

alertas['TIPO_ALERTA'] = np.where(

    (
        (alertas['IGE'] >= 20) &
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
        by='IGE',
        ascending=False
    ),

    use_container_width=True
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
