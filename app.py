# =========================================================
# CENTRO DE FUSIÓN GEOESPACIAL E INTELIGENCIA TERRITORIAL
# VERSION OPERACIONAL AVANZADA
# FASE:
# INTELIGENCIA RELACIONAL + CORREDORES +
# PRESION PERIFERICA + EXPANSION DINAMICA +
# ALERTAS PREDICTIVAS + DETECCION DE NODOS
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

    page_title="Fusion Territorial Avanzada",

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

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

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

c7.metric(
    "Alertas Criticas",
    int(
        (df_filtrado['ALERTA'] == 'CRITICO').sum()
    )
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

        radius=35,

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

    exp = df_filtrado.groupby(
        'SEMANA',
        as_index=False
    ).agg({

        'EVENTOS':'sum',

        'RIESGO':'mean'
    })

    fig_exp = px.line(

        exp,

        x='SEMANA',

        y='EVENTOS',

        markers=True
    )

    st.plotly_chart(
        fig_exp,
        use_container_width=True
    )

# =========================================================
# PRESION PERIFERICA
# =========================================================

st.subheader(
    "Presión Periférica"
)

presion = df_filtrado.groupby(
    'DEPARTAMENTO',
    as_index=False
).agg({

    'EVENTOS':'sum',

    'RIESGO':'mean',

    'GAO_PRESENTES':'mean'
})

presion['PRESION_OPERACIONAL'] = (

    presion['EVENTOS'] * 0.5 +

    presion['RIESGO'] * 0.3 +

    presion['GAO_PRESENTES'] * 0.2
)

fig_presion = px.bar(

    presion.sort_values(
        by='PRESION_OPERACIONAL',
        ascending=False
    ),

    x='DEPARTAMENTO',

    y='PRESION_OPERACIONAL',

    color='PRESION_OPERACIONAL',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_presion,
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
# ANOMALIAS
# =========================================================

anom_cols = [

    'EVENTOS',
    'RIESGO',
    'IET',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS'
]

if all(existe(df_filtrado,c) for c in anom_cols):

    st.subheader(
        "Anomalías Estratégicas"
    )

    anom_df = df_filtrado[
        ['MUNICIPIO'] + anom_cols
    ].dropna()

    detector = IsolationForest(

        contamination=0.05,

        random_state=42
    )

    detector.fit(
        anom_df[anom_cols]
    )

    anom_df['ANOMALIA'] = detector.predict(

        anom_df[anom_cols]
    )

    anomalos = anom_df[
        anom_df['ANOMALIA'] == -1
    ]

    fig_anom = px.scatter(

        anomalos,

        x='RIESGO',

        y='EVENTOS',

        size='IET',

        color='UNIDADES_EXPUESTAS',

        hover_name='MUNICIPIO',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_anom,
        use_container_width=True
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
# INNOVACION Y TRANSFORMACION GAO
# =========================================================

if COL_GAO and COL_ACTIVIDAD:

    st.subheader(
        "Capacidad Innovación y Transformación GAO"
    )

    org = df_op.groupby(
        COL_GAO,
        as_index=False
    ).agg({

        'MUNICIPIO':'nunique',

        COL_ACTIVIDAD:'nunique'
    })

    org.columns = [

        'ORGANIZACION',

        'EXPANSION',

        'TACTICAS'
    ]

    org['ADAPTACION'] = (

        org['EXPANSION'] * 0.5 +

        org['TACTICAS'] * 0.5
    )

    fig_org = px.scatter(

        org,

        x='EXPANSION',

        y='TACTICAS',

        size='ADAPTACION',

        color='ADAPTACION',

        hover_name='ORGANIZACION',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_org,
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
# CORREDORES REALES
# =========================================================

if COL_GAO and existe(df_op,'DEPARTAMENTO'):

    st.subheader(
        "Corredores Operacionales Reales"
    )

    corredores = df_op.groupby(
        'DEPARTAMENTO',
        as_index=False
    ).agg({

        COL_GAO:'nunique',

        'MUNICIPIO':'nunique'
    })

    corredores.columns = [

        'DEPARTAMENTO',

        'GAO',

        'MUNICIPIOS'
    ]

    corredores['CORREDOR_SCORE'] = (

        corredores['GAO'] * 0.6 +

        corredores['MUNICIPIOS'] * 0.4
    )

    fig_corr = px.bar(

        corredores.sort_values(
            by='CORREDOR_SCORE',
            ascending=False
        ),

        x='DEPARTAMENTO',

        y='CORREDOR_SCORE',

        color='CORREDOR_SCORE',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

# =========================================================
# DETECCION DE NODOS CRITICOS
# =========================================================

st.subheader(
    "Nodos Críticos"
)

nodos = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
).agg({

    'EVENTOS':'sum',

    'RIESGO':'mean',

    'GAO_PRESENTES':'mean'
})

nodos['NODO_SCORE'] = (

    nodos['EVENTOS'] * 0.5 +

    nodos['RIESGO'] * 0.3 +

    nodos['GAO_PRESENTES'] * 0.2
)

fig_nodos = px.bar(

    nodos.sort_values(
        by='NODO_SCORE',
        ascending=False
    ).head(20),

    x='MUNICIPIO',

    y='NODO_SCORE',

    color='NODO_SCORE',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_nodos,
    use_container_width=True
)

# =========================================================
# ALERTAS PREDICTIVAS
# =========================================================

st.subheader(
    "Alertas Predictivas"
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
