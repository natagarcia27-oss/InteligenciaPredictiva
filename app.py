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

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(

    page_title="Sistema Inteligencia Predictiva Territorial",

    layout="wide"
)

# =========================================================
# ESTILO VISUAL
# =========================================================

st.markdown("""

<style>

.main {
    background-color: #0d1117;
    color: white;
}

h1, h2, h3 {
    color: #58a6ff;
}

[data-testid="stMetricValue"] {
    color: #58a6ff;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# FUNCIONES
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
# LIMPIAR COLUMNAS
# =========================================================

def limpiar_columnas(df):

    df.columns = df.columns.str.strip()

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
            .replace("-", "_")
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
# DETECCIÓN COLUMNAS
# =========================================================

def detectar_columna(candidatas, dataframe):

    for col in candidatas:

        if col in dataframe.columns:

            return col

    return None

COL_GAO = detectar_columna(

    [
        'GAO',
        'FG/BLOQUE/ESTRUCTURA',
        'FG_BLOQUE_ESTRUCTURA',
        'ESTRUCTURA',
        'SUBESTRUCTURA'
    ],

    df_op
)

COL_ACTIVIDAD = detectar_columna(

    [
        'ACTIVIDAD',
        'TIPO_DE_ACTIVIDAD'
    ],

    df_op
)

COL_AFECTACION = detectar_columna(

    [
        'TIPO_DE_AFECTACIÓN',
        'TIPO_DE_AFECTACION'
    ],

    df_op
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Filtros Estratégicos"
)

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
# TITULO
# =========================================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

st.markdown("""

### Plataforma integrada para:

- Inteligencia operacional
- Riesgo territorial
- Prospectiva estratégica
- IA predictiva
- Detección anomalías
- Inteligencia organizacional
- Corredores estratégicos
- Priorización operacional

""")

# =========================================================
# KPI
# =========================================================

st.subheader(
    "Indicadores Estratégicos"
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Municipios",
    df_filtrado['MUNICIPIO'].nunique()
)

col2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

col3.metric(
    "Riesgo",
    round(df_filtrado['RIESGO'].mean(), 2)
)

col4.metric(
    "GAO",
    round(df_filtrado['GAO_PRESENTES'].mean(), 2)
)

col5.metric(
    "IET",
    round(df_filtrado['IET'].mean(), 2)
)

# =========================================================
# IPE
# =========================================================

st.subheader(
    "Índice Predictivo Estratégico"
)

ipe = df_filtrado.copy()

ipe['IPE'] = (

    ipe['RIESGO'] * 0.30

    +

    ipe['IET'] * 0.25

    +

    ipe['EVENTOS'] * 0.20

    +

    ipe['UNIDADES_EXPUESTAS'] * 0.15

    +

    ipe['GAO_PRESENTES'] * 0.10
)

ipe_rank = ipe.groupby(
    'MUNICIPIO'
)['IPE'].mean().reset_index()

ipe_rank = ipe_rank.sort_values(
    by='IPE',
    ascending=False
)

fig_ipe = px.bar(

    ipe_rank.head(20),

    x='MUNICIPIO',

    y='IPE',

    color='IPE',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_ipe,
    use_container_width=True
)

# =========================================================
# MAPA
# =========================================================

st.subheader(
    "Mapa Estratégico"
)

mapa = df_filtrado.groupby(
    [
        'MUNICIPIO',
        'LATITUD',
        'LONGITUD'
    ]
).agg({

    'IPE':'mean',

    'EVENTOS':'sum'

}).reset_index()

fig_map = px.scatter_mapbox(

    mapa,

    lat='LATITUD',

    lon='LONGITUD',

    size='EVENTOS',

    color='IPE',

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
# IA PREDICTIVA
# =========================================================

st.subheader(
    "Motor Predictivo IA"
)

features = [

    'FREQ_HISTORICA',

    'DIV_TACTICA',

    'GAO_PRESENTES',

    'UNIDADES_EXPUESTAS',

    'IET',

    'RIESGO'
]

target = 'EVENTO_FUTURO'

validas = True

for col in features + [target]:

    if col not in df_filtrado.columns:

        validas = False

if validas:

    modelo_df = df_filtrado[
        features + [target]
    ].dropna()

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

        subsample=0.8,

        colsample_bytree=0.8,

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
        round(acc, 3)
    )

# =========================================================
# ANOMALÍAS
# =========================================================

st.subheader(
    "Anomalías Estratégicas"
)

anomalias_features = [

    'EVENTOS',

    'RIESGO',

    'IET',

    'GAO_PRESENTES',

    'UNIDADES_EXPUESTAS'
]

anomalias_df = df_filtrado[
    ['MUNICIPIO'] + anomalias_features
].dropna()

X_anom = anomalias_df[
    anomalias_features
]

detector = IsolationForest(

    contamination=0.05,

    random_state=42
)

detector.fit(X_anom)

anomalias_df['ANOMALIA'] = detector.predict(
    X_anom
)

anomalias = anomalias_df[
    anomalias_df['ANOMALIA'] == -1
]

fig_anom = px.scatter(

    anomalias,

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

st.subheader(
    "Clustering Territorial"
)

cluster_features = [

    'RIESGO',

    'EVENTOS',

    'IET',

    'GAO_PRESENTES'
]

cluster_df = df_filtrado[
    ['MUNICIPIO'] + cluster_features
].dropna()

X_cluster = cluster_df[
    cluster_features
]

modelo_cluster = KMeans(

    n_clusters=4,

    random_state=42
)

cluster_df['CLUSTER'] = modelo_cluster.fit_predict(
    X_cluster
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
# INTELIGENCIA ORGANIZACIONAL
# =========================================================

if COL_GAO and COL_ACTIVIDAD:

    st.subheader(
        "Inteligencia Organizacional"
    )

    gao_intel = df_op.groupby(
        COL_GAO
    ).agg({

        'MUNICIPIO':'nunique',

        COL_ACTIVIDAD:'nunique'

    }).reset_index()

    gao_intel.columns = [

        'ORGANIZACION',

        'EXPANSION',

        'TACTICAS'
    ]

    gao_intel['ADAPTACION'] = (

        gao_intel['EXPANSION'] * 0.5

        +

        gao_intel['TACTICAS'] * 0.5
    )

    fig_org = px.scatter(

        gao_intel,

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
# MATRIZ RIESGO
# =========================================================

st.subheader(
    "Matriz Estratégica"
)

fig_heat = px.density_heatmap(

    df_filtrado,

    x='RIESGO',

    y='EVENTOS',

    z='IET',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_heat,
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
