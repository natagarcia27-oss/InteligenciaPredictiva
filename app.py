# =========================================================
# CENTRO DE FUSIÓN E INTELIGENCIA TERRITORIAL
# VERSION FULL GEOINT + IA + ALERTAS + REDES
# VERSION ESTABLE STREAMLIT CLOUD
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

# =========================================================
# CONFIGURACION
# =========================================================

st.set_page_config(

    page_title="Centro Fusion Inteligencia Territorial",

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

section[data-testid="stSidebar"] {
    background-color: #161b22;
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

# =========================================================
# LIMPIAR COLUMNAS
# =========================================================

def limpiar_columnas(df):

    nuevas = []

    for col in df.columns:

        col = (
            str(col)
            .strip()
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

        nuevas.append(col)

    df.columns = nuevas

    return df

# =========================================================
# CARGA DE DATOS
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
# VALIDACION
# =========================================================

if df.empty:

    st.error(
        "No se pudo cargar riesgo_municipal.csv"
    )

    st.stop()

# =========================================================
# FUNCION EXISTE
# =========================================================

def existe(dataframe, columna):

    return columna in dataframe.columns

# =========================================================
# DETECTAR COLUMNAS OPERACIONALES
# =========================================================

def detectar_columna(lista, dataframe):

    for col in lista:

        if col in dataframe.columns:

            return col

    return None

COL_GAO = detectar_columna(

    [
        "GAO",
        "FG/BLOQUE/ESTRUCTURA",
        "FG_BLOQUE_ESTRUCTURA"
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

        sorted(
            df['DEPARTAMENTO']
            .dropna()
            .unique()
        )
    )

if existe(df, 'MUNICIPIO'):

    mun_sel = st.sidebar.multiselect(

        "Municipio",

        sorted(
            df['MUNICIPIO']
            .dropna()
            .unique()
        )
    )

if existe(df, 'AÑO'):

    anio_sel = st.sidebar.multiselect(

        "Año",

        sorted(
            df['AÑO']
            .dropna()
            .unique()
        )
    )

# =========================================================
# FILTRADO
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
# VALIDACION COLUMNAS
# =========================================================

columnas_numericas = [

    'RIESGO',
    'IET',
    'EVENTOS',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS',
    'FREQ_HISTORICA',
    'DIV_TACTICA'
]

for col in columnas_numericas:

    if existe(df_filtrado, col):

        df_filtrado[col] = pd.to_numeric(
            df_filtrado[col],
            errors='coerce'
        ).fillna(0)

# =========================================================
# IPE
# =========================================================

if all(existe(df_filtrado, c) for c in [

    'RIESGO',
    'IET',
    'EVENTOS',
    'UNIDADES_EXPUESTAS',
    'GAO_PRESENTES'
]):

    df_filtrado['IPE'] = (

        df_filtrado['RIESGO'] * 0.30 +

        df_filtrado['IET'] * 0.25 +

        df_filtrado['EVENTOS'] * 0.20 +

        df_filtrado['UNIDADES_EXPUESTAS'] * 0.15 +

        df_filtrado['GAO_PRESENTES'] * 0.10
    )

else:

    df_filtrado['IPE'] = 0

# =========================================================
# IPO
# =========================================================

if all(existe(df_filtrado, c) for c in [

    'EVENTOS',
    'RIESGO',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS'
]):

    df_filtrado['IPO'] = (

        df_filtrado['EVENTOS'] * 0.35 +

        df_filtrado['GAO_PRESENTES'] * 0.25 +

        df_filtrado['UNIDADES_EXPUESTAS'] * 0.20 +

        df_filtrado['RIESGO'] * 0.20
    )

else:

    df_filtrado['IPO'] = 0

# =========================================================
# IGE
# =========================================================

df_filtrado['IGE'] = (

    df_filtrado['IPE'] * 0.60 +

    df_filtrado['IPO'] * 0.40
)

# =========================================================
# ALERTAS
# =========================================================

def clasificar_alerta(valor):

    if valor >= 25:
        return "CRITICO"

    elif valor >= 15:
        return "ALTO"

    elif valor >= 8:
        return "MEDIO"

    else:
        return "BAJO"

df_filtrado['ALERTA'] = df_filtrado[
    'IGE'
].apply(clasificar_alerta)

# =========================================================
# TITULO
# =========================================================

st.title(
    "Centro de Fusión e Inteligencia Territorial"
)

st.markdown("""

### Plataforma Integrada de:

- Inteligencia Estratégica
- GEOINT
- IA Predictiva
- Prospectiva Territorial
- Clustering Territorial
- Alertas Tempranas
- Inteligencia Operacional
- Corredores Estratégicos
- Convergencia Criminal

""")

# =========================================================
# KPIs
# =========================================================

st.subheader(
    "Indicadores Estratégicos"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Municipios",
    int(df_filtrado['MUNICIPIO'].nunique())
)

c2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

c3.metric(
    "Riesgo Prom",
    round(df_filtrado['RIESGO'].mean(), 2)
)

c4.metric(
    "IPE",
    round(df_filtrado['IPE'].mean(), 2)
)

c5.metric(
    "IPO",
    round(df_filtrado['IPO'].mean(), 2)
)

c6.metric(
    "IGE",
    round(df_filtrado['IGE'].mean(), 2)
)

# =========================================================
# MAPA GEOESTRATEGICO
# =========================================================

if all(existe(df_filtrado, c) for c in [

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
# MUNICIPIOS PRIORITARIOS
# =========================================================

st.subheader(
    "Municipios Prioritarios"
)

prioridad = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
)['IGE'].mean()

prioridad = prioridad.sort_values(
    by='IGE',
    ascending=False
)

fig_prioridad = px.bar(

    prioridad.head(20),

    x='MUNICIPIO',

    y='IGE',

    color='IGE',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_prioridad,
    use_container_width=True
)

# =========================================================
# EVOLUCION TEMPORAL
# =========================================================

if all(existe(df_filtrado, c) for c in [

    'SEMANA',
    'EVENTOS'
]):

    st.subheader(
        "Evolución Temporal"
    )

    temporal = df_filtrado.groupby(
        'SEMANA',
        as_index=False
    )['EVENTOS'].sum()

    fig_temp = px.line(

        temporal,

        x='SEMANA',

        y='EVENTOS',

        markers=True
    )

    st.plotly_chart(
        fig_temp,
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

if all(existe(df_filtrado, c) for c in features + [target]):

    st.subheader(
        "Motor IA Predictiva"
    )

    modelo_df = df_filtrado[
        features + [target]
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
            "Precision IA",
            round(acc, 3)
        )

        probabilidades = modelo.predict_proba(X)[:,1]

        modelo_df['RIESGO_PREDICTIVO'] = (
            probabilidades * 100
        )

        predicciones = df_filtrado.loc[
            modelo_df.index
        ].copy()

        predicciones['RIESGO_PREDICTIVO'] = (
            modelo_df['RIESGO_PREDICTIVO']
        )

        top_pred = predicciones.groupby(
            'MUNICIPIO',
            as_index=False
        )['RIESGO_PREDICTIVO'].mean()

        fig_pred = px.bar(

            top_pred.sort_values(
                by='RIESGO_PREDICTIVO',
                ascending=False
            ).head(20),

            x='MUNICIPIO',

            y='RIESGO_PREDICTIVO',

            color='RIESGO_PREDICTIVO',

            color_continuous_scale='Turbo'
        )

        st.plotly_chart(
            fig_pred,
            use_container_width=True
        )

# =========================================================
# ANOMALIAS
# =========================================================

anomalias_cols = [

    'EVENTOS',
    'RIESGO',
    'IET',
    'GAO_PRESENTES',
    'UNIDADES_EXPUESTAS'
]

if all(existe(df_filtrado, c) for c in anomalias_cols):

    st.subheader(
        "Anomalías Estratégicas"
    )

    anom_df = df_filtrado[
        ['MUNICIPIO'] + anomalias_cols
    ].dropna()

    if len(anom_df) > 20:

        X_anom = anom_df[
            anomalias_cols
        ]

        detector = IsolationForest(

            contamination=0.05,

            random_state=42
        )

        detector.fit(X_anom)

        anom_df['ANOMALIA'] = detector.predict(
            X_anom
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

if all(existe(df_filtrado, c) for c in cluster_cols):

    st.subheader(
        "Clustering Territorial"
    )

    cluster_df = df_filtrado[
        ['MUNICIPIO'] + cluster_cols
    ].dropna()

    if len(cluster_df) > 10:

        X_cluster = cluster_df[
            cluster_cols
        ]

        modelo_cluster = KMeans(

            n_clusters=4,

            random_state=42,

            n_init=10
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
# CONVERGENCIA CRIMINAL
# =========================================================

if COL_GAO and existe(df_op, 'MUNICIPIO'):

    st.subheader(
        "Convergencia Criminal"
    )

    convergencia = df_op.groupby(
        'MUNICIPIO'
    )[COL_GAO].nunique().reset_index()

    convergencia.columns = [

        'MUNICIPIO',
        'GAO_CONVERGENTES'
    ]

    convergencia = convergencia.sort_values(
        by='GAO_CONVERGENTES',
        ascending=False
    )

    fig_conv = px.bar(

        convergencia.head(20),

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
# MUNICIPIOS EMERGENTES
# =========================================================

st.subheader(
    "Municipios Emergentes"
)

emergentes = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
).agg({

    'RIESGO':'mean',

    'EVENTOS':'sum'
})

emergentes['INDICE_EMERGENTE'] = (

    emergentes['RIESGO'] * 0.6 +

    emergentes['EVENTOS'] * 0.4
)

emergentes = emergentes.sort_values(
    by='INDICE_EMERGENTE',
    ascending=False
)

fig_emerg = px.bar(

    emergentes.head(20),

    x='MUNICIPIO',

    y='INDICE_EMERGENTE',

    color='INDICE_EMERGENTE',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_emerg,
    use_container_width=True
)

# =========================================================
# CORREDORES ESTRATEGICOS
# =========================================================

if existe(df_filtrado, 'DEPARTAMENTO'):

    st.subheader(
        "Corredores Estratégicos"
    )

    corredores = df_filtrado.groupby(
        'DEPARTAMENTO',
        as_index=False
    )['EVENTOS'].sum()

    corredores = corredores.sort_values(
        by='EVENTOS',
        ascending=False
    )

    fig_corr = px.bar(

        corredores,

        x='DEPARTAMENTO',

        y='EVENTOS',

        color='EVENTOS',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

# =========================================================
# MATRIZ ESTRATEGICA
# =========================================================

if all(existe(df_filtrado, c) for c in [

    'RIESGO',
    'EVENTOS',
    'IET'
]):

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
# ALERTAS ESTRATEGICAS
# =========================================================

st.subheader(
    "Alertas Estratégicas"
)

alertas = df_filtrado.groupby(
    'MUNICIPIO',
    as_index=False
)['IGE'].mean()

alertas['ALERTA'] = alertas['IGE'].apply(
    clasificar_alerta
)

st.dataframe(

    alertas.sort_values(
        by='IGE',
        ascending=False
    ),

    use_container_width=True
)

# =========================================================
# TABLA CONSOLIDADA
# =========================================================

st.subheader(
    "Vista Estratégica Consolidada"
)

st.dataframe(
    df_filtrado,
    use_container_width=True
)
