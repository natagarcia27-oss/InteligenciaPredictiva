# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

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
# ESTILO
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

    df.columns = [

        col.strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")

        for col in df.columns
    ]

    return df

# =========================================================
# CARGAR DATOS
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
# DETECTAR COLUMNAS
# =========================================================

def detectar_columna(lista, dataframe):

    for col in lista:

        if col in dataframe.columns:

            return col

    return None

COL_GAO = detectar_columna(

    [
        'GAO',
        'FG/BLOQUE/ESTRUCTURA',
        'FG_BLOQUE_ESTRUCTURA'
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
        'TIPO_DE_AFECTACION',
        'TIPO_DE_AFECTACIÓN'
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
# CREAR IPE ANTES DE TODO
# =========================================================

columnas_ipe = [

    'RIESGO',

    'IET',

    'EVENTOS',

    'UNIDADES_EXPUESTAS',

    'GAO_PRESENTES'
]

validas_ipe = all(
    col in df_filtrado.columns
    for col in columnas_ipe
)

if validas_ipe:

    df_filtrado['IPE'] = (

        df_filtrado['RIESGO'] * 0.30

        +

        df_filtrado['IET'] * 0.25

        +

        df_filtrado['EVENTOS'] * 0.20

        +

        df_filtrado['UNIDADES_EXPUESTAS'] * 0.15

        +

        df_filtrado['GAO_PRESENTES'] * 0.10
    )

else:

    df_filtrado['IPE'] = 0

# =========================================================
# TITULO
# =========================================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

st.markdown("""

### Plataforma integrada de:

- Inteligencia Operacional
- Prospectiva Territorial
- IA Predictiva
- Corredores Estratégicos
- Riesgo Territorial
- Detección de Anomalías
- Clustering Territorial

""")

# =========================================================
# KPI
# =========================================================

st.subheader(
    "Indicadores Estratégicos"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Municipios",
    df_filtrado['MUNICIPIO'].nunique()
)

c2.metric(
    "Eventos",
    int(df_filtrado['EVENTOS'].sum())
)

c3.metric(
    "Riesgo",
    round(df_filtrado['RIESGO'].mean(), 2)
)

c4.metric(
    "IET",
    round(df_filtrado['IET'].mean(), 2)
)

c5.metric(
    "IPE",
    round(df_filtrado['IPE'].mean(), 2)
)

# =========================================================
# MUNICIPIOS CRÍTICOS
# =========================================================

st.subheader(
    "Municipios Críticos"
)

top = df_filtrado.groupby(
    'MUNICIPIO'
)['IPE'].mean().reset_index()

top = top.sort_values(
    by='IPE',
    ascending=False
)

fig_top = px.bar(

    top.head(20),

    x='MUNICIPIO',

    y='IPE',

    color='IPE',

    color_continuous_scale='Turbo'
)

st.plotly_chart(
    fig_top,
    use_container_width=True
)

# =========================================================
# EVOLUCIÓN TEMPORAL
# =========================================================

if 'SEMANA' in df_filtrado.columns:

    st.subheader(
        "Evolución Temporal"
    )

    temporal = df_filtrado.groupby(
        'SEMANA'
    )['EVENTOS'].sum().reset_index()

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
# MAPA
# =========================================================

lat_ok = 'LATITUD' in df_filtrado.columns
lon_ok = 'LONGITUD' in df_filtrado.columns

if lat_ok and lon_ok:

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
    "Motor IA Predictiva"
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

validas = all(
    col in df_filtrado.columns
    for col in features + [target]
)

if validas:

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
            "Precisión IA",
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
            'MUNICIPIO'
        )['RIESGO_PREDICTIVO'].mean().reset_index()

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
# ANOMALÍAS
# =========================================================

st.subheader(
    "Anomalías Estratégicas"
)

anomalias_cols = [

    'EVENTOS',

    'RIESGO',

    'IET',

    'GAO_PRESENTES',

    'UNIDADES_EXPUESTAS'
]

anom_ok = all(
    col in df_filtrado.columns
    for col in anomalias_cols
)

if anom_ok:

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

st.subheader(
    "Clustering Territorial"
)

cluster_cols = [

    'RIESGO',

    'EVENTOS',

    'IET',

    'GAO_PRESENTES'
]

cluster_ok = all(
    col in df_filtrado.columns
    for col in cluster_cols
)

if cluster_ok:

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
        COL_GAO
    ).agg({

        'MUNICIPIO':'nunique',

        COL_ACTIVIDAD:'nunique'

    }).reset_index()

    org.columns = [

        'ORGANIZACION',

        'EXPANSION',

        'TACTICAS'
    ]

    org['ADAPTACION'] = (

        org['EXPANSION'] * 0.5

        +

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
# MATRIZ RIESGO
# =========================================================

st.subheader(
    "Matriz Estratégica"
)

if all(
    col in df_filtrado.columns
    for col in ['RIESGO', 'EVENTOS', 'IET']
):

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
