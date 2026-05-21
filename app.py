import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Sistema Inteligencia Predictiva Territorial",
    layout="wide"
)

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
# LIMPIEZA COLUMNAS
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
# DETECCIÓN COLUMNAS OPERACIONALES
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

with st.sidebar.expander(
    "Columnas Base Predictiva"
):

    st.write(df.columns.tolist())

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
# FILTRAR BASE
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
### Plataforma Integrada de:

- Inteligencia Operacional
- Prospectiva Territorial
- Riesgo Estratégico
- Adaptación Organizacional
- Corredores Críticos
- Alertas Tempranas
- IA Predictiva
- Detección de Anomalías
""")

# =========================================================
# KPIs
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
    "Riesgo Promedio",
    round(df_filtrado['RIESGO'].mean(), 2)
)

col4.metric(
    "GAO Presentes",
    int(df_filtrado['GAO_PRESENTES'].mean())
)

col5.metric(
    "IET Promedio",
    round(df_filtrado['IET'].mean(), 2)
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

    title='Top Municipios Críticos'
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =========================================================
# EVOLUCIÓN TEMPORAL
# =========================================================

st.subheader(
    "Evolución Temporal"
)

temporal = df_filtrado.groupby(
    ['AÑO', 'SEMANA']
).agg({

    'EVENTOS':'sum'

}).reset_index()

temporal['PERIODO'] = (

    temporal['AÑO'].astype(str)
    + '-S'
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

df_alertas['IPT'] = (

    df_alertas['RIESGO'] * 0.5

    +

    df_alertas['IET'] * 0.3

    +

    df_alertas['EVENTOS'] * 0.2
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

st.subheader(
    "Inteligencia Organizacional"
)

if COL_GAO and COL_ACTIVIDAD:

    gao_intel = df_op.groupby(
        COL_GAO
    ).agg({

        'MUNICIPIO':'nunique',

        COL_ACTIVIDAD:'nunique'

    }).reset_index()

    gao_intel.columns = [

        'ORGANIZACION',

        'EXPANSION_TERRITORIAL',

        'DIVERSIDAD_TACTICA'
    ]

    gao_intel['INDICE_ADAPTACION'] = (

        gao_intel['EXPANSION_TERRITORIAL'] * 0.5

        +

        gao_intel['DIVERSIDAD_TACTICA'] * 0.5
    )

    gao_intel = gao_intel.sort_values(

        by='INDICE_ADAPTACION',

        ascending=False
    )

    st.dataframe(
        gao_intel,
        use_container_width=True
    )

    fig_org = px.scatter(

        gao_intel,

        x='EXPANSION_TERRITORIAL',

        y='DIVERSIDAD_TACTICA',

        size='INDICE_ADAPTACION',

        color='INDICE_ADAPTACION',

        hover_name='ORGANIZACION',

        title='Adaptación y Expansión Organizacional',

        color_continuous_scale='Reds'
    )

    st.plotly_chart(
        fig_org,
        use_container_width=True
    )

# =========================================================
# MUTACIÓN OPERACIONAL
# =========================================================

st.subheader(
    "Detección de Mutación Operacional"
)

if COL_GAO and COL_ACTIVIDAD and COL_AFECTACION:

    mutacion = df_op.groupby(
        COL_GAO
    ).agg({

        COL_ACTIVIDAD:'nunique',

        COL_AFECTACION:'nunique',

        'MUNICIPIO':'nunique'

    }).reset_index()

    mutacion.columns = [

        'ORGANIZACION',

        'DIVERSIDAD_ACTIVIDAD',

        'DIVERSIDAD_AFECTACION',

        'EXPANSION_TERRITORIAL'
    ]

    mutacion['INDICE_MUTACION'] = (

        mutacion['DIVERSIDAD_ACTIVIDAD'] * 0.4

        +

        mutacion['DIVERSIDAD_AFECTACION'] * 0.3

        +

        mutacion['EXPANSION_TERRITORIAL'] * 0.3
    )

    mutacion = mutacion.sort_values(

        by='INDICE_MUTACION',

        ascending=False
    )

    st.dataframe(
        mutacion,
        use_container_width=True
    )

    fig_mut = px.bar(

        mutacion.head(20),

        x='ORGANIZACION',

        y='INDICE_MUTACION',

        color='INDICE_MUTACION',

        color_continuous_scale='Reds',

        title='Índice de Mutación Operacional'
    )

    st.plotly_chart(
        fig_mut,
        use_container_width=True
    )

# =========================================================
# CORREDORES ESTRATÉGICOS
# =========================================================

st.subheader(
    "Corredores Estratégicos"
)

corredores = df_filtrado.groupby(
    [
        'DEPARTAMENTO',
        'MUNICIPIO',
        'LATITUD',
        'LONGITUD'
    ]
).agg({

    'EVENTOS':'sum',

    'RIESGO':'mean',

    'IET':'mean'

}).reset_index()

corredores['INDICE_CORREDOR'] = (

    corredores['EVENTOS'] * 0.4

    +

    corredores['RIESGO'] * 40

    +

    corredores['IET'] * 10
)

corredores = corredores.sort_values(
    by='INDICE_CORREDOR',
    ascending=False
)

top_corredores = corredores.head(30)

st.dataframe(
    top_corredores,
    use_container_width=True
)

fig_corr = px.density_mapbox(

    top_corredores,

    lat='LATITUD',

    lon='LONGITUD',

    z='INDICE_CORREDOR',

    radius=25,

    center=dict(
        lat=4.5,
        lon=-74
    ),

    zoom=4,

    height=700,

    mapbox_style='carto-darkmatter'
)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

# =========================================================
# PROSPECTIVA TERRITORIAL
# =========================================================

st.subheader(
    "Prospectiva Territorial"
)

prospectiva = df_filtrado.groupby(
    'MUNICIPIO'
).agg({

    'RIESGO':'mean',

    'EVENTOS':'sum',

    'IET':'mean'

}).reset_index()

prospectiva['RIESGO_FUTURO_30D'] = (

    prospectiva['RIESGO'] * 0.5

    +

    prospectiva['EVENTOS'] * 0.1

    +

    prospectiva['IET'] * 0.4
)

prospectiva = prospectiva.sort_values(
    by='RIESGO_FUTURO_30D',
    ascending=False
)

st.dataframe(
    prospectiva.head(30),
    use_container_width=True
)

fig_future = px.bar(

    prospectiva.head(20),

    x='MUNICIPIO',

    y='RIESGO_FUTURO_30D',

    color='RIESGO_FUTURO_30D',

    color_continuous_scale='Reds',

    title='Proyección Riesgo Próximos 30 Días'
)

st.plotly_chart(
    fig_future,
    use_container_width=True
)

# =========================================================
# MOTOR PREDICTIVO IA
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
        "Precisión Modelo IA",
        round(acc, 3)
    )

    probabilidades = modelo.predict_proba(X)[:,1]

    modelo_df['PROBABILIDAD_ATAQUE'] = probabilidades

    modelo_df['RIESGO_PREDICTIVO'] = (

        modelo_df['PROBABILIDAD_ATAQUE'] * 100
    )

    predicciones = df_filtrado.copy()

    predicciones = predicciones.loc[
        modelo_df.index
    ]

    predicciones['RIESGO_PREDICTIVO'] = (

        modelo_df['RIESGO_PREDICTIVO']
    )

    top_pred = predicciones.groupby(
        'MUNICIPIO'
    )['RIESGO_PREDICTIVO'].mean().reset_index()

    top_pred = top_pred.sort_values(

        by='RIESGO_PREDICTIVO',

        ascending=False
    )

    st.dataframe(
        top_pred.head(30),
        use_container_width=True
    )

    fig_pred = px.bar(

        top_pred.head(20),

        x='MUNICIPIO',

        y='RIESGO_PREDICTIVO',

        color='RIESGO_PREDICTIVO',

        color_continuous_scale='Reds',

        title='Probabilidad Predictiva de Ataque'
    )

    st.plotly_chart(
        fig_pred,
        use_container_width=True
    )

# =========================================================
# DETECCIÓN DE ANOMALÍAS
# =========================================================

st.subheader(
    "Detección de Anomalías Estratégicas"
)

anomalias_features = [

    'EVENTOS',

    'RIESGO',

    'IET',

    'GAO_PRESENTES',

    'UNIDADES_EXPUESTAS'
]

anomalias_validas = True

for col in anomalias_features:

    if col not in df_filtrado.columns:

        anomalias_validas = False

if anomalias_validas:

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

    anomalias_df['SCORE_ANOMALIA'] = detector.decision_function(
        X_anom
    )

    anomalos = anomalias_df[
        anomalias_df['ANOMALIA'] == -1
    ]

    anomalos = anomalos.sort_values(
        by='SCORE_ANOMALIA'
    )

    st.dataframe(
        anomalos.head(30),
        use_container_width=True
    )

    fig_anom = px.scatter(

        anomalos.head(50),

        x='RIESGO',

        y='EVENTOS',

        size='IET',

        color='SCORE_ANOMALIA',

        hover_name='MUNICIPIO',

        title='Municipios Atípicos Detectados',

        color_continuous_scale='Turbo'
    )

    st.plotly_chart(
        fig_anom,
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
