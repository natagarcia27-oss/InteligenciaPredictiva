import streamlit as st
import pandas as pd

# ==================================
# CONFIGURACIÓN
# ==================================

st.set_page_config(
    page_title="Inteligencia Territorial",
    layout="wide"
)

# ==================================
# TÍTULO
# ==================================

st.title(
    "Sistema Inteligencia Predictiva Territorial"
)

# ==================================
# CARGAR DATOS
# ==================================

@st.cache_data
def cargar_datos():

    return pd.read_csv(
        "riesgo_municipal.csv"
    )

df = cargar_datos()

# ==================================
# KPIs
# ==================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Municipios",
    df['MUNICIPIO'].nunique()
)

col2.metric(
    "Eventos",
    int(df['EVENTOS'].sum())
)

col3.metric(
    "Registros",
    len(df)
)

# ==================================
# TABLA
# ==================================

st.subheader(
    "Vista Estratégica"
)

st.dataframe(
    df.head(100)
)
