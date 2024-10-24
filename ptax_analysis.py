import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuração da página Streamlit
st.set_page_config(page_title="Análise PTAX", layout="wide")
st.title("Análise e Projeção de PTAX")

# Função para obter dados PTAX
@st.cache_data
def get_ptax_data(start_date):
    try:
        st.write("Tentando obter dados do Yahoo Finance...")
        # Usando yfinance para obter dados do Yahoo Finance
        df = yf.download('USDBRL=X', start=start_date)
        if df.empty:
            st.error("Nenhum dado foi retornado do Yahoo Finance")
            return pd.DataFrame()
        df = df[['Close']]  # Pegando apenas o preço de fechamento
        df.columns = ['PTAX']
        st.success(f"Dados obtidos com sucesso. Total de {len(df)} registros.")
        return df
    except Exception as e:
        st.error(f"Erro ao obter dados: {str(e)}")
        return pd.DataFrame()

# Debug info
st.write("Versão do Python:", pd.show_versions())
st.write("Versão do YFinance:", yf.__version__)

# Sidebar para controles
st.sidebar.header("Configurações")

# Seleção de período
min_date = pd.to_datetime('2000-01-01')
max_date = pd.to_datetime('today')
start_date = st.sidebar.date_input(
    "Selecione a data inicial",
    value=pd.to_datetime('2020-01-01'),
    min_value=min_date,
    max_value=max_date
)

st.write("Data inicial selecionada:", start_date)

# Obtenção dos dados
data = get_ptax_data(start_date)

if data.empty:
    st.error("Não foi possível obter os dados. Por favor, verifique sua conexão com a internet ou tente um período diferente.")
else:
    st.write("Dados obtidos com sucesso!")
    st.write("Primeiras linhas dos dados:")
    st.write(data.head())

# [Resto do código continua igual...]