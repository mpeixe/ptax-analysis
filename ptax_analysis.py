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
        # Usando yfinance para obter dados do Yahoo Finance
        df = yf.download('USDBRL=X', start=start_date)
        df = df[['Close']]  # Pegando apenas o preço de fechamento
        df.columns = ['PTAX']
        return df
    except Exception as e:
        st.error(f"Erro ao obter dados: {str(e)}")
        return pd.DataFrame()

# [Resto do código continua igual...]