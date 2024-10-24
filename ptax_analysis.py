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
        df = yf.download('USDBRL=X', start=start_date)
        if df.empty:
            st.error("Nenhum dado foi retornado do Yahoo Finance")
            return pd.DataFrame()
        df = df[['Close']]
        df.columns = ['PTAX']
        st.success(f"Dados obtidos com sucesso. Total de {len(df)} registros.")
        return df
    except Exception as e:
        st.error(f"Erro ao obter dados: {str(e)}")
        return pd.DataFrame()

# Sidebar para controles
st.sidebar.header("Configurações")

# Configurações do modelo
st.sidebar.subheader("Configurações do Modelo")
n_estimators = st.sidebar.slider("Número de árvores na Random Forest", 50, 300, 100, 50)
test_size = st.sidebar.slider("Tamanho do conjunto de teste (%)", 10, 40, 20, 5) / 100
forecast_days = st.sidebar.slider("Dias para projeção", 5, 60, 30, 5)

# Seleção de período
min_date = pd.to_datetime('2000-01-01')
max_date = pd.to_datetime('today')
start_date = st.sidebar.date_input(
    "Selecione a data inicial",
    value=pd.to_datetime('2020-01-01'),
    min_value=min_date,
    max_value=max_date
)

# Obtenção dos dados
data = get_ptax_data(start_date)

if not data.empty:
    # Pré-processamento
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # Features técnicas
    data['MA5'] = data['PTAX'].rolling(window=5).mean()
    data['MA20'] = data['PTAX'].rolling(window=20).mean()
    data['Volatility'] = data['PTAX'].rolling(window=20).std()
    
    # Layout em duas colunas para visualização
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Série Histórica PTAX")
        show_ma5 = st.checkbox("Mostrar Média Móvel 5 dias", value=True)
        show_ma20 = st.checkbox("Mostrar Média Móvel 20 dias", value=True)
        
        fig = px.line(data, x='Date', y='PTAX', title='Série Histórica PTAX')
        if show_ma5:
            fig.add_scatter(x=data['Date'], y=data['MA5'], name='MM5', line=dict(color='orange'))
        if show_ma20:
            fig.add_scatter(x=data['Date'], y=data['MA20'], name='MM20', line=dict(color='red'))
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Estatísticas Descritivas")
        st.write(data['PTAX'].describe())
        
        fig_vol = px.line(data, x='Date', y='Volatility', title='Volatilidade (Desvio Padrão 20 dias)')
        st.plotly_chart(fig_vol)
    
    # Preparação para Machine Learning
    st.subheader("Modelagem com Random Forest")
    
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'MA5', 'MA20', 'Volatility']
    data_ml = data.dropna()  # Remove linhas com NaN
    
    X = data_ml[features]
    y = data_ml['PTAX']
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Treinamento do modelo
    with st.spinner('Treinando o modelo...'):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Métricas de avaliação
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Erro Quadrático Médio", f"{mse:.4f}")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
    
    # Projeção futura
    st.subheader(f"Projeção para os Próximos {forecast_days} Dias")
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    future_data = pd.DataFrame(index=future_dates)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data['Day'] = future_data.index.day
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['MA5'] = data['PTAX'].iloc[-5:].mean()
    future_data['MA20'] = data['PTAX'].iloc[-20:].mean()
    future_data['Volatility'] = data['PTAX'].iloc[-20:].std()
    
    future_pred = model.predict(future_data[features])
    future_data['PTAX_Predicted'] = future_pred
    
    # Gráfico com histórico e projeção
    fig_proj = px.line()
    fig_proj.add_scatter(x=data.index, y=data['PTAX'], name='Histórico')
    fig_proj.add_scatter(x=future_data.index, y=future_data['PTAX_Predicted'], 
                        name='Projeção', line=dict(dash='dash'))
    fig_proj.update_layout(title='Histórico e Projeção PTAX', 
                          xaxis_title='Data', 
                          yaxis_title='PTAX')
    st.plotly_chart(fig_proj)
    
    # Download dos dados
    st.subheader("Download dos Dados")
    col5, col6 = st.columns(2)
    
    with col5:
        csv_historical = data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Dados Históricos",
            data=csv_historical,
            file_name='ptax_historical.csv',
            mime='text/csv'
        )
    
    with col6:
        csv_projection = future_data[['PTAX_Predicted']].to_csv().encode('utf-8')
        st.download_button(
            label="Download Projeções",
            data=csv_projection,
            file_name='ptax_projections.csv',
            mime='text/csv'
        )

else:
    st.error("Não foi possível obter os dados. Por favor, verifique sua conexão com a internet ou tente um período diferente.")