import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise PTAX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    /* Reset de containers para total transparência */
    .section-container, 
    .section-container-alt,
    .chart-container,
    .stats-container,
    .stat-item,
    .stat-card {
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        box-shadow: none !important;
    }

    /* Ajuste específico para alternância suave entre seções */
    .section-container-alt {
        background-color: rgba(128, 128, 128, 0.03) !important;
    }

    /* Ajuste dos cartões de métricas */
    .metric-card {
        background-color: rgba(30, 136, 229, 0.05) !important;
        border: none !important;
        border-left: 4px solid rgba(30, 136, 229, 0.5) !important;
        box-shadow: none !important;
    }

    /* Mantém apenas o gradiente no cabeçalho */
    .header-section {
        background: linear-gradient(90deg, rgba(30, 136, 229, 0.8) 0%, rgba(100, 181, 246, 0.8) 100%) !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Ajustes para os cartões de estatísticas */
    .stat-card,
    .stat-item {
        background-color: rgba(128, 128, 128, 0.03) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
    }

    /* Remove bordas brancas dos gráficos */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }

    /* Ajuste dos textos para melhor contraste */
    .metric-card h4,
    .stat-card h4,
    .stat-item h4 {
        color: rgba(255, 255, 255, 0.6) !important;
    }

    .metric-card h2,
    .stat-card h2,
    .stat-item h2 {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Ajuste do fundo dos botões de download */
    .stDownloadButton button {
        background-color: rgba(30, 136, 229, 0.1) !important;
        border: 1px solid rgba(30, 136, 229, 0.2) !important;
        transition: all 0.3s ease;
    }

    .stDownloadButton button:hover {
        background-color: rgba(30, 136, 229, 0.2) !important;
        border: 1px solid rgba(30, 136, 229, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cabeçalho
st.markdown("""
    <div class="header-section">
        <h1 style='margin-bottom: 0;'>Análise e Projeção de PTAX</h1>
        <p style='font-size: 1.1rem; margin-top: 0.5rem;'>Sistema de análise e projeção da taxa de câmbio USD/BRL</p>
    </div>
    """, unsafe_allow_html=True)

# Configuração da sidebar
with st.sidebar:
    st.markdown("""
        <h3 style='text-align: center; color: #1E88E5; margin-bottom: 2rem;'>
            ⚙️ Configurações
        </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📊 Parâmetros do Modelo")
    with st.expander("ℹ️ Ajuste os parâmetros do modelo", expanded=True):
        n_estimators = st.slider(
            "Número de árvores",
            min_value=50,
            max_value=300,
            value=100,
            step=50,
            help="Quantidade de árvores na Random Forest. Mais árvores = modelo mais robusto, mas mais lento"
        )
        
        test_size = st.slider(
            "Tamanho do conjunto de teste",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
            help="Porcentagem dos dados usada para teste do modelo"
        ) / 100
        
        forecast_days = st.slider(
            "Dias para projeção",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            help="Quantidade de dias futuros para projeção"
        )
    
    st.markdown("#### 📅 Período de Análise")
    min_date = pd.to_datetime('2000-01-01')
    max_date = pd.to_datetime('today')
    start_date = st.date_input(
        "Data inicial da análise",
        value=pd.to_datetime('2020-01-01'),
        min_value=min_date,
        max_value=max_date,
        help="Selecione a data de início para análise"
    )

# Função para obter dados
@st.cache_data(ttl=3600)
def get_ptax_data(start_date):
    try:
        with st.spinner('📊 Obtendo dados do Yahoo Finance...'):
            df = yf.download('USDBRL=X', start=start_date)
            if df.empty:
                st.error("❌ Nenhum dado foi retornado do Yahoo Finance")
                return pd.DataFrame()
            df = df[['Close']]
            df.columns = ['PTAX']
            st.success(f"✅ Dados obtidos com sucesso! Total de {len(df):,} registros.")
            return df
    except Exception as e:
        st.error(f"❌ Erro ao obter dados: {str(e)}")
        return pd.DataFrame()

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
    
    # Seção 1: Análise Histórica
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📈 Análise Histórica</h2>', unsafe_allow_html=True)
    
    # Estatísticas principais
    stats = data['PTAX'].describe()
    st.markdown("""
        <div class="stats-overview">
            <div class="stat-card">
                <h4 style='color: #666;'>Média</h4>
                <h2 style='color: #1E88E5; margin: 0;'>R$ {:.4f}</h2>
            </div>
            <div class="stat-card">
                <h4 style='color: #666;'>Mediana</h4>
                <h2 style='color: #1E88E5; margin: 0;'>R$ {:.4f}</h2>
            </div>
            <div class="stat-card">
                <h4 style='color: #666;'>Mínimo</h4>
                <h2 style='color: #1E88E5; margin: 0;'>R$ {:.4f}</h2>
            </div>
            <div class="stat-card">
                <h4 style='color: #666;'>Máximo</h4>
                <h2 style='color: #1E88E5; margin: 0;'>R$ {:.4f}</h2>
            </div>
        </div>
    """.format(
        stats['mean'],
        stats['50%'],
        stats['min'],
        stats['max']
    ), unsafe_allow_html=True)
    
    # Gráfico principal
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    show_ma5 = st.checkbox("Mostrar Média Móvel 5 dias", value=True)
    show_ma20 = st.checkbox("Mostrar Média Móvel 20 dias", value=True)
    
    fig = go.Figure()
    
    # Linha principal PTAX
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['PTAX'],
        name='PTAX',
        line=dict(color='#1E88E5', width=2)
    ))

    if show_ma5:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['MA5'],
            name='MM5',
            line=dict(color='#FFA726', width=2, dash='dot')
        ))
        
    if show_ma20:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['MA20'],
            name='MM20',
            line=dict(color='#EF5350', width=2, dash='dash')
        ))
    
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Seção 2: Volatilidade
    st.markdown('<div class="section-container-alt">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Análise de Volatilidade</h2>', unsafe_allow_html=True)
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Volatility'],
        fill='tozeroy',
        name='Volatilidade',
        line=dict(color='#7CB342', width=2)
    ))
    
    fig_vol.update_layout(
        title="Volatilidade (Desvio Padrão 20 dias)",
        template='plotly_white',
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Seção 3: Modelagem
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">🤖 Modelagem e Projeção</h2>', unsafe_allow_html=True)
    
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'MA5', 'MA20', 'Volatility']
    data_ml = data.dropna()
    
    X = data_ml[features]
    y = data_ml['PTAX']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    with st.spinner('🔄 Treinando o modelo...'):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col_metrics = st.columns(2)
        with col_metrics[0]:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style='color: #666; margin-bottom: 0.5rem;'>Erro Quadrático Médio</h4>
                    <h2 style='color: #1E88E5; margin: 0;'>{mse:.4f}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col_metrics[1]:
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style='color: #666; margin-bottom: 0.5rem;'>R² Score</h4>
                    <h2 style='color: #1E88E5; margin: 0;'>{r2:.4f}</h2>
                </div>
            """, unsafe_allow_html=True)
    
    # Seção 4: Projeção
    st.markdown('<div class="section-container-alt">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">🔮 Projeção para os Próximos {forecast_days} Dias</h2>', unsafe_allow_html=True)
    
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
    
    fig_proj = go.Figure()
    
    # Dados históricos
    fig_proj.add_trace(go.Scatter(
        x=data.index,
        y=data['PTAX'],
        name='Histórico',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Projeção
    fig_proj.add_trace(go.Scatter(
        x=future_data.index,
        y=future_data['PTAX_Predicted'],
        name='Projeção',
        line=dict(color='#FFA726', width=2, dash='dash')
    ))
    
    fig_proj.update_layout(
        title='Histórico e Projeção PTAX',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Seção 5: Download
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📥 Download dos Dados</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="download-container">', unsafe_allow_html=True)
    csv_historical = data.to_csv().encode('utf-8')
    st.download_button(
        label="📊 Download Dados Históricos",
        data=csv_historical,
        file_name='ptax_historical.csv',
        mime='text/csv',
        key='download1',
        help='Baixar série histórica completa'
    )
    
    csv_projection = future_data[['PTAX_Predicted']].to_csv().encode('utf-8')
    st.download_button(
        label="🔮 Download Projeções",
        data=csv_projection,
        file_name='ptax_projections.csv',
        mime='text/csv',
        key='download2',
        help='Baixar dados das projeções futuras'
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("❌ Não foi possível obter os dados. Por favor, verifique sua conexão com a internet ou tente um período diferente.")