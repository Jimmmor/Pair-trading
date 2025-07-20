import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from constants.tickers import tickers

# Pagina-instellingen
st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Monitor")

# Stijl aanpassingen
custom_style = """
<style>
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Sidebar instellingen
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)
    
    st.markdown("---")
    st.header("📊 Data Instellingen")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y", "2y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", min_value=5, max_value=60, value=20, step=1)
    
    st.markdown("---")
    st.header("⚙️ Trading Parameters")
    zscore_entry_threshold = st.slider("Z-score entry threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

# Data ophalen en verwerken
@st.cache_data
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker}: {e}")
        return pd.DataFrame()

def preprocess_data(data1, data2):
    if isinstance(data1.columns, pd.MultiIndex):
        df1 = data1['Close'].iloc[:, 0].dropna()
        df2 = data2['Close'].iloc[:, 0].dropna()
    else:
        df1 = data1['Close'].dropna()
        df2 = data2['Close'].dropna()
    
    if not isinstance(df1, pd.Series):
        df1 = pd.Series(df1)
    if not isinstance(df2, pd.Series):
        df2 = pd.Series(df2)
    
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    
    df = pd.DataFrame({
        'price1': df1_aligned,
        'price2': df2_aligned
    }).dropna()
    
    return df

# Laad data
data1 = load_data(tickers[name1], periode, interval)
data2 = load_data(tickers[name2], periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins. Probeer een andere combinatie of periode.")
    st.stop()

df = preprocess_data(data1, data2)

if df.empty:
    st.error("Geen overlappende data beschikbaar voor beide coins.")
    st.stop()

# === ANALYSE SECTIE ===
with st.expander("📊 Statistische Analyse", expanded=True):
    st.header("📊 Statistische Analyse")
    
    # Bereken statistieken
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]
    r_squared = model.score(X, y)
    
    df['spread'] = df['price2'] - (alpha + beta * df['price1'])
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    # Rolling correlatie
    df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
    pearson_corr = df['price1'].corr(df['price2'])
    
    # Ratiografiek
    df['ratio'] = df['price1'] / df['price2']
    
    # Returns voor scatterplot
    df['returns1'] = df['price1'].pct_change()
    df['returns2'] = df['price2'].pct_change()
    
    # Layout met tabs
    tab1, tab2 = st.tabs(["Prijsanalyse", "Correlatieanalyse"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
          # Prijsgrafiek met 2 y-assen
                fig_prices = go.Figure()
                
                # Coin 1 (links)
                fig_prices.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['price1'], 
                    name=name1, 
                    line=dict(color='#1f77b4')  # Blauw
                ))
                
                # Coin 2 (rechts)
                fig_prices.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['price2'], 
                    name=name2, 
                    yaxis="y2",
                    line=dict(color='#ff7f0e')  # Oranje
                ))
                
                # Update layout in a cleaner way
                fig_prices.update_layout(
                    title="Prijsverloop",
                    xaxis_title="Datum",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                
                # Separate y-axis updates to avoid the error
                fig_prices.update_yaxes(
                    title_text=f"{name1} Prijs (USD)",
                    title_font=dict(color="#1f77b4"),
                    tickfont=dict(color="#1f77b4")
                )
                
                fig_prices.update_yaxes(
                    title_text=f"{name2} Prijs (USD)",
                    title_font=dict(color="#ff7f0e"),
                    tickfont=dict(color="#ff7f0e"),
                    overlaying="y",
                    side="right",
                    secondary_y=True
                )
                
                st.plotly_chart(fig_prices, use_container_width=True)
            
            # Spread grafiek met groen gebied
            fig_spread = go.Figure()
            
            fig_spread.add_trace(go.Scatter(
                x=df.index, 
                y=df['spread'], 
                name='Spread',
                line=dict(color='#2ca02c'),  # Groen
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.2)'
            ))
            
            fig_spread.update_layout(
                title="Spread tussen de coins",
                xaxis_title="Datum",
                yaxis_title="Spread",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_spread, use_container_width=True)
        
        with col2:
            # Ratiografiek met groen gebied
            fig_ratio = go.Figure()
            
            fig_ratio.add_trace(go.Scatter(
                x=df.index, 
                y=df['ratio'], 
                name=f"{name1}/{name2} Ratio",
                line=dict(color='#9467bd'),  # Paars
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            
            fig_ratio.update_layout(
                title=f"{name1}/{name2} Prijs Ratio",
                xaxis_title="Datum",
                yaxis_title="Ratio",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Z-score grafiek met entry/exit thresholds
            fig_zscore = go.Figure()
            
            fig_zscore.add_trace(go.Scatter(
                x=df.index, 
                y=df['zscore'], 
                name='Z-score',
                line=dict(color='#d62728')  # Rood
            ))
            
            # Entry thresholds
            fig_zscore.add_hline(
                y=zscore_entry_threshold, 
                line=dict(color='#d62728', dash='dash', width=1), 
                annotation_text='Entry Threshold',
                annotation_position="top right"
            )
            fig_zscore.add_hline(
                y=-zscore_entry_threshold, 
                line=dict(color='#d62728', dash='dash', width=1), 
                annotation_text='Entry Threshold',
                annotation_position="bottom right"
            )
            
            # Exit thresholds
            fig_zscore.add_hline(
                y=zscore_exit_threshold, 
                line=dict(color='#17becf', dash='dot', width=1), 
                annotation_text='Exit Threshold',
                annotation_position="top right"
            )
            fig_zscore.add_hline(
                y=-zscore_exit_threshold, 
                line=dict(color='#17becf', dash='dot', width=1), 
                annotation_text='Exit Threshold',
                annotation_position="bottom right"
            )
            
            fig_zscore.update_layout(
                title="Z-score van de spread",
                xaxis_title="Datum",
                yaxis_title="Z-score",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlatiegrafiek
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df.index, 
                y=df['rolling_corr'], 
                name='Rolling Correlatie',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_corr.update_layout(
                title=f"Rolling Correlatie ({corr_window} dagen window)",
                xaxis_title="Datum",
                yaxis_title="Correlatie",
                yaxis_range=[-1, 1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Toon statistieken onder correlatiegrafiek
            st.subheader("Statistieken")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.metric("Pearson Correlatie", f"{pearson_corr:.4f}")
                st.metric("Rolling Correlatie (huidig)", f"{df['rolling_corr'].iloc[-1]:.4f}")
                st.metric("Beta (β)", f"{beta:.4f}")
            
            with stats_col2:
                st.metric("Alpha (α)", f"{alpha:.6f}")
                st.metric("R-squared", f"{r_squared:.4f}")
                st.metric("Spread Volatiliteit", f"{spread_std:.4f}")
        
        with col2:
            # Returns scatterplot in stijl van bijlage
            fig_scatter = px.scatter(
                df.dropna(), 
                x='returns1', 
                y='returns2',
                title=f"Returns Scatterplot ({name1} vs {name2})",
                labels={'returns1': f'{name1} Returns', 'returns2': f'{name2} Returns'},
                color_discrete_sequence=['#17becf'],
                opacity=0.6
            )
            
            # Voeg regressielijn toe
            fig_scatter.add_trace(go.Scatter(
                x=np.linspace(df['returns1'].min(), df['returns1'].max(), 100),
                y=alpha + beta * np.linspace(df['returns1'].min(), df['returns1'].max(), 100),
                mode='lines',
                line=dict(color='#d62728', width=2),
                name='Regressielijn'
            ))
            
            fig_scatter.update_layout(
                xaxis_title=f"{name1} Returns",
                yaxis_title=f"{name2} Returns",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600,
                showlegend=True
            )
            
            # Pas marker grootte en stijl aan
            fig_scatter.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

# === BACKTESTING SECTIE ===
with st.expander("🔙 Backtesting", expanded=False):
    # ... (behoud je bestaande backtesting code, maar pas de stijl aan) ...
    pass

# Export functionaliteit
with st.expander("📤 Export", expanded=False):
    # ... (behoud je bestaande export code) ...
    pass
