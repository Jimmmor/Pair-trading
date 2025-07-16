# DEEL 1: Algemene instellingen, data ophalen en basisberekeningen

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Pagina instellingen
st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Monitor")

st.markdown("""
Vergelijk twee coins, bereken de spread, Z-score en statistieken (alpha, beta, R², Pearson R).  
Gebruik dit voor pairs trading. Inclusief aanbeveling op basis van ratio, correlatie en Z-score.
""")

# Beschikbare tickers
tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Shiba Inu (SHIB)": "SHIB-USD",
    "TRON (TRX)": "TRX-USD",
    "Uniswap (UNI)": "UNI-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Stellar (XLM)": "XLM-USD",
    "VeChain (VET)": "VET-USD",
    "NEAR Protocol (NEAR)": "NEAR-USD",
    "Aptos (APT)": "APT-USD",
    "Filecoin (FIL)": "FIL-USD",
    "The Graph (GRT)": "GRT-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "Tezos (XTZ)": "XTZ-USD",
    "Hedera (HBAR)": "HBAR-USD",
    "Fantom (FTM)": "FTM-USD",
    "EOS (EOS)": "EOS-USD",
    "Zcash (ZEC)": "ZEC-USD",
    "Dash (DASH)": "DASH-USD",
    "Chiliz (CHZ)": "CHZ-USD",
    "THETA (THETA)": "THETA-USD",
    "Internet Computer (ICP)": "ICP-USD",
    "Arbitrum (ARB)": "ARB-USD",
    "Optimism (OP)": "OP-USD",
    "Injective (INJ)": "INJ-USD",
    "SUI (SUI)": "SUI-USD",
    "Lido DAO (LDO)": "LDO-USD",
    "Aave (AAVE)": "AAVE-USD",
    "Maker (MKR)": "MKR-USD",
    "Curve DAO (CRV)": "CRV-USD",
    "1inch (1INCH)": "1INCH-USD",
    "Gala (GALA)": "GALA-USD",
    "Render (RNDR)": "RNDR-USD"
}

# Sidebar: selectie van coins, periode en instellingen
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)

    st.markdown("---")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", 5, 60, 20)

    st.markdown("---")
    zscore_entry_threshold = st.slider("Z-score entry threshold", 1.0, 5.0, 2.0, 0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", 0.0, 2.0, 0.5, 0.1)

# Tickers vertalen
coin1 = tickers[name1]
coin2 = tickers[name2]

# Caching van data
@st.cache_data
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker}: {e}")
        return pd.DataFrame()

data1 = load_data(coin1, periode, interval)
data2 = load_data(coin2, periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins. Probeer een andere combinatie of periode.")
    st.stop()

# Extract sluitprijzen
df1 = data1['Close'] if 'Close' in data1 else data1.iloc[:, 0]
df2 = data2['Close'] if 'Close' in data2 else data2.iloc[:, 0]

# Serie conversie en align
if not isinstance(df1, pd.Series): df1 = pd.Series(df1)
if not isinstance(df2, pd.Series): df2 = pd.Series(df2)
df1, df2 = df1.align(df2, join='inner')

df = pd.DataFrame({'price1': df1, 'price2': df2}).dropna()
if df.empty:
    st.error("Geen overlappende data beschikbaar voor beide coins.")
    st.stop()

# Regressie model
X = df['price1'].values.reshape(-1, 1)
y = df['price2'].values
model = LinearRegression().fit(X, y)

alpha = model.intercept_
beta = model.coef_[0]
r_squared = model.score(X, y)

df['spread'] = df['price2'] - (alpha + beta * df['price1'])
df['zscore'] = (df['spread'] - df['spread'].mean()) / df['spread'].std()
df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
# Visuele weergave van prijzen
st.subheader("📊 Prijsontwikkeling van beide coins")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['price1'], mode='lines', name=name1))
fig.add_trace(go.Scatter(x=df.index, y=df['price2'], mode='lines', name=name2))
fig.update_layout(title='Coin Prijzen Over Tijd', xaxis_title='Datum', yaxis_title='Prijs', height=400)
st.plotly_chart(fig, use_container_width=True)

# Spread & Z-score grafiek
st.subheader("📉 Spread & Z-score Analyse")
col1, col2 = st.columns(2)

with col1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))
    fig2.update_layout(title='Spread Tussen Coins', xaxis_title='Datum', yaxis_title='Spread', height=300)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['zscore'], mode='lines', name='Z-score'))
    fig3.add_hline(y=zscore_entry_threshold, line_dash='dot', line_color='green', annotation_text='Entry threshold', annotation_position='top left')
    fig3.add_hline(y=-zscore_entry_threshold, line_dash='dot', line_color='green')
    fig3.add_hline(y=zscore_exit_threshold, line_dash='dash', line_color='red', annotation_text='Exit threshold', annotation_position='bottom left')
    fig3.add_hline(y=-zscore_exit_threshold, line_dash='dash', line_color='red')
    fig3.update_layout(title='Z-score van Spread', xaxis_title='Datum', yaxis_title='Z-score', height=300)
    st.plotly_chart(fig3, use_container_width=True)

# Rolling correlatie
st.subheader("🔗 Rolling Correlatie")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df.index, y=df['rolling_corr'], mode='lines', name='Rolling Correlatie'))
fig4.update_layout(title=f'Rolling Correlatie ({corr_window} dagen)', xaxis_title='Datum', yaxis_title='Correlatie', height=300)
st.plotly_chart(fig4, use_container_width=True)

# Statistieken en aanbeveling
st.subheader("📌 Statistieken & Advies")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Alpha", f"{alpha:.4f}")
metric_col2.metric("Beta", f"{beta:.4f}")
metric_col3.metric("R² (Regressie)", f"{r_squared:.4f}")

pearson_r = df['price1'].corr(df['price2'])
st.metric("📈 Pearson Correlatie", f"{pearson_r:.4f}")

# Aanbeveling logica
z = df['zscore'].iloc[-1]
advies = ""
kleur = ""

if abs(z) > zscore_entry_threshold:
    if z > 0:
        advies = f"Short {name2}, Long {name1}"
        kleur = "red"
    else:
        advies = f"Long {name2}, Short {name1}"
        kleur = "green"
elif abs(z) < zscore_exit_threshold:
    advies = "📤 Sluit de positie - Z-score binnen neutraal bereik"
    kleur = "blue"
else:
    advies = "⏳ Wacht op entry-signaal - Geen duidelijke afwijking"
    kleur = "gray"

st.markdown(f"### 📢 Advies: <span style='color:{kleur}'>{advies}</span>", unsafe_allow_html=True)

