import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Pagina-instellingen
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

# Sidebar
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)
    
    st.markdown("---")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", min_value=5, max_value=60, value=20, step=1)

coin1 = tickers[name1]
coin2 = tickers[name2]

# Data ophalen
@st.cache_data
def load_data(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval)
    except:
        return pd.DataFrame()

data1 = load_data(coin1, periode, interval)
data2 = load_data(coin2, periode, interval)

# Validatie
if data1.empty or data2.empty or "Close" not in data1.columns or "Close" not in data2.columns:
    st.error("❌ Data kon niet geladen worden.")
    st.stop()

# Data voorbereiden
s1 = data1["Close"].copy()
s1.name = coin1
s2 = data2["Close"].copy()
s2.name = coin2
df = pd.concat([s1, s2], axis=1).dropna()

# Regressie
x = df[[coin2]].values
y = df[coin1].values
reg = LinearRegression().fit(x, y)

alpha = reg.intercept_
beta = reg.coef_[0]
r2 = reg.score(x, y)

# Pearson R
pearson_r = df[coin1].corr(df[coin2])

# Spread & Z-score
df["Spread"] = df[coin1] - (alpha + beta * df[coin2])
spread_mean = df["Spread"].mean()
spread_std = df["Spread"].std()
df["Z-score"] = (df["Spread"] - spread_mean) / spread_std

# Ratio en correlatie
df["Ratio"] = df[coin1] / df[coin2]
df["Rolling Correlatie"] = df[coin1].rolling(window=corr_window).corr(df[coin2])
mean_ratio = df["Ratio"].mean()

# Statistieken uitleg
with st.expander("📊 Statistieken & Evaluatie"):
    st.markdown(f"""
    - **Alpha (α): {alpha:.4f}** → Verwachte waarde van {coin1} als {coin2} nul is.
    - **Beta (β): {beta:.4f}** → Voor elke 1% verandering in {coin2}, beweegt {coin1} gemiddeld {beta:.2f}% mee.
    - **R²: {r2:.4f}** → {r2 * 100:.1f}% van de beweging in {coin1} wordt verklaard door {coin2}.
    - **Pearson R: {pearson_r:.4f}** → De lineaire samenhang is {'sterk' if abs(pearson_r) > 0.7 else 'matig' if abs(pearson_r) > 0.4 else 'zwak'}.
    - **Gemiddelde Spread: {spread_mean:.4f}** | σ: {spread_std:.4f}
    - **Gemiddelde Ratio {coin1}/{coin2}: {mean_ratio:.4f}**
    """)
# Scatterplot + regressielijn
with st.expander("📊 Scatterplot"):
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=df[coin2], y=df[coin1], mode='markers', name='Punten', marker=dict(color='lightblue')))
    x_range = np.linspace(df[coin2].min(), df[coin2].max(), 100)
    fig_scatter.add_trace(go.Scatter(x=x_range, y=alpha + beta * x_range, mode='lines', name='Regressielijn', line=dict(color='orange')))
    fig_scatter.update_layout(title="Scatterplot & Regressielijn", xaxis_title=coin2, yaxis_title=coin1, template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)
# Correlatie
with st.expander("📊 Correlatie"):
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=df.index,
        y=df["Rolling Correlatie"],
        mode='lines',
        name="Rolling Correlatie",
        line=dict(color='lightgreen'),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
))
fig_corr.update_layout(title="Rolling Correlatie", xaxis_title="Datum", yaxis_title="Correlatie", template="plotly_dark", yaxis=dict(range=[-1, 1]))
st.plotly_chart(fig_corr, use_container_width=True)
# Ratio-grafiek met groene vulling
with st.expander("📈 Ratio"):
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=df.index,
        y=df["Ratio"],
        mode='lines',
        name="Ratio",
        line=dict(color='red'),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    fig_ratio.update_layout(title=f"Ratio {coin1}/{coin2}", xaxis_title="Datum", yaxis_title="Ratio", template="plotly_dark")
    st.plotly_chart(fig_ratio, use_container_width=True)
# Prijsvergelijking
with st.expander("📈 Prijsvergelijking"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[coin1], name=coin1, yaxis="y1", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df[coin2], name=coin2, yaxis="y2", line=dict(color="orange")))
    fig.update_layout(title="Prijsvergelijking", xaxis=dict(title="Datum"),
                      yaxis=dict(title=coin1, side="left"),
                      yaxis2=dict(title=coin2, overlaying="y", side="right"),
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
# Z-score
st.subheader("📉 Z-score en Entry-signalen")
fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=df.index, y=df["Z-score"], mode='lines', name="Z-score", line=dict(color="skyblue")))
fig_z.add_trace(go.Scatter(x=df.index, y=[-1]*len(df), mode='lines', name='Long Entry', line=dict(dash='dash', color='green')))
fig_z.add_trace(go.Scatter(x=df.index, y=[1]*len(df), mode='lines', name='Short Entry', line=dict(dash='dash', color='red')))
fig_z.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Mean (0)', line=dict(dash='dot', color='gray')))
fig_z.update_layout(title="Z-score met Entry Drempels", xaxis_title="Datum", yaxis_title="Z-score", template="plotly_dark")
st.plotly_chart(fig_z, use_container_width=True)

# Analyse en aanbeveling
st.subheader("🤖 Aanbeveling op basis van actuele data")

latest_z = df["Z-score"].iloc[-1]
latest_ratio = df["Ratio"].iloc[-1]
latest_corr = df["Rolling Correlatie"].iloc[-1]

def interpretatie():
    actie = ""
    if latest_z > 1:
        actie = "📉 **Short** de spread — verwacht dat de ratio weer naar het gemiddelde daalt."
    elif latest_z < -1:
        actie = "📈 **Long** de spread — verwacht dat de ratio terugkeert naar het gemiddelde."
    else:
        actie = "⏸️ Geen actie — Z-score is binnen neutrale zone."

    richting = "boven" if latest_ratio > mean_ratio else "onder"
    correlatie = (
        "sterke correlatie" if latest_corr > 0.7 else
        "matige correlatie" if latest_corr > 0.4 else
        "zwakke correlatie"
    )

    return f"""
    - De **huidige Z-score is {latest_z:.2f}** → {actie}  
    - De **ratio staat {richting} het gemiddelde ({mean_ratio:.4f})**.  
    - De **rolling correlatie is {latest_corr:.2f}** → {correlatie}.  
    """

st.markdown(interpretatie())
