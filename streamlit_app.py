import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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

#sidebar
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)
    
    st.markdown("---")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", min_value=5, max_value=60, value=20, step=1)
    
    st.markdown("---")
    zscore_entry_threshold = st.slider("Z-score entry threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

# Data ophalen met caching
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
# === Sidebar uitbreiden met thresholds voor z-score trade signalen ===
with st.sidebar:
    zscore_entry_threshold = st.number_input(
        "Entry Z-score threshold", min_value=0.5, max_value=5.0, value=2.0, step=0.1
    )
    zscore_exit_threshold = st.number_input(
        "Exit Z-score threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1
    )

# Start- en einddatum voor samenvatting bepalen
start_date = data1.index.min().strftime("%Y-%m-%d") if not data1.empty else "N/A"
end_date = data1.index.max().strftime("%Y-%m-%d") if not data1.empty else "N/A"# === Trade signalen gebaseerd op z-score thresholds ===

# Signalen: long entry als zscore < -entry_threshold, short entry als zscore > entry_threshold
df['long_entry'] = df['zscore'] < -zscore_entry_threshold
df['short_entry'] = df['zscore'] > zscore_entry_threshold
df['exit'] = df['zscore'].abs() < zscore_exit_threshold

# Huidige positie bepalen (laatste data punt)
if df['long_entry'].iloc[-1]:
    current_position = f"Long Spread (koop {name2}, verkoop {name1})"
elif df['short_entry'].iloc[-1]:
    current_position = f"Short Spread (verkoop {name2}, koop {name1})"
elif df['exit'].iloc[-1]:
    current_position = "Exit positie (geen trade)"
else:
    current_position = "Geen duidelijk signaal"

st.subheader("🚦 Huidige trade signaal")
st.write(f"**Z-score laatste waarde:** {df['zscore'].iloc[-1]:.2f}")
st.write(f"**Signaal:** {current_position}")

# === Entry, exit, stoploss niveaus visualiseren ===

# Entry lijnen in spread termen
entry_long_level = -zscore_entry_threshold * spread_std + spread_mean
entry_short_level = zscore_entry_threshold * spread_std + spread_mean
exit_level_pos = zscore_exit_threshold * spread_std + spread_mean
exit_level_neg = -zscore_exit_threshold * spread_std + spread_mean

fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))

# Entry en exit lijnen tekenen
fig_signal.add_hline(y=entry_long_level, line=dict(color='green', dash='dash'), annotation_text='Long Entry', annotation_position='bottom left')
fig_signal.add_hline(y=entry_short_level, line=dict(color='red', dash='dash'), annotation_text='Short Entry', annotation_position='top left')
fig_signal.add_hline(y=exit_level_pos, line=dict(color='blue', dash='dot'), annotation_text='Exit', annotation_position='top right')
fig_signal.add_hline(y=exit_level_neg, line=dict(color='blue', dash='dot'), annotation_text='Exit', annotation_position='bottom right')

fig_signal.update_layout(title="Spread met Entry en Exit niveaus", yaxis_title="Spread", xaxis_title="Datum")

st.plotly_chart(fig_signal, use_container_width=True)

# === Stoploss niveau (voorbeeld) ===
stoploss_pct = 0.05  # 5% stoploss van spread, aanpasbaar

stoploss_upper = spread_mean * (1 + stoploss_pct)
stoploss_lower = spread_mean * (1 - stoploss_pct)

st.subheader("🛑 Stoploss niveau (voorbeeld)")
st.write(f"Stoploss boven: {stoploss_upper:.4f}")
st.write(f"Stoploss onder: {stoploss_lower:.4f}")
# === Samenvatting en toelichting ===

st.header("📊 Samenvatting van de pairs trading analyse")

st.markdown(f"""
- **Asset 1:** {name1}  
- **Asset 2:** {name2}  
- **Periode:** {df.index.min().date()} tot {df.index.max().date()}  
- **Data punten:** {len(df)}  

**Regressie resultaten:**  
- Alpha: {alpha:.6f}  
- Beta: {beta:.6f}  
- R²: {r_squared:.4f}  

**Spread statistieken:**  
- Gemiddelde spread: {spread_mean:.4f}  
- Standaarddeviatie spread: {spread_std:.4f}  

**Laatste z-score:** {df['zscore'].iloc[-1]:.2f}

""")

st.header("💡 Mogelijke uitbreidingen")

st.markdown("""
- Voeg dynamische parameters toe via sliders, bijvoorbeeld z-score thresholds en stoploss percentage.  
- Implementeer backtesting van de strategy om performance over tijd te bekijken.  
- Gebruik real-time data feeds voor live trading signalen.  
- Voeg meerdere paren toe met rangschikking op basis van cointegratie.  
- Integreer een ordermanagement systeem (via broker API).  
- Visualiseer winst/verlies scenario's en risicomanagement.  
""")

st.write("### Bedankt voor het gebruiken van deze pairs trading tool!")

# Optioneel: mogelijkheid om data te exporteren
if st.button("Exporteer analyse naar CSV"):
    csv = df.to_csv(index=True)
    st.download_button(label="Download CSV", data=csv, file_name="pairs_trading_analysis.csv", mime='text/csv')



