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
# Check of data binnen is
if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins. Probeer een andere combinatie of periode.")
    st.stop()

# Data prijzen - sluitingsprijzen gebruiken
prices1 = data1['Close']
prices2 = data2['Close']

# Op gelijke lengte brengen (inner join op index)
# Functie om data op te halen en om te zetten naar een Series
def get_data(ticker):
    prices = all_data[ticker]
    return pd.Series(prices, name=ticker)

# Haal de correcte Series op voor beide assets
prices1 = get_data(ticker1)
prices2 = get_data(ticker2)

# Combineer ze in een dataframe
df = pd.DataFrame({'price1': prices1, 'price2': prices2}).dropna()


# Linear regression price2 ~ price1 (beta * price1 + alpha)
X = df[['price1']].values
y = df['price2'].values
model = LinearRegression().fit(X, y)
beta = model.coef_[0]
alpha = model.intercept_
r2 = model.score(X, y)
corr = np.corrcoef(df['price1'], df['price2'])[0, 1]

# Spread berekenen: residuals van de regressie
df['spread'] = df['price2'] - (beta * df['price1'] + alpha)

# Spread stats
spread_mean = df['spread'].mean()
spread_std = df['spread'].std()

# Z-score van spread
df['zscore'] = (df['spread'] - spread_mean) / spread_std

# Rolling correlatie over gekozen window
df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])

# Plot 1: Prijzen van beide coins
fig_prices = go.Figure()
fig_prices.add_trace(go.Scatter(x=df.index, y=df['price1'], mode='lines', name=name1))
fig_prices.add_trace(go.Scatter(x=df.index, y=df['price2'], mode='lines', name=name2))
fig_prices.update_layout(title="Sluitingsprijzen", yaxis_title="Prijs (USD)", xaxis_title="Datum")

# Plot 2: Spread en Z-score
fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))
fig_spread.add_trace(go.Scatter(x=df.index, y=df['zscore'], mode='lines', name='Z-score', yaxis='y2'))

fig_spread.update_layout(
    title="Spread en Z-score",
    yaxis=dict(title='Spread', side='left'),
    yaxis2=dict(title='Z-score', overlaying='y', side='right'),
    xaxis_title="Datum"
)

# Plot 3: Rolling correlatie
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=df.index, y=df['rolling_corr'], mode='lines', name='Rolling correlatie'))
fig_corr.update_layout(
    title=f"Rolling correlatie (window={corr_window} dagen)",
    yaxis_title="Correlatie",
    xaxis_title="Datum",
    yaxis=dict(range=[-1,1])
)

# Resultaten en statistieken tonen
st.subheader("📊 Statistieken")
st.write(f"**Regressie formule:** {name2} = {beta:.4f} × {name1} + {alpha:.4f}")
st.write(f"**R²:** {r2:.4f}")
st.write(f"**Pearson correlatie:** {corr:.4f}")
st.write(f"**Gemiddelde spread:** {spread_mean:.4f}")
st.write(f"**Standaarddeviatie spread:** {spread_std:.4f}")

# Interpretatie aanbeveling
st.subheader("💡 Trading aanbeveling")
if abs(corr) < 0.7:
    st.warning("Let op: correlatie tussen coins is laag, pairs trading mogelijk minder betrouwbaar.")
if r2 < 0.6:
    st.warning("Let op: R² is laag, regressiefit is mogelijk niet sterk.")
if abs(df['zscore'].iloc[-1]) > 2:
    direction = "short" if df['zscore'].iloc[-1] > 2 else "long"
    st.success(f"Z-score extreem hoog ({df['zscore'].iloc[-1]:.2f}): Overweeg een **{direction} positie** op de spread.")
else:
    st.info(f"Z-score normaal ({df['zscore'].iloc[-1]:.2f}): Geen directe trade signalen.")

# Grafieken tonen
st.plotly_chart(fig_prices, use_container_width=True)
st.plotly_chart(fig_spread, use_container_width=True)
st.plotly_chart(fig_corr, use_container_width=True)
# === Trade signalen gebaseerd op z-score thresholds ===

entry_threshold = zscore_entry_threshold
exit_threshold = zscore_exit_threshold

# Signalen: long spread = zscore < -entry_threshold, short spread = zscore > entry_threshold
df['long_entry'] = df['zscore'] < -entry_threshold
df['short_entry'] = df['zscore'] > entry_threshold
df['exit'] = df['zscore'].abs() < exit_threshold

# Huidige positie bepalen (laatste data punt)
if df['long_entry'].iloc[-1]:
    current_position = "Long Spread (koop coin2, verkoop coin1)"
elif df['short_entry'].iloc[-1]:
    current_position = "Short Spread (verkoop coin2, koop coin1)"
elif df['exit'].iloc[-1]:
    current_position = "Exit positie (geen trade)"
else:
    current_position = "Geen duidelijk signaal"

st.subheader("🚦 Huidige trade signaal")
st.write(f"**Z-score laatste waarde:** {df['zscore'].iloc[-1]:.2f}")
st.write(f"**Signaal:** {current_position}")

# === Entry, exit, stoploss niveaus visualiseren ===

# Entry lijnen
entry_long_level = -entry_threshold * spread_std + spread_mean
entry_short_level = entry_threshold * spread_std + spread_mean
exit_level_pos = exit_threshold * spread_std + spread_mean
exit_level_neg = -exit_threshold * spread_std + spread_mean

fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))

# Entry en exit lijnen
fig_signal.add_hline(y=entry_long_level, line=dict(color='green', dash='dash'), annotation_text='Long Entry', annotation_position='bottom left')
fig_signal.add_hline(y=entry_short_level, line=dict(color='red', dash='dash'), annotation_text='Short Entry', annotation_position='top left')
fig_signal.add_hline(y=exit_level_pos, line=dict(color='blue', dash='dot'), annotation_text='Exit', annotation_position='top right')
fig_signal.add_hline(y=exit_level_neg, line=dict(color='blue', dash='dot'), annotation_text='Exit', annotation_position='bottom right')

fig_signal.update_layout(title="Spread met Entry en Exit niveaus", yaxis_title="Spread", xaxis_title="Datum")

st.plotly_chart(fig_signal, use_container_width=True)

# === Stoploss level (optioneel) ===
stoploss_pct = 0.05  # bv. 5% stoploss van spread (kan aanpasbaar gemaakt worden)

# Bijvoorbeeld stoploss niveaus +/- 5% van spread mean
stoploss_upper = spread_mean * (1 + stoploss_pct)
stoploss_lower = spread_mean * (1 - stoploss_pct)

st.subheader("🛑 Stoploss niveau (voorbeeld)")
st.write(f"Stoploss boven: {stoploss_upper:.4f}")
st.write(f"Stoploss onder: {stoploss_lower:.4f}")

# === Samenvatting en advies ===
st.subheader("📈 Samenvatting en advies")

st.markdown(f"""
- **Coin pair:** {name1} & {name2}  
- **Periode:** {start_date} tot {end_date}  
- **Correlatie:** {corr:.2f}  
- **R²:** {r2:.2f}  
- **Laatste z-score:** {df['zscore'].iloc[-1]:.2f}  

**Interpretatie:**
- Bij een z-score > ±{entry_threshold} wordt een trade signaal gegeven (long/short spread).  
- Bij een z-score binnen ±{exit_threshold} wordt aangeraden positie te sluiten.  
- Wees voorzichtig bij een lage correlatie (<0.7) of lage R² (<0.6), de betrouwbaarheid van de trade neemt af.

**Huidig signaal:** {current_position}  
""")



