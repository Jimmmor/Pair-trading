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
    
    st.markdown("---")
    zscore_entry_threshold = st.slider("Z-score entry threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

# Converteer namen naar ticker symbolen
coin1 = tickers[name1]
coin2 = tickers[name2]

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

# Data verwerken - sluitprijzen gebruiken
# Controleer of data multi-column is (yfinance kan soms multi-level columns geven)
if isinstance(data1.columns, pd.MultiIndex):
    df1 = data1['Close'].iloc[:, 0].dropna()  # Neem eerste kolom als er meerdere zijn
    df2 = data2['Close'].iloc[:, 0].dropna()
else:
    df1 = data1['Close'].dropna()
    df2 = data2['Close'].dropna()

# Debug info
st.write(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")
st.write(f"df1 type: {type(df1)}, df2 type: {type(df2)}")

# Zorg ervoor dat we Series hebben
if not isinstance(df1, pd.Series):
    df1 = pd.Series(df1)
if not isinstance(df2, pd.Series):
    df2 = pd.Series(df2)

# Combineer data op basis van datum - gebruik align om indexen te matchen
df1_aligned, df2_aligned = df1.align(df2, join='inner')

# Maak DataFrame
df = pd.DataFrame({
    'price1': df1_aligned,
    'price2': df2_aligned
}).dropna()

if df.empty:
    st.error("Geen overlappende data beschikbaar voor beide coins.")
    st.stop()

# Lineaire regressie voor spread berekening
X = df['price1'].values.reshape(-1, 1)
y = df['price2'].values

model = LinearRegression()
model.fit(X, y)

alpha = model.intercept_
beta = model.coef_[0]
r_squared = model.score(X, y)

# Spread berekenen
df['spread'] = df['price2'] - (alpha + beta * df['price1'])

# Spread statistieken
spread_mean = df['spread'].mean()
spread_std = df['spread'].std()

# Z-score berekenen
df['zscore'] = (df['spread'] - spread_mean) / spread_std

# Rolling correlatie
df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])

# Pearson correlatie
pearson_corr = df['price1'].corr(df['price2'])

# Trade signalen gebaseerd op z-score thresholds
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

# Start- en einddatum voor samenvatting bepalen
start_date = df.index.min().strftime("%Y-%m-%d") if not df.empty else "N/A"
end_date = df.index.max().strftime("%Y-%m-%d") if not df.empty else "N/A"

# Huidige signaal weergeven
st.subheader("🚦 Huidige trade signaal")
st.write(f"**Z-score laatste waarde:** {df['zscore'].iloc[-1]:.2f}")
st.write(f"**Signaal:** {current_position}")

# Entry, exit, stoploss niveaus visualiseren
entry_long_level = -zscore_entry_threshold * spread_std + spread_mean
entry_short_level = zscore_entry_threshold * spread_std + spread_mean
exit_level_pos = zscore_exit_threshold * spread_std + spread_mean
exit_level_neg = -zscore_exit_threshold * spread_std + spread_mean

fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))

# Entry en exit lijnen tekenen
fig_signal.add_hline(y=entry_long_level, line=dict(color='green', dash='dash'), 
                    annotation_text='Long Entry', annotation_position='bottom left')
fig_signal.add_hline(y=entry_short_level, line=dict(color='red', dash='dash'), 
                    annotation_text='Short Entry', annotation_position='top left')
fig_signal.add_hline(y=exit_level_pos, line=dict(color='blue', dash='dot'), 
                    annotation_text='Exit', annotation_position='top right')
fig_signal.add_hline(y=exit_level_neg, line=dict(color='blue', dash='dot'), 
                    annotation_text='Exit', annotation_position='bottom right')

fig_signal.update_layout(title="Spread met Entry en Exit niveaus", yaxis_title="Spread", xaxis_title="Datum")

st.plotly_chart(fig_signal, use_container_width=True)

# Stoploss niveau (voorbeeld)
stoploss_pct = 0.05  # 5% stoploss van spread, aanpasbaar

stoploss_upper = spread_mean * (1 + stoploss_pct)
stoploss_lower = spread_mean * (1 - stoploss_pct)

st.subheader("🛑 Stoploss niveau (voorbeeld)")
st.write(f"Stoploss boven: {stoploss_upper:.4f}")
st.write(f"Stoploss onder: {stoploss_lower:.4f}")

# Aanvullende visualisaties
col1, col2 = st.columns(2)

with col1:
    # Prijs grafiek met dubbele y-as
    fig_prices = go.Figure()
    
    # Coin 1 op primaire y-as
    fig_prices.add_trace(go.Scatter(
        x=df.index, 
        y=df['price1'], 
        name=name1, 
        line=dict(color='blue')
    ))
    
    # Coin 2 op secundaire y-as
    fig_prices.add_trace(go.Scatter(
        x=df.index, 
        y=df['price2'], 
        name=name2, 
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    # Layout configuratie
    fig_prices.update_layout(
        title="Prijsverloop",
        xaxis_title="Datum",
        yaxis_title=f"{name1} Prijs (USD)",
        yaxis2=dict(
            title=f"{name2} Prijs (USD)",
            overlaying='y',
            side='right'
        )
    )
    
    st.plotly_chart(fig_prices, use_container_width=True)

with col2:
    # Z-score grafiek
    fig_zscore = go.Figure()
    fig_zscore.add_trace(go.Scatter(x=df.index, y=df['zscore'], name='Z-score', line=dict(color='purple')))
    fig_zscore.add_hline(y=zscore_entry_threshold, line=dict(color='red', dash='dash'), 
                        annotation_text='Entry Threshold')
    fig_zscore.add_hline(y=-zscore_entry_threshold, line=dict(color='green', dash='dash'), 
                        annotation_text='Entry Threshold')
    fig_zscore.add_hline(y=zscore_exit_threshold, line=dict(color='blue', dash='dot'), 
                        annotation_text='Exit Threshold')
    fig_zscore.add_hline(y=-zscore_exit_threshold, line=dict(color='blue', dash='dot'), 
                        annotation_text='Exit Threshold')
    fig_zscore.update_layout(title="Z-score", yaxis_title="Z-score", xaxis_title="Datum")
    st.plotly_chart(fig_zscore, use_container_width=True)

# Rolling correlatie grafiek met groen transparant oppervlak
fig_corr = go.Figure()

# Voeg het groene transparante oppervlak toe
fig_corr.add_trace(go.Scatter(
    x=df.index, 
    y=df['rolling_corr'], 
    fill='tozeroy',
    fillcolor='rgba(0, 255, 0, 0.3)',  # Groen transparant
    line=dict(color='orange', width=2),
    name='Rolling Correlatie'
))

fig_corr.update_layout(
    title="Rolling Correlatie met Transparant Oppervlak", 
    yaxis_title="Correlatie", 
    xaxis_title="Datum",
    yaxis=dict(range=[-1, 1])  # Correlatie is altijd tussen -1 en 1
)
st.plotly_chart(fig_corr, use_container_width=True)

# === NIEUWE SECTIE: Correlatie Analyse ===
st.header("📊 Correlatie Analyse")

# Bereken returns voor scatter plot
df['returns1'] = df['price1'].pct_change()
df['returns2'] = df['price2'].pct_change()

# Verwijder NaN waarden
returns_clean = df[['returns1', 'returns2']].dropna()

col1, col2 = st.columns(2)

with col1:
    # Scatter plot van returns (zoals in je afbeelding)
    fig_scatter = go.Figure()
    
    # Scatter plot
    fig_scatter.add_trace(go.Scatter(
        x=returns_clean['returns1'] * 100,  # Converteer naar percentages
        y=returns_clean['returns2'] * 100,
        mode='markers',
        marker=dict(
            color='purple',
            size=8,
            opacity=0.6
        ),
        name='Daily Returns',
        showlegend=False
    ))
    
    # Voeg regressielijn toe
    from sklearn.linear_model import LinearRegression
    X_scatter = returns_clean['returns1'].values.reshape(-1, 1)
    y_scatter = returns_clean['returns2'].values
    
    model_scatter = LinearRegression()
    model_scatter.fit(X_scatter, y_scatter)
    
    # Regressielijn data
    x_line = np.linspace(returns_clean['returns1'].min(), returns_clean['returns1'].max(), 100)
    y_line = model_scatter.predict(x_line.reshape(-1, 1))
    
    fig_scatter.add_trace(go.Scatter(
        x=x_line * 100,
        y=y_line * 100,
        mode='lines',
        line=dict(color='yellow', width=3),
        name=f'y = {model_scatter.coef_[0]:.3f}x + {model_scatter.intercept_:.3f}',
        showlegend=False
    ))
    
    # Voeg regressie vergelijking toe als annotatie
    fig_scatter.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'y = {model_scatter.coef_[0]:.3f}x + {model_scatter.intercept_:.6f}',
        showarrow=False,
        font=dict(size=12, color='yellow'),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='yellow',
        borderwidth=1
    )
    
    fig_scatter.update_layout(
        title="Returns Correlatie Scatter Plot",
        xaxis_title=f"{name1} Returns (%)",
        yaxis_title=f"{name2} Returns (%)",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Boxplot van correlaties
    fig_box = go.Figure()
    
    # Bereken rolling correlatie voor verschillende windows
    correlations_data = []
    windows = [10, 20, 30, 60]
    
    for window in windows:
        if len(df) >= window:
            rolling_corr = df['price1'].rolling(window=window).corr(df['price2']).dropna()
            correlations_data.extend([(window, corr) for corr in rolling_corr])
    
    # Converteer naar DataFrame voor boxplot
    corr_df = pd.DataFrame(correlations_data, columns=['Window', 'Correlation'])
    
    for window in windows:
        window_data = corr_df[corr_df['Window'] == window]['Correlation']
        if len(window_data) > 0:
            fig_box.add_trace(go.Box(
                y=window_data,
                name=f'{window}d window',
                boxpoints='outliers'
            ))
    
    fig_box.update_layout(
        title="Correlatie Distributie (Boxplot)",
        yaxis_title="Correlatie",
        xaxis_title="Rolling Window"
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

# Correlatie statistieken tabel
st.subheader("📈 Correlatie Statistieken")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pearson Correlatie", f"{pearson_corr:.4f}")
    st.metric("Beta (β)", f"{beta:.4f}")
    st.metric("R-squared", f"{r_squared:.4f}")

with col2:
    current_rolling_corr = df['rolling_corr'].iloc[-1]
    avg_rolling_corr = df['rolling_corr'].mean()
    st.metric("Huidige Rolling Correlatie", f"{current_rolling_corr:.4f}")
    st.metric("Gemiddelde Rolling Correlatie", f"{avg_rolling_corr:.4f}")
    st.metric("Alpha (α)", f"{alpha:.6f}")

with col3:
    returns_corr = returns_clean['returns1'].corr(returns_clean['returns2'])
    volatility_ratio = returns_clean['returns2'].std() / returns_clean['returns1'].std()
    st.metric("Returns Correlatie", f"{returns_corr:.4f}")
    st.metric("Volatiliteit Ratio", f"{volatility_ratio:.4f}")
    st.metric("Std Error (β)", f"{np.sqrt(np.mean((df['price2'] - (alpha + beta * df['price1']))**2)):.4f}")

# Correlatie beoordeling
st.subheader("🎯 Correlatie Beoordeling")

if abs(pearson_corr) > 0.8:
    corr_assessment = "🟢 Uitstekend - Sterke correlatie, geschikt voor pairs trading"
elif abs(pearson_corr) > 0.6:
    corr_assessment = "🟡 Goed - Redelijke correlatie, geschikt met voorzichtigheid"
elif abs(pearson_corr) > 0.4:
    corr_assessment = "🟠 Matig - Zwakke correlatie, verhoogd risico"
else:
    corr_assessment = "🔴 Slecht - Zeer zwakke correlatie, niet geschikt voor pairs trading"

st.write(f"**Correlatie beoordeling:** {corr_assessment}")

# R-squared beoordeling
if r_squared > 0.7:
    r2_assessment = "🟢 Uitstekend - Sterke lineaire relatie"
elif r_squared > 0.5:
    r2_assessment = "🟡 Goed - Redelijke lineaire relatie"
elif r_squared > 0.3:
    r2_assessment = "🟠 Matig - Zwakke lineaire relatie"
else:
    r2_assessment = "🔴 Slecht - Zeer zwakke lineaire relatie"

st.write(f"**R-squared beoordeling:** {r2_assessment}")

# Stabiliteit beoordeling
rolling_corr_std = df['rolling_corr'].std()
if rolling_corr_std < 0.1:
    stability_assessment = "🟢 Stabiel - Correlatie is consistent over tijd"
elif rolling_corr_std < 0.2:
    stability_assessment = "🟡 Redelijk stabiel - Enige fluctuatie in correlatie"
else:
    stability_assessment = "🔴 Instabiel - Correlatie varieert sterk over tijd"

st.write(f"**Stabiliteit beoordeling:** {stability_assessment}")

st.markdown("---")

# Samenvatting van de pairs trading analyse
st.header("📊 Samenvatting van de pairs trading analyse")

st.markdown(f"""
- **Asset 1:** {name1} ({coin1})
- **Asset 2:** {name2} ({coin2})
- **Periode:** {start_date} tot {end_date}
- **Data punten:** {len(df)}

**Regressie resultaten:**
- Alpha: {alpha:.6f}
- Beta: {beta:.6f}
- R²: {r_squared:.4f}
- Pearson correlatie: {pearson_corr:.4f}

**Spread statistieken:**
- Gemiddelde spread: {spread_mean:.4f}
- Standaarddeviatie spread: {spread_std:.4f}

**Laatste z-score:** {df['zscore'].iloc[-1]:.2f}

**Huidige signaal:** {current_position}
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
