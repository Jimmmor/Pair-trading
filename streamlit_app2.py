# === DEEL 1: Invoer, data en berekeningen ===
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Monitor")
st.markdown("Vergelijk twee coins, bereken spread, ratio, Z-score en statistieken voor pairs trading.")

# Sidebar: keuzes
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    tickers = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD", "Cardano (ADA)": "ADA-USD"}
    name1 = st.selectbox("Coin 1", list(tickers.keys()), 0)
    name2 = st.selectbox("Coin 2", [k for k in tickers if k != name1], 1)
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y"], 2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y"] else ["1d", "1h", "30m"], 0)
    corr_window = st.slider("Rolling correlatie window", 5, 60, 20)
    entry_thresh = st.slider("Z-score entry", 1.0, 5.0, 2.0, 0.1)
    exit_thresh = st.slider("Z-score exit", 0.0, 2.0, 0.5, 0.1)

# Data ophalen
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, period=periode, interval=interval)["Close"].dropna()

coin1, coin2 = tickers[name1], tickers[name2]
df1, df2 = load_data(coin1), load_data(coin2)
df1, df2 = df1.align(df2, join='inner')

# DataFrame met prijzen en returns
df = pd.DataFrame({"price1": df1, "price2": df2}).dropna()
X = df[["price1"]]
y = df["price2"]
model = LinearRegression().fit(X, y)

alpha, beta = model.intercept_, model.coef_[0]
df["spread"] = df["price2"] - (alpha + beta * df["price1"])
df["zscore"] = (df["spread"] - df["spread"].mean()) / df["spread"].std()
df["rolling_corr"] = df["price1"].rolling(corr_window).corr(df["price2"])

# Signalen
df["long_entry"] = df["zscore"] < -entry_thresh
df["short_entry"] = df["zscore"] > entry_thresh
df["exit"] = df["zscore"].abs() < exit_thresh

# Laatste signaal
last = df.iloc[-1]
if last["long_entry"]:
    signaal = f"Long (koop {name2}, verkoop {name1})"
elif last["short_entry"]:
    signaal = f"Short (verkoop {name2}, koop {name1})"
elif last["exit"]:
    signaal = "Exit (geen positie)"
else:
    signaal = "Geen duidelijk signaal"

st.subheader("🚦 Huidig Signaal")
st.write(f"Z-score: {last['zscore']:.2f} → **{signaal}**")
let markers = [];
const getIcon = (type) =>
  L.icon({
    iconUrl: icons[type] || icons.default,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  });

function fetchAndRenderMarkers({ gemeente = '', search = '' } = {}) {
  let url = 'get_markers.php';
  if (gemeente || search) {
    const params = new URLSearchParams({ gemeente, search });
    url += '?' + params.toString();
  }

  fetch(url)
    .then((res) => res.json())
    .then((data) => {
      markers.forEach((m) => map.removeLayer(m));
      markers = data.map((loc) => {
        const marker = L.marker([loc.lat, loc.lng], {
          icon: getIcon(loc.type),
        }).addTo(map);

        marker.bindPopup(`
          <b>${loc.naam}</b><br>
          ${loc.straat} ${loc.huisnummer}<br>
          ${loc.postcode} ${loc.plaats}<br>
          <a href="${loc.website}" target="_blank">Website</a>
        `);

        return marker;
      });
    })
    .catch((err) => console.error('Marker fetch failed', err));
}
const gemeenteFilter = document.getElementById('gemeente-filter');
const zoekInput = document.getElementById('zoek-input');

gemeenteFilter.addEventListener('change', () => {
  fetchAndRenderMarkers({
    gemeente: gemeenteFilter.value,
    search: zoekInput.value.trim(),
  });
});

zoekInput.addEventListener('input', () => {
  fetchAndRenderMarkers({
    gemeente: gemeenteFilter.value,
    search: zoekInput.value.trim(),
  });
});

document.getElementById('reset-button').addEventListener('click', () => {
  gemeenteFilter.value = '';
  zoekInput.value = '';
  fetchAndRenderMarkers(); // alles tonen
});

// Laad initiale markers bij start
fetchAndRenderMarkers();
