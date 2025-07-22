import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from constants.tickers import tickers

# Pagina-instellingen
st.set_page_config(layout="wide")
st.title("📈 Advanced Pairs Trading Monitor")

# Sidebar instellingen
with st.sidebar:
    st.header("🔍 Pair Selection")
    # Gebruik de keys uit het tickers dictionary
    ticker_keys = list(tickers.keys())
    name1 = st.selectbox("Asset 1", ticker_keys, index=0)
    # Filter de tweede selectie om dubbele selectie te voorkomen
    remaining = [k for k in ticker_keys if k != name1]
    name2 = st.selectbox("Asset 2", remaining, index=0 if len(remaining) > 1 else 0)
    
    st.markdown("---")
    st.header("📊 Data Settings")
    periode = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y", "2y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlation window (days)", 5, 60, 20)
    
    st.markdown("---")
    st.header("⚙️ Trading Parameters")
    zscore_entry_threshold = st.slider("Entry Z-score", 1.0, 5.0, 2.0, 0.1)
    zscore_exit_threshold = st.slider("Exit Z-score", 0.0, 2.0, 0.5, 0.1)
    leverage = st.slider("Leverage", 1, 10, 3)
    risk_per_trade = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)

class PairTradingCalculator:
    def __init__(self, leverage=1, risk_per_trade=0.01):
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
    
    def calculate_trade_params(self, df):
        """Bereken alle trading parameters voor elk punt in de dataframe"""
        df = df.copy()
        
        # Spread berekening
        X = df['price1'].values.reshape(-1, 1)
        y = df['price2'].values
        model = LinearRegression().fit(X, y)
        df['spread'] = df['price2'] - (model.intercept_ + model.coef_[0] * df['price1'])
        
        # Z-score berekening
        spread_mean = df['spread'].mean()
        spread_std = df['spread'].std()
        df['zscore'] = (df['spread'] - spread_mean) / spread_std
        
        # Signal detection
        df['signal'] = np.where(
            df['zscore'] > zscore_entry_threshold, 'SHORT',
            np.where(
                df['zscore'] < -zscore_entry_threshold, 'LONG', 
                np.nan
            )
        )
        
        # Bereken trade parameters voor alle signalen
        for idx, row in df[df['signal'].notna()].iterrows():
            if row['signal'] == 'SHORT':
                trade_params = self._calculate_short_trade(row)
            else:
                trade_params = self._calculate_long_trade(row)
            
            for key, val in trade_params.items():
                df.loc[idx, key] = val
        
        return df

    def _calculate_short_trade(self, row):
        entry_price1 = row['price1']
        entry_price2 = row['price2']
        
        # Exit prijzen gebaseerd op z-score
        exit_price1 = entry_price1 * (1 - (row['zscore'] - zscore_exit_threshold)/10)
        exit_price2 = entry_price2 * (1 + (row['zscore'] - zscore_exit_threshold)/10)
        
        # Stoploss (2x de entry threshold)
        stoploss1 = entry_price1 * (1 + (row['zscore']/5))
        stoploss2 = entry_price2 * (1 - (row['zscore']/5))
        
        # Position sizing
        position_size = self._calculate_position_size(entry_price1, stoploss1)
        
        # Liquidatie prijzen
        liquidation1 = entry_price1 * (1 + 1/self.leverage) - stoploss1 * (1/self.leverage)
        liquidation2 = entry_price2 * (1 - 1/self.leverage) + stoploss2 * (1/self.leverage)
        
        return {
            'entry_price1': entry_price1,
            'entry_price2': entry_price2,
            'exit_price1': exit_price1,
            'exit_price2': exit_price2,
            'stoploss1': stoploss1,
            'stoploss2': stoploss2,
            'liquidation1': liquidation1,
            'liquidation2': liquidation2,
            'position_size': position_size
        }

    def _calculate_long_trade(self, row):
        entry_price1 = row['price1']
        entry_price2 = row['price2']
        
        exit_price1 = entry_price1 * (1 + (abs(row['zscore']) - zscore_exit_threshold)/10)
        exit_price2 = entry_price2 * (1 - (abs(row['zscore']) - zscore_exit_threshold)/10)
        
        stoploss1 = entry_price1 * (1 - (abs(row['zscore'])/5))
        stoploss2 = entry_price2 * (1 + (abs(row['zscore'])/5))
        
        position_size = self._calculate_position_size(entry_price1, stoploss1)
        
        liquidation1 = entry_price1 * (1 - 1/self.leverage) + stoploss1 * (1/self.leverage)
        liquidation2 = entry_price2 * (1 + 1/self.leverage) - stoploss2 * (1/self.leverage)
        
        return {
            'entry_price1': entry_price1,
            'entry_price2': entry_price2,
            'exit_price1': exit_price1,
            'exit_price2': exit_price2,
            'stoploss1': stoploss1,
            'stoploss2': stoploss2,
            'liquidation1': liquidation1,
            'liquidation2': liquidation2,
            'position_size': position_size
        }

    def _calculate_position_size(self, entry_price, stoploss_price):
        """Bereken positie grootte gebaseerd op risico per trade"""
        risk_amount = self.risk_per_trade / 100  # Percentage omzetten naar decimaal
        price_diff = abs(entry_price - stoploss_price)
        return risk_amount / price_diff if price_diff != 0 else 0

@st.cache_data
def load_data(ticker_key, period, interval):
    """Laad data voor een ticker key uit het tickers dictionary"""
    try:
        # Haal de echte ticker symbol op uit het tickers dict
        ticker_symbol = tickers[ticker_key]
        
        # Download de data
        data = yf.download(
            tickers=ticker_symbol,
            period=period,
            interval=interval,
            progress=False
        )
        
        # Behoud de originele return structuur
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].dropna()
        return data['Close'].dropna()
        
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker_key}: {e}")
        return pd.Series()

# Originele data verwerkingsfunctie
def preprocess_data(data1, data2):
    """Preprocessing zoals in originele code"""
    if not isinstance(data1, pd.Series):
        data1 = pd.Series(data1)
    if not isinstance(data2, pd.Series):
        data2 = pd.Series(data2)
    
    data1_aligned, data2_aligned = data1.align(data2, join='inner')
    
    df = pd.DataFrame({
        'price1': data1_aligned,
        'price2': data2_aligned
    }).dropna()
    
    return df

# Gebruik in de app - blijft identiek aan origineel
data1 = load_data(name1, periode, interval)
data2 = load_data(name2, periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins")
    st.stop()

df = preprocess_data(data1, data2)

# Bereken trading parameters
calculator = PairTradingCalculator(leverage=leverage, risk_per_trade=risk_per_trade)
df = calculator.calculate_trade_params(df)

# Visualisatie
def plot_trading_signals(df):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )
    
    # Prijsplot
    fig.add_trace(go.Scatter(
        x=df.index, y=df['price1'],
        name=f"{name1} Price",
        line=dict(color='#636EFA', width=2),
        hovertemplate="%{y:.6f}"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['price2'],
        name=f"{name2} Price",
        line=dict(color='#EF553B', width=2),
        hovertemplate="%{y:.6f}",
        secondary_y=True
    ), row=1, col=1)

    # Spread plot
    fig.add_trace(go.Scatter(
        x=df.index, y=df['price1'] - df['price2'],
        name="Spread",
        line=dict(color='#00CC96', width=1),
        hovertemplate="%{y:.6f}"
    ), row=2, col=1)

    # Z-score plot
    fig.add_trace(go.Scatter(
        x=df.index, y=df['zscore'],
        name="Z-score",
        line=dict(color='#AB63FA', width=1.5),
        hovertemplate="Z: %{y:.2f}"
    ), row=3, col=1)

    # Threshold lines
    for threshold, color, name in [
        (zscore_entry_threshold, 'red', 'Entry Threshold'),
        (-zscore_entry_threshold, 'green', 'Entry Threshold'),
        (zscore_exit_threshold, 'pink', 'Exit Threshold'),
        (-zscore_exit_threshold, 'lightgreen', 'Exit Threshold')
    ]:
        fig.add_hline(
            y=threshold,
            line=dict(color=color, width=1, dash="dot"),
            row=3, col=1
        )

    # Trade signals
    signals = df[df['signal'].notna()]
    for idx, row in signals.iterrows():
        color = 'red' if row['signal'] == 'SHORT' else 'green'
        direction = row['signal']
        
        # Prijsplot markers
        fig.add_vline(
            x=idx,
            line=dict(color=color, width=1, dash='dash'),
            row=1, col=1
        )
        
        # Gedetailleerde annotatie
        fig.add_annotation(
            x=idx,
            y=row['price1'],
            text=f"<b>{direction}</b><br>"
                 f"Entry: {row['entry_price1']:.6f}<br>"
                 f"Exit: {row['exit_price1']:.6f}<br>"
                 f"Stop: {row['stoploss1']:.6f}<br>"
                 f"Liq: {row['liquidation1']:.6f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40 if direction == 'SHORT' else 40,
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            row=1, col=1
        )

        # Z-score markers
        fig.add_vline(
            x=idx,
            line=dict(color=color, width=1),
            row=3, col=1
        )
        
        fig.add_annotation(
            x=idx,
            y=row['zscore'],
            text=direction,
            showarrow=False,
            bgcolor=color,
            font=dict(color='white'),
            row=3, col=1
        )

    # Layout
    fig.update_layout(
        height=900,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(240,240,240,0.8)'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    
    return fig

# Toon de plot
st.plotly_chart(plot_trading_signals(df), use_container_width=True)

# Toon trading signals tabel
with st.expander("📋 Trading Signals Table"):
    signals_df = df[df['signal'].notna()][[
        'signal', 'price1', 'price2', 'zscore',
        'entry_price1', 'entry_price2',
        'exit_price1', 'exit_price2',
        'stoploss1', 'stoploss2',
        'liquidation1', 'liquidation2',
        'position_size'
    ]]
    st.dataframe(signals_df.style.format("{:.6f}"))

# Backtesting sectie
with st.expander("🔙 Backtesting"):
    st.header("Backtesting Results")
    
    if st.button("Run Backtest"):
        # Vereenvoudigde backtest - implementeer je eigen logica
        initial_capital = 10000
        portfolio_value = initial_capital
        trades = []
        
        for idx, row in df[df['signal'].notna()].iterrows():
            if row['signal'] == 'LONG':
                pnl = (row['exit_price1'] - row['entry_price1']) + (row['exit_price2'] - row['entry_price2'])
            else:
                pnl = (row['entry_price1'] - row['exit_price1']) + (row['entry_price2'] - row['exit_price2'])
            
            trades.append({
                'Date': idx,
                'Signal': row['signal'],
                'Entry1': row['entry_price1'],
                'Entry2': row['entry_price2'],
                'Exit1': row['exit_price1'],
                'Exit2': row['exit_price2'],
                'P&L': pnl * row['position_size']
            })
        
        if trades:
            trades_df = pd.DataFrame(trades)
            portfolio_value += trades_df['P&L'].sum()
            st.success(f"Backtest complete! Final portfolio value: ${portfolio_value:,.2f}")
            st.dataframe(trades_df)
        else:
            st.warning("No trades executed during backtest period")
