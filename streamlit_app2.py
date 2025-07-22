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
    all_tickers = list(tickers.keys())
    
    name1 = st.selectbox("Asset 1", all_tickers, index=0)
    remaining_tickers = [t for t in all_tickers if t != name1]
    name2 = st.selectbox("Asset 2", remaining_tickers, index=0 if len(remaining_tickers) > 0 else 0)
    
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
        df = df.copy()
        
        X = df['price1'].values.reshape(-1, 1)
        y = df['price2'].values
        model = LinearRegression().fit(X, y)
        df['spread'] = df['price2'] - (model.intercept_ + model.coef_[0] * df['price1'])
        
        spread_mean = df['spread'].mean()
        spread_std = df['spread'].std()
        df['zscore'] = (df['spread'] - spread_mean) / spread_std
        
        df['signal'] = np.nan
        df.loc[df['zscore'] > zscore_entry_threshold, 'signal'] = 'SHORT'
        df.loc[df['zscore'] < -zscore_entry_threshold, 'signal'] = 'LONG'
        
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
        
        exit_price1 = entry_price1 * (1 - (row['zscore'] - zscore_exit_threshold)/10)
        exit_price2 = entry_price2 * (1 + (row['zscore'] - zscore_exit_threshold)/10)
        
        stoploss1 = entry_price1 * (1 + (row['zscore']/5))
        stoploss2 = entry_price2 * (1 - (row['zscore']/5))
        
        position_size = self._calculate_position_size(entry_price1, stoploss1)
        
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
        risk_amount = self.risk_per_trade / 100
        price_diff = abs(entry_price - stoploss_price)
        return risk_amount / price_diff if price_diff != 0 else 0

@st.cache_data
def load_data(ticker_key, period, interval):
    try:
        ticker_symbol = tickers[ticker_key]
        data = yf.download(
            tickers=ticker_symbol,
            period=period,
            interval=interval,
            progress=False
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].dropna()
        return data['Close'].dropna()
        
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker_key}: {e}")
        return pd.Series()

def preprocess_data(data1, data2):
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

data1 = load_data(name1, periode, interval)
data2 = load_data(name2, periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins")
    st.stop()

df = preprocess_data(data1, data2)

calculator = PairTradingCalculator(leverage=leverage, risk_per_trade=risk_per_trade)
df = calculator.calculate_trade_params(df)

def plot_trading_signals(df):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[
            [{"secondary_y": True}],
            [{}],
            [{}]
        ]
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['price1'],
            name=f"{name1} Price",
            line=dict(color='blue', width=2),
            hovertemplate="%{y:.6f}"
        ),
        row=1, col=1,
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['price2'],
            name=f"{name2} Price",
            line=dict(color='red', width=2),
            hovertemplate="%{y:.6f}"
        ),
        row=1, col=1,
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['price1'] - df['price2'],
            name="Spread",
            line=dict(color='green', width=1),
            hovertemplate="%{y:.6f}"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['zscore'],
            name="Z-score",
            line=dict(color='purple', width=1.5),
            hovertemplate="Z: %{y:.2f}"
        ),
        row=3, col=1
    )

    fig.add_hline(
        y=zscore_entry_threshold,
        line=dict(color="red", width=1, dash="dash"),
        row=3, col=1
    )
    fig.add_hline(
        y=-zscore_entry_threshold,
        line=dict(color="green", width=1, dash="dash"),
        row=3, col=1
    )
    fig.add_hline(
        y=zscore_exit_threshold,
        line=dict(color="pink", width=1, dash="dot"),
        row=3, col=1
    )
    fig.add_hline(
        y=-zscore_exit_threshold,
        line=dict(color="lightgreen", width=1, dash="dot"),
        row=3, col=1
    )

    fig.update_layout(
        title=f"Pairs Trading: {name1} vs {name2}",
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )
    
    fig.update_yaxes(title_text=f"{name1} Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{name2} Price", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    
    return fig

st.plotly_chart(plot_trading_signals(df), use_container_width=True)

# Aangepast gedeelte - hier was de fout
with st.expander("📋 Trading Signals Table"):
    signals_df = df[df['signal'].notna()][[
        'signal', 'price1', 'price2', 'zscore',
        'entry_price1', 'entry_price2',
        'exit_price1', 'exit_price2',
        'stoploss1', 'stoploss2',
        'liquidation1', 'liquidation2',
        'position_size'
    ]]
    
    # Alleen numerieke kolommen formatteren
    numeric_cols = signals_df.select_dtypes(include=[np.number]).columns
    if not signals_df.empty:
        st.dataframe(
            signals_df.style.format(
                {col: "{:.6f}" for col in numeric_cols}
            )
        )
    else:
        st.write("No trading signals found")

with st.expander("🔙 Backtesting"):
    st.header("Backtesting Results")
    
    if st.button("Run Backtest"):
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
