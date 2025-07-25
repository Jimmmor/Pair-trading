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
st.title("Advanced Pairs Trading Monitor")

# Sidebar instellingen
with st.sidebar:
    st.header("Pair Selection")
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
        st.error(f"Error retrieving data for {ticker_key}: {e}")
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

# Load data
data1 = load_data(name1, periode, interval)
data2 = load_data(name2, periode, interval)

if data1.empty or data2.empty:
    st.error("No data available for one or both assets")
    st.stop()

df = preprocess_data(data1, data2)

calculator = PairTradingCalculator(leverage=leverage, risk_per_trade=risk_per_trade)
df = calculator.calculate_trade_params(df)

# === STATISTICAL ANALYSIS SECTION ===
with st.expander("📊 Statistical Analysis", expanded=True):
    st.header("📊 Statistical Analysis")
    
    # Calculate statistics
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
    
    # Rolling correlation
    df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
    pearson_corr = df['price1'].corr(df['price2'])
    
    # Ratio chart
    df['ratio'] = df['price1'] / df['price2']
    
    # Returns for scatterplot
    df['returns1'] = df['price1'].pct_change()
    df['returns2'] = df['price2'].pct_change()
    
    # Layout with tabs
    tab1, tab2 = st.tabs(["Price Analysis", "Correlation Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart with 2 y-axes
            fig_prices = go.Figure()
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price1'], name=name1, line=dict(color='blue')))
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price2'], name=name2, line=dict(color='red'), yaxis='y2'))
            
            fig_prices.update_layout(
                title="Price Movement",
                xaxis_title="Date",
                yaxis_title=f"{name1} Price (USD)",
                yaxis2=dict(
                    title=f"{name2} Price (USD)",
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
            
            # Spread chart
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=df.index, y=df['spread'], name='Spread',
                line=dict(color='green'), fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            fig_spread.update_layout(
                title="Spread Between Assets",
                xaxis_title="Date",
                yaxis_title="Spread",
                height=400
            )
            st.plotly_chart(fig_spread, use_container_width=True)
        
        with col2:
            # Ratio chart
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=df.index, y=df['ratio'], name=f"{name1}/{name2} Ratio",
                line=dict(color='purple'), fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            fig_ratio.update_layout(
                title=f"{name1}/{name2} Price Ratio",
                xaxis_title="Date",
                yaxis_title="Ratio",
                height=400
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Z-score chart with trading signals
            fig_zscore = go.Figure()
            
            # Z-score line
            fig_zscore.add_trace(go.Scatter(
                x=df.index,
                y=df['zscore'],
                name='Z-score',
                line=dict(color='#2ca02c', width=2)
            ))
            
            # Long entry (below -entry threshold)
            fig_zscore.add_hline(
                y=-zscore_entry_threshold,
                line=dict(color='green', dash='dash', width=1),
                annotation_text="LONG ENTRY (buy spread)",
                annotation_position="bottom right"
            )
            
            # Long exit (above -exit threshold)
            fig_zscore.add_hline(
                y=-zscore_exit_threshold,
                line=dict(color='blue', dash='dot', width=1),
                annotation_text="LONG EXIT",
                annotation_position="bottom right"
            )
            
            # Short entry (above entry threshold)
            fig_zscore.add_hline(
                y=zscore_entry_threshold,
                line=dict(color='red', dash='dash', width=1),
                annotation_text="SHORT ENTRY (sell spread)",
                annotation_position="top right"
            )
            
            # Short exit (below exit threshold)
            fig_zscore.add_hline(
                y=zscore_exit_threshold,
                line=dict(color='purple', dash='dot', width=1),
                annotation_text="SHORT EXIT",
                annotation_position="top right"
            )
            
            # Zero line
            fig_zscore.add_hline(
                y=0,
                line=dict(color='black', width=1)
            )
            
            fig_zscore.update_layout(
                title="Z-score with Trading Signals",
                xaxis_title="Date",
                yaxis_title="Z-score value",
                height=400,
                showlegend=True,
                annotations=[
                    dict(
                        x=0.5,
                        y=-zscore_entry_threshold-0.5,
                        xref="paper",
                        yref="y",
                        text="LONG zone",
                        showarrow=False,
                        font=dict(color="green")
                    ),
                    dict(
                        x=0.5,
                        y=zscore_entry_threshold+0.5,
                        xref="paper",
                        yref="y",
                        text="SHORT zone",
                        showarrow=False,
                        font=dict(color="red")
                    )
                ]
            )
            
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation chart
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df.index, y=df['rolling_corr'], name='Rolling Correlation',
                line=dict(color='blue')
            ))
            fig_corr.update_layout(
                title=f"Rolling Correlation ({corr_window}d)",
                xaxis_title="Date",
                yaxis_title="Correlation",
                yaxis_range=[-1, 1],
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Statistics
            st.subheader("Statistics")
            st.metric("Pearson Correlation", f"{pearson_corr:.4f}")
            st.metric("Rolling Correlation", f"{df['rolling_corr'].iloc[-1]:.4f}")
            st.metric("Beta (β)", f"{beta:.4f}")
            st.metric("Alpha (α)", f"{alpha:.6f}")
            st.metric("R-squared", f"{r_squared:.4f}")
        
        with col2:
            # Scatterplot
            fig_scatter = px.scatter(
                df.dropna(), x='returns1', y='returns2',
                title=f"Returns {name1} vs {name2}",
                labels={'returns1': f'{name1} Returns', 'returns2': f'{name2} Returns'}
            )
            fig_scatter.update_traces(marker=dict(size=8, color='blue', opacity=0.6))
            st.plotly_chart(fig_scatter, use_container_width=True)

# === TRADING SIGNALS SECTION ===
with st.expander("Trading Signals - Praktische Uitvoering", expanded=True):
    st.header("Praktische Trade Uitvoering")
    
    # Huidige marktdata
    current_price1 = df['price1'].iloc[-1]
    current_price2 = df['price2'].iloc[-1]
    current_zscore = df['zscore'].iloc[-1]
    hedge_ratio = beta
    
    # Bereken fair value spread
    spread_mean = df['spread'].rolling(30).mean().iloc[-1]
    fair_value2 = df['price1'].iloc[-1] * hedge_ratio + spread_mean
    
    # Toon huidige marktsituatie
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{name1} Prijs", f"${current_price1:.4f}")
    with col2:
        st.metric(f"{name2} Prijs", f"${current_price2:.4f}")
    with col3:
        st.metric("Current Z-score", f"{current_zscore:.2f}")
    
    # Trade execution parameters
    if current_zscore < -zscore_entry_threshold:
        st.success(f"**LONG SPREAD SIGNAL (Z = {current_zscore:.2f})**")
        st.markdown(f"""
        ### 📈 Uitvoering:
        1. **Koop {name1}** tegen huidige prijs: ${current_price1:.4f}
        2. **Verkoop {hedge_ratio:.4f} {name2}** per {name1} tegen: ${current_price2:.4f}
        3. **Hedge Ratio**: 1 {name1} = {hedge_ratio:.4f} {name2}
        
        ### 🔎 Verwacht Herstel:
        - Richting fair value: ${fair_value2:.4f} (+{(fair_value2-current_price2)/current_price2*100:.2f}%)
        - Target spread: ${spread_mean:.4f}
        """)
        
    elif current_zscore > zscore_entry_threshold:
        st.error(f"**SHORT SPREAD SIGNAL (Z = {current_zscore:.2f})**")
        st.markdown(f"""
        ### 📉 Uitvoering:
        1. **Verkoop {name1}** tegen huidige prijs: ${current_price1:.4f}
        2. **Koop {hedge_ratio:.4f} {name2}** per {name1} tegen: ${current_price2:.4f}
        3. **Hedge Ratio**: 1 {name1} = {hedge_ratio:.4f} {name2}
        
        ### 🔎 Verwacht Herstel:
        - Richting fair value: ${fair_value2:.4f} ({(fair_value2-current_price2)/current_price2*100:.2f}%)
        - Target spread: ${spread_mean:.4f}
        """)
    else:
        st.info(f"**GEEN SIGNAL (Z = {current_zscore:.2f})**")
        st.markdown(f"""
        ### ⏳ Wacht op:
        - LONG entry onder Z = -{zscore_entry_threshold:.1f}
        - SHORT entry boven Z = {zscore_entry_threshold:.1f}
        """)
    
    # Toon praktische trading levels
    st.markdown("---")
    st.subheader("Trading Parameters")
    
    entry_levels = pd.DataFrame({
        'Parameter': [
            'Hedge Ratio',
            'Spread Mean',
            'Spread STD',
            'Current Spread',
            'Z-score Entry',
            'Z-score Exit'
        ],
        'Waarde': [
            f"{hedge_ratio:.4f} {name2}/{name1}",
            f"${spread_mean:.4f}",
            f"${df['spread'].rolling(30).std().iloc[-1]:.4f}",
            f"${current_price2 - current_price1*hedge_ratio:.4f}",
            f"±{zscore_entry_threshold:.1f}",
            f"±{zscore_exit_threshold:.1f}"
        ]
    })
    
    st.table(entry_levels)
    
    # FIXED: Z-score visualisatie in plaats van spread
    st.subheader("Z-score met Trading Levels")
    fig = go.Figure()
    
    # Z-score lijn
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['zscore'],
        name='Z-score',
        line=dict(color='blue', width=2)
    ))
    
    # Mean lijn (bij z-score = 0)
    fig.add_hline(
        y=0,
        line=dict(color='black', width=1),
        annotation_text="Mean (Z=0)"
    )
    
    # Entry levels
    fig.add_hline(
        y=zscore_entry_threshold,
        line=dict(color='red', dash='dash', width=2),
        annotation_text=f"Short Entry (Z={zscore_entry_threshold})"
    )
    
    fig.add_hline(
        y=-zscore_entry_threshold,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f"Long Entry (Z={-zscore_entry_threshold})"
    )
    
    # Exit levels
    fig.add_hline(
        y=zscore_exit_threshold,
        line=dict(color='pink', dash='dot', width=1),
        annotation_text=f"Short Exit (Z={zscore_exit_threshold})"
    )
    
    fig.add_hline(
        y=-zscore_exit_threshold,
        line=dict(color='lightgreen', dash='dot', width=1),
        annotation_text=f"Long Exit (Z={-zscore_exit_threshold})"
    )
    
    # Kleur de trading zones
    fig.add_hrect(
        y0=zscore_entry_threshold, y1=5,
        fillcolor="rgba(255,0,0,0.1)",
        layer="below", line_width=0,
        annotation_text="SHORT Zone", annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=-5, y1=-zscore_entry_threshold,
        fillcolor="rgba(0,255,0,0.1)",
        layer="below", line_width=0,
        annotation_text="LONG Zone", annotation_position="bottom left"
    )
    
    fig.update_layout(
        title=f"Z-score Trading Levels ({name1} vs {name2})",
        xaxis_title="Date",
        yaxis_title="Z-score",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# === ENHANCED BACKTESTING SECTION ===
def run_advanced_backtest(df, entry_threshold, exit_threshold, initial_capital, 
                         transaction_cost, max_position_size, stop_loss_pct, 
                         take_profit_pct, lookback_period=30):
    """
    Verbeterde backtest met meer realistische berekeningen
    """
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    df = df.copy()
    df['spread'] = df['price2'] - (model.intercept_ + model.coef_[0] * df['price1'])
    
    # Rolling statistics voor meer realistische berekeningen
    df['spread_mean'] = df['spread'].rolling(lookback_period).mean()
    df['spread_std'] = df['spread'].rolling(lookback_period).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    cash = initial_capital
    position = 0
    shares1 = 0
    shares2 = 0
    entry_price1 = 0
    entry_price2 = 0
    entry_date = None
    entry_zscore = 0
    max_position_value = (max_position_size / 100) * initial_capital
    
    trades = []
    portfolio_values = []
    positions = []
    drawdowns = []
    peak_value = initial_capital

    for i in range(lookback_period, len(df)):  # Start na lookback periode
        current_zscore = df['zscore'].iloc[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_date = df.index[i]
        
        # Bereken huidige portfolio waarde
        if position != 0:
            position_value = shares1 * current_price1 + shares2 * current_price2
            portfolio_value = cash + position_value
        else:
            portfolio_value = cash
        
        # Track drawdown
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        drawdown = (peak_value - portfolio_value) / peak_value * 100
        drawdowns.append(drawdown)
        
        # Entry logic - alleen als geen positie open staat
        if position == 0 and not pd.isna(current_zscore):
            if current_zscore < -entry_threshold:  # Long spread
                position = 1
                position_size = min(max_position_value, portfolio_value * 0.9)
                
                # Bereken hedge ratio dynamisch
                hedge_ratio = model.coef_[0]
                
                # Posities berekenen
                notional_per_leg = position_size / 2
                shares1 = notional_per_leg / current_price1  # Long asset 1
                shares2 = -(notional_per_leg / current_price2) * hedge_ratio  # Short asset 2
                
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                entry_zscore = current_zscore
                
                # Transaction costs
                total_notional = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                transaction_fee = total_notional * (transaction_cost / 100)
                cash -= transaction_fee
                
            elif current_zscore > entry_threshold:  # Short spread
                position = -1
                position_size = min(max_position_value, portfolio_value * 0.9)
                
                hedge_ratio = model.coef_[0]
                
                notional_per_leg = position_size / 2
                shares1 = -(notional_per_leg / current_price1)  # Short asset 1
                shares2 = (notional_per_leg / current_price2) * hedge_ratio  # Long asset 2
                
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                entry_zscore = current_zscore
                
                total_notional = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                transaction_fee = total_notional * (transaction_cost / 100)
                cash -= transaction_fee
        
        # Exit logic
        elif position != 0 and not pd.isna(current_zscore):
            exit_trade = False
            exit_reason = ""
            
            # Z-score mean reversion exit
            if (position == 1 and current_zscore > -exit_threshold) or \
               (position == -1 and current_zscore < exit_threshold):
                exit_trade = True
                exit_reason = "Mean reversion"
            
            # P&L berekening
            current_position_value = shares1 * current_price1 + shares2 * current_price2
            pnl = current_position_value
            pnl_pct = (pnl / max_position_value) * 100 if max_position_value > 0 else 0
            
            # Stop loss
            if pnl_pct < -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            
            # Take profit
            elif pnl_pct > take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            # Time-based exit (maximum 30 dagen)
            elif (current_date - entry_date).days > 30:
                exit_trade = True
                exit_reason = "Time exit"
            
            if exit_trade:
                # Sluit positie
                exit_value = shares1 * current_price1 + shares2 * current_price2
                total_notional = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                exit_transaction_fee = total_notional * (transaction_cost / 100)
                final_pnl = exit_value - exit_transaction_fee
                
                cash += final_pnl
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Position': 'Long Spread' if position == 1 else 'Short Spread',
                    'Entry Z-score': entry_zscore,
                    'Exit Z-score': current_zscore,
                    'Entry Price 1': entry_price1,
                    'Entry Price 2': entry_price2,
                    'Exit Price 1': current_price1,
                    'Exit Price 2': current_price2,
                    'Shares 1': shares1,
                    'Shares 2': shares2,
                    'Position Size': max_position_value,
                    'P&L': final_pnl,
                    'P&L %': (final_pnl / max_position_value) * 100,
                    'Exit Reason': exit_reason,
                    'Days
