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
with st.expander("📋 Trading Signals - Praktische Uitvoering", expanded=True):
    st.header("💰 Praktische Trade Uitvoering")
    
    # Huidige marktdata
    current_price1 = df['price1'].iloc[-1]
    current_price2 = df['price2'].iloc[-1]
    current_zscore = df['zscore'].iloc[-1]
    hedge_ratio = df['hedge_ratio'].iloc[-1]  # Aantal eenheden asset2 per eenheid asset1
    
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
        st.metric("Fair Value Spread", f"${fair_value2:.4f}")
    
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
    st.subheader("🔧 Trading Parameters")
    
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
    
    # Visualisatie van spread met trading levels
    fig = go.Figure()
    
    # Spread
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price2'] - df['price1']*df['hedge_ratio'],
        name='Actual Spread',
        line=dict(color='blue')
    ))
    
    # Mean
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[spread_mean]*len(df),
        name='Mean Spread',
        line=dict(color='green', dash='dash')
    ))
    
    # Entry levels
    entry_upper = spread_mean + zscore_entry_threshold * df['spread'].rolling(30).std().iloc[-1]
    entry_lower = spread_mean - zscore_entry_threshold * df['spread'].rolling(30).std().iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[entry_upper]*len(df),
        name='Short Entry',
        line=dict(color='red', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[entry_lower]*len(df),
        name='Long Entry',
        line=dict(color='green', dash='dot')
    ))
    
    # Exit levels
    exit_upper = spread_mean + zscore_exit_threshold * df['spread'].rolling(30).std().iloc[-1]
    exit_lower = spread_mean - zscore_exit_threshold * df['spread'].rolling(30).std().iloc[-1]
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[exit_upper]*len(df),
        name='Short Exit',
        line=dict(color='pink', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[exit_lower]*len(df),
        name='Long Exit',
        line=dict(color='lightgreen', dash='dot')
    ))
    
    fig.update_layout(
        title=f"Spread Trading Levels ({name1} vs {name2})",
        yaxis_title="Spread Value (USD)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
# === BACKTESTING SECTION ===
def run_backtest(df, entry_threshold, exit_threshold, initial_capital, transaction_cost, max_position_size, stop_loss_pct, take_profit_pct):
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    df['spread'] = df['price2'] - (model.intercept_ + model.coef_[0] * df['price1'])
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    cash = initial_capital
    position = 0
    coin1_shares = 0
    coin2_shares = 0
    entry_price1 = 0
    entry_price2 = 0
    entry_date = None
    position_value = 0
    max_position_value = (max_position_size / 100) * initial_capital
    
    trades = []
    portfolio_values = []
    positions = []

    for i in range(len(df)):
        current_zscore = df['zscore'].iloc[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_date = df.index[i]
        
        position_market_value = coin1_shares * current_price1 + coin2_shares * current_price2
        portfolio_value = cash + position_market_value
        
        # Entry logic
        if position == 0 and i > 0:
            if current_zscore < -entry_threshold:  # Long spread
                position = 1
                position_value = min(max_position_value, portfolio_value * 0.95)
                coin2_shares = (position_value / 2) / current_price2
                coin1_shares = -(position_value / 2) / current_price1
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                cash -= position_value * (transaction_cost / 100)
                
            elif current_zscore > entry_threshold:  # Short spread
                position = -1
                position_value = min(max_position_value, portfolio_value * 0.95)
                coin1_shares = (position_value / 2) / current_price1
                coin2_shares = -(position_value / 2) / current_price2
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                cash -= position_value * (transaction_cost / 100)
        
        # Exit logic
        elif position != 0:
            exit_trade = False
            exit_reason = ""
            
            if abs(current_zscore) < exit_threshold:
                exit_trade = True
                exit_reason = "Z-score exit"
            
            pnl_dollar = (coin1_shares * (current_price1 - entry_price1) + 
                         coin2_shares * (current_price2 - entry_price2))
            pnl_pct = (pnl_dollar / position_value) * 100
            
            if pnl_pct < -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            elif pnl_pct > take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            if exit_trade:
                final_pnl = pnl_dollar - (abs(coin1_shares * current_price1) + abs(coin2_shares * current_price2)) * (transaction_cost / 100)
                cash += (coin1_shares * current_price1 + coin2_shares * current_price2 + final_pnl)
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Position': 'Long Spread' if position == 1 else 'Short Spread',
                    'Entry Z-score': df['zscore'].loc[entry_date],
                    'Exit Z-score': current_zscore,
                    'Entry Price 1': entry_price1,
                    'Entry Price 2': entry_price2,
                    'Exit Price 1': current_price1,
                    'Exit Price 2': current_price2,
                    'Coin1 Shares': coin1_shares,
                    'Coin2 Shares': coin2_shares,
                    'Position Size': position_value,
                    'P&L': final_pnl,
                    'P&L %': (final_pnl / position_value) * 100,
                    'Exit Reason': exit_reason,
                    'Days Held': (current_date - entry_date).days
                })
                
                position = 0
                coin1_shares = 0
                coin2_shares = 0
                entry_price1 = 0
                entry_price2 = 0
                entry_date = None
                position_value = 0
        
        portfolio_values.append(portfolio_value)
        positions.append(position)
    
    df_result = df.copy()
    df_result['portfolio_value'] = portfolio_values
    df_result['position'] = positions
    return df_result, trades

with st.expander("🔙 Backtesting", expanded=False):
    st.header("🔙 Backtesting")
    
    with st.sidebar:
        st.markdown("---")
        st.header("🎯 Backtesting Settings")
        initial_capital = st.number_input("Initial Capital (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
        transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        max_position_size = st.slider("Max position size (% of capital)", min_value=10, max_value=100, value=50, step=10)
        
        st.subheader("🛡️ Risk Management")
        stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        take_profit_pct = st.slider("Take Profit (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    
    if st.button("Run Backtest"):
        df_backtest, trades = run_backtest(
            df, zscore_entry_threshold, zscore_exit_threshold,
            initial_capital, transaction_cost, max_position_size,
            stop_loss_pct, take_profit_pct
        )
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            final_value = df_backtest['portfolio_value'].iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            
            st.success(f"Backtest complete! Total return: {total_return:.2f}%")
            
            tab1, tab2 = st.tabs(["Performance Metrics", "Trade History"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Value", f"${final_value:,.2f}")
                    st.metric("Number of Trades", len(trades_df))
                    winning_trades = trades_df[trades_df['P&L'] > 0]
                    st.metric("Win Rate", f"{len(winning_trades)/len(trades_df)*100:.1f}%")
                
                with col2:
                    st.metric("Avg Win Trade", f"${winning_trades['P&L'].mean():.2f}")
                    losing_trades = trades_df[trades_df['P&L'] < 0]
                    st.metric("Avg Loss Trade", f"${losing_trades['P&L'].mean():.2f}")
                    st.metric("Profit Factor", f"{abs(winning_trades['P&L'].sum()/losing_trades['P&L'].sum()):.2f}")
            
            with tab2:
                st.dataframe(trades_df.sort_values('Exit Date', ascending=False))
            
            # Portfolio value chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_backtest.index,
                y=df_backtest['portfolio_value'],
                name='Portfolio Value'
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trades executed during backtest")

# Export functionality
with st.expander("📤 Export", expanded=False):
    st.header("📤 Export Data")
    if 'df_backtest' in locals():
        if st.button("Export backtest results"):
            csv = df_backtest.to_csv(index=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"backtest_results_{name1}_{name2}.csv",
                mime='text/csv'
            )
