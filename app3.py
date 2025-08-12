import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# =============================================================================
# CORE DATA CLASS - ALLE LOGICA IN 1 PLEK
# =============================================================================

class PairsTrader:
    """
    Centrale klasse voor alle pairs trading logica.
    Bereken alles 1x, gebruik overal.
    """
    
    def __init__(self, price1, price2, lookback_window=20):
        self.price1 = price1
        self.price2 = price2
        self.lookback_window = lookback_window
        
        # Bereken alles in init - 1x gedaan
        self._fit_model()
        self._calculate_signals()
    
    def _fit_model(self):
        """Fit regressie model 1x"""
        X = self.price1.values.reshape(-1, 1)
        y = self.price2.values
        
        self.model = LinearRegression().fit(X, y)
        self.alpha = self.model.intercept_
        self.beta = self.model.coef_[0]
        self.r_squared = self.model.score(X, y)
        
        # Bereken spread en z-score 1x
        self.spread = self.price2 - (self.alpha + self.beta * self.price1)
        self.spread_mean = self.spread.mean()
        self.spread_std = self.spread.std()
        self.zscore = (self.spread - self.spread_mean) / self.spread_std
        
        # Rolling stats voor backtest
        self.rolling_corr = self.price1.rolling(self.lookback_window).corr(self.price2)
        
        # Ratio
        self.ratio = self.price1 / self.price2
        
        # Returns
        self.returns1 = self.price1.pct_change()
        self.returns2 = self.price2.pct_change()
        
        # Current values
        self.current_price1 = self.price1.iloc[-1]
        self.current_price2 = self.price2.iloc[-1]
        self.current_zscore = self.zscore.iloc[-1]
        self.current_spread = self.spread.iloc[-1]
    
    def _calculate_signals(self):
        """Bereken trading signalen"""
        self.pearson_corr = self.price1.corr(self.price2)
    
    def get_position_size(self, capital, risk_pct=2.0):
        """Bereken position sizing"""
        risk_amount = capital * (risk_pct / 100)
        
        # Simple equal dollar weighting met hedge ratio correctie
        capital_per_leg = capital * 0.45  # 45% per leg, 10% cash buffer
        
        shares_asset1 = capital_per_leg / self.current_price1
        shares_asset2 = shares_asset1 * abs(self.beta)  # hedge ratio
        
        cost1 = shares_asset1 * self.current_price1
        cost2 = shares_asset2 * self.current_price2
        
        return {
            'shares_asset1': round(shares_asset1, 6),
            'shares_asset2': round(shares_asset2, 0),
            'cost1': cost1,
            'cost2': cost2,
            'total_cost': cost1 + cost2,
            'efficiency': ((cost1 + cost2) / capital) * 100
        }
    
    def get_exit_prices(self, target_zscore, price_buffer_pct=5.0):
        """Bereken exit price levels"""
        target_spread = self.spread_mean + target_zscore * self.spread_std
        buffer = price_buffer_pct / 100
        
        # Price ranges
        price1_range = (
            self.current_price1 * (1 - buffer),
            self.current_price1 * (1 + buffer)
        )
        price2_range = (
            self.current_price2 * (1 - buffer), 
            self.current_price2 * (1 + buffer)
        )
        
        # Exit triggers
        price2_when_p1_stable = target_spread + self.alpha + self.beta * self.current_price1
        price1_when_p2_stable = (self.current_price2 - self.alpha - target_spread) / self.beta if self.beta != 0 else self.current_price1
        
        return {
            'target_spread': target_spread,
            'target_zscore': target_zscore,
            'price1_range': price1_range,
            'price2_range': price2_range,
            'price2_trigger': price2_when_p1_stable,
            'price1_trigger': price1_when_p2_stable
        }
    
    def run_backtest(self, entry_threshold=2.0, exit_threshold=0.5, 
                    initial_capital=100000, position_size_pct=20):
        """Simple backtest"""
        
        # Rolling z-score voor realistic backtest
        rolling_zscore = self.zscore.rolling(self.lookback_window).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        trades = []
        cash = initial_capital
        position = None
        
        for i in range(self.lookback_window, len(rolling_zscore)):
            z = rolling_zscore.iloc[i]
            
            if pd.isna(z):
                continue
                
            # Entry logic
            if position is None:
                if z <= -entry_threshold:
                    position = {'type': 'LONG', 'entry_i': i, 'entry_z': z}
                elif z >= entry_threshold:
                    position = {'type': 'SHORT', 'entry_i': i, 'entry_z': z}
            
            # Exit logic  
            else:
                exit_trade = False
                if position['type'] == 'LONG' and z >= -exit_threshold:
                    exit_trade = True
                elif position['type'] == 'SHORT' and z <= exit_threshold:
                    exit_trade = True
                
                if exit_trade:
                    pnl = self._calculate_trade_pnl(position, i, z)
                    trades.append({
                        'entry_date': self.price1.index[position['entry_i']],
                        'exit_date': self.price1.index[i],
                        'type': position['type'],
                        'entry_z': position['entry_z'],
                        'exit_z': z,
                        'pnl': pnl,
                        'days': (self.price1.index[i] - self.price1.index[position['entry_i']]).days
                    })
                    cash += pnl
                    position = None
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def _calculate_trade_pnl(self, position, exit_i, exit_z):
        """Calculate trade P&L"""
        entry_spread = self.spread.iloc[position['entry_i']]
        exit_spread = self.spread.iloc[exit_i]
        
        if position['type'] == 'LONG':
            return (exit_spread - entry_spread) * 10000  # Scale factor
        else:
            return (entry_spread - exit_spread) * 10000

# =============================================================================
# STREAMLIT APP - CLEAN & SIMPLE
# =============================================================================

# Page config
st.set_page_config(layout="wide", page_title="Clean Pairs Trading")
st.title("🎯 Clean Pairs Trading Monitor")

# Load tickers (je moet dit aanpassen naar jouw ticker source)
@st.cache_data
def load_sample_tickers():
    return {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD', 
        'AAPL': 'AAPL',
        'GOOGL': 'GOOGL',
        'TSLA': 'TSLA',
        'MSFT': 'MSFT'
    }

tickers = load_sample_tickers()

# =============================================================================
# SIDEBAR - PARAMETERS
# =============================================================================

with st.sidebar:
    st.header("🎯 Pair Selection")
    all_tickers = list(tickers.keys())
    
    asset1 = st.selectbox("Asset 1", all_tickers, index=0)
    remaining = [t for t in all_tickers if t != asset1]
    asset2 = st.selectbox("Asset 2", remaining, index=0)
    
    st.markdown("---")
    st.header("📊 Data Settings")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    st.markdown("---")  
    st.header("⚙️ Trading Parameters")
    entry_threshold = st.slider("Entry Z-score", 1.0, 4.0, 2.0, 0.1)
    exit_threshold = st.slider("Exit Z-score", 0.0, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    st.header("💰 Capital Settings") 
    trading_capital = st.number_input("Trading Capital ($)", 500, 100000, 5000, 100)
    risk_per_trade = st.slider("Risk per Trade (%)", 1.0, 10.0, 2.0, 0.5)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(symbol, period):
    try:
        data = yf.download(symbol, period=period, progress=False)
        return data['Close'].dropna()
    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")
        return pd.Series()

# Load data
data1 = load_data(tickers[asset1], period)
data2 = load_data(tickers[asset2], period) 

if data1.empty or data2.empty:
    st.error("❌ No data available")
    st.stop()

# Align data
price1, price2 = data1.align(data2, join='inner')
price1 = price1.dropna()
price2 = price2.dropna()

if len(price1) < 20:
    st.error("❌ Not enough data points")
    st.stop()

# =============================================================================
# MAIN ANALYSIS - 1 OBJECT, ALL CALCULATIONS
# =============================================================================

trader = PairsTrader(price1, price2)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

# Key Metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Current Z-Score", f"{trader.current_zscore:.2f}")
with col2:
    signal = "🟢 LONG" if trader.current_zscore <= -entry_threshold else \
            "🔴 SHORT" if trader.current_zscore >= entry_threshold else "⏳ WAIT"
    st.metric("Signal", signal)
with col3:
    st.metric("Correlation", f"{trader.pearson_corr:.3f}")
with col4:
    st.metric("Hedge Ratio (β)", f"{trader.beta:.4f}")
with col5:
    st.metric("R²", f"{trader.r_squared:.3f}")

# =============================================================================
# CHARTS
# =============================================================================

st.subheader("📈 Charts")

# Create 2x2 subplot layout
fig = go.Figure()

# Z-score chart with signals
fig.add_trace(go.Scatter(
    x=trader.zscore.index,
    y=trader.zscore.values,
    name='Z-Score',
    line=dict(color='blue', width=2)
))

# Add threshold lines
fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", 
              annotation_text="SHORT Entry")
fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green",
              annotation_text="LONG Entry") 
fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange",
              annotation_text="Exit")
fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange")
fig.add_hline(y=0, line_color="black", line_width=1)

fig.update_layout(
    title="Z-Score with Trading Signals",
    xaxis_title="Date",
    yaxis_title="Z-Score",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Price charts
col1, col2 = st.columns(2)

with col1:
    # Dual axis price chart
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(
        x=price1.index, y=price1.values, name=asset1, line=dict(color='blue')
    ))
    fig_prices.add_trace(go.Scatter(
        x=price2.index, y=price2.values, name=asset2, yaxis='y2', 
        line=dict(color='red')
    ))
    fig_prices.update_layout(
        title="Price Movement",
        yaxis_title=f"{asset1} Price",
        yaxis2=dict(title=f"{asset2} Price", overlaying='y', side='right'),
        height=300
    )
    st.plotly_chart(fig_prices, use_container_width=True)

with col2:
    # Scatter plot
    fig_scatter = px.scatter(
        x=trader.returns1.dropna(), y=trader.returns2.dropna(),
        title=f"Returns Correlation",
        labels={'x': f'{asset1} Returns', 'y': f'{asset2} Returns'}
    )
    fig_scatter.update_traces(marker=dict(size=4, opacity=0.6))
    fig_scatter.update_layout(height=300)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================================
# TRADING EXECUTION
# =============================================================================

if abs(trader.current_zscore) >= entry_threshold:
    
    st.subheader("🎯 Trading Execution")
    
    # Get position sizing
    position = trader.get_position_size(trading_capital, risk_per_trade)
    
    # Determine trade type
    is_long_spread = trader.current_zscore <= -entry_threshold
    trade_type = "LONG SPREAD" if is_long_spread else "SHORT SPREAD"
    
    st.success(f"**{trade_type}** Signal Active (Z-Score: {trader.current_zscore:.2f})")
    
    # Position details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action1 = "🟢 BUY" if is_long_spread else "🔴 SHORT"
        st.markdown(f"""
        #### {action1} {asset1}
        - **Shares**: {position['shares_asset1']}
        - **Price**: ${trader.current_price1:.6f}
        - **Cost**: ${position['cost1']:.2f}
        """)
    
    with col2:
        action2 = "🔴 SHORT" if is_long_spread else "🟢 BUY"
        st.markdown(f"""
        #### {action2} {asset2}  
        - **Shares**: {position['shares_asset2']}
        - **Price**: ${trader.current_price2:.6f}
        - **Cost**: ${position['cost2']:.2f}
        """)
    
    with col3:
        st.markdown(f"""
        #### 📊 Summary
        - **Total Cost**: ${position['total_cost']:.2f}
        - **Capital Used**: {position['efficiency']:.1f}%
        - **Max Risk**: ${trading_capital * (risk_per_trade/100):.2f}
        """)
    
    # Exit Rules
    st.markdown("---")
    st.subheader("🚨 Exit Rules")
    
    profit_exit = trader.get_exit_prices(-exit_threshold if is_long_spread else exit_threshold)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("#### 🎯 PROFIT EXIT")
        st.markdown(f"""
        **Close Position When:**
        - {asset2} {'≤' if is_long_spread else '≥'} ${profit_exit['price2_trigger']:.6f}
        - OR {asset1} {'≥' if is_long_spread else '≤'} ${profit_exit['price1_trigger']:.6f}
        - **Target Z-Score**: {profit_exit['target_zscore']:.1f}
        """)
    
    with col2:
        stop_exit = trader.get_exit_prices(entry_threshold * 1.5 if is_long_spread else -entry_threshold * 1.5)
        st.error("#### 🛑 STOP LOSS")
        st.markdown(f"""
        **Emergency Exit When:**
        - Z-Score reaches {'≥' if is_long_spread else '≤'} {entry_threshold:.1f}
        - Position moves against us
        - **Max Loss**: ${trading_capital * (risk_per_trade/100):.2f}
        """)

else:
    st.info(f"⏳ **NO SIGNAL** - Z-Score: {trader.current_zscore:.2f} (Need ≥{entry_threshold:.1f} or ≤{-entry_threshold:.1f})")

# =============================================================================
# BACKTEST
# =============================================================================

st.subheader("📊 Quick Backtest")

if st.button("🚀 Run Backtest", type="primary"):
    with st.spinner("Running backtest..."):
        trades_df = trader.run_backtest(entry_threshold, exit_threshold, 100000, 20)
    
    if not trades_df.empty:
        # Performance metrics
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        num_trades = len(trades_df)
        avg_days = trades_df['days'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total P&L", f"${total_pnl:,.0f}")
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")  
        with col3:
            st.metric("# Trades", f"{num_trades}")
        with col4:
            st.metric("Avg Days", f"{avg_days:.1f}")
        
        # Show recent trades
        st.dataframe(trades_df.tail(10), use_container_width=True)
        
    else:
        st.warning("No trades generated with current parameters")

# =============================================================================
# FOOTER INFO
# =============================================================================

with st.expander("ℹ️ How It Works"):
    st.markdown(f"""
    **Current Setup:**
    - **Regression**: {asset2} = {trader.alpha:.6f} + {trader.beta:.6f} × {asset1}
    - **Spread**: {asset2} - ({trader.alpha:.6f} + {trader.beta:.6f} × {asset1})  
    - **Z-Score**: (Spread - {trader.spread_mean:.6f}) / {trader.spread_std:.6f}
    
    **Trading Logic:**
    - **LONG SPREAD**: Buy {asset1}, Short {asset2} when Z-Score ≤ -{entry_threshold}
    - **SHORT SPREAD**: Short {asset1}, Buy {asset2} when Z-Score ≥ +{entry_threshold}
    - **EXIT**: Close when Z-Score returns to ±{exit_threshold}
    
    **Risk Management:**  
    - Max {risk_per_trade}% risk per trade
    - Stop loss at Z-Score reversal
    - Position sizing based on hedge ratio
    """)
