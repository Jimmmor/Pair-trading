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

# Page configuration
st.set_page_config(layout="wide")
st.title("Advanced Pairs Trading Monitor")

class PairsDataProcessor:
    """Unified data processing and statistical calculations"""
    
    def __init__(self, zscore_window=20, correlation_window=20):
        self.zscore_window = zscore_window
        self.correlation_window = correlation_window
        self.model = LinearRegression()
        
    def process_pair_data(self, data1, data2):
        """Main data processing pipeline"""
        # Align data
        df = self._align_data(data1, data2)
        
        # Calculate regression and spread
        df = self._calculate_regression_spread(df)
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        return df
    
    def _align_data(self, data1, data2):
        """Align and clean data"""
        if not isinstance(data1, pd.Series):
            data1 = pd.Series(data1)
        if not isinstance(data2, pd.Series):
            data2 = pd.Series(data2)
        
        data1_aligned, data2_aligned = data1.align(data2, join='inner')
        
        return pd.DataFrame({
            'price1': data1_aligned,
            'price2': data2_aligned
        }).dropna()
    
    def _calculate_regression_spread(self, df):
        """Calculate regression parameters and spread - UNIFIED VERSION"""
        X = df['price1'].values.reshape(-1, 1)
        y = df['price2'].values
        
        self.model.fit(X, y)
        self.alpha = self.model.intercept_
        self.beta = self.model.coef_[0]
        self.r_squared = self.model.score(X, y)
        
        # UNIFIED SPREAD CALCULATION: price2 - (alpha + beta * price1)
        df['spread'] = df['price2'] - (self.alpha + self.beta * df['price1'])
        
        return df
    
    def _calculate_indicators(self, df):
        """Calculate all technical indicators"""
        # Rolling statistics for Z-score
        df['spread_mean'] = df['spread'].rolling(self.zscore_window).mean()
        df['spread_std'] = df['spread'].rolling(self.zscore_window).std()
        df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        # Correlation metrics
        df['rolling_corr'] = df['price1'].rolling(self.correlation_window).corr(df['price2'])
        self.pearson_corr = df['price1'].corr(df['price2'])
        
        # Additional metrics
        df['ratio'] = df['price1'] / df['price2']
        df['returns1'] = df['price1'].pct_change()
        df['returns2'] = df['price2'].pct_change()
        
        return df

class PairsTradingCalculator:
    """Unified trading calculations and position sizing"""
    
    def __init__(self, leverage=1, risk_per_trade=2.0):
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade / 100
        
    def calculate_balanced_positions(self, capital, price1, price2, hedge_ratio, 
                                   max_position_ratio=3.0, min_notional=1.0):
        """Robust position sizing with proper hedge ratio"""
        # Safety guards
        price1 = max(price1, 1e-12)
        price2 = max(price2, 1e-12)
        hedge_ratio = hedge_ratio if abs(hedge_ratio) > 1e-12 else 1.0
        
        capital_per_leg = capital * 0.49
        
        # Calculate quantities using hedge ratio
        if abs(hedge_ratio) < 1:
            # Asset2 needs fewer units
            quantity1 = capital_per_leg / price1
            quantity2 = quantity1 * abs(hedge_ratio)
        else:
            # Asset1 needs fewer units
            quantity2 = capital_per_leg / price2
            quantity1 = quantity2 / abs(hedge_ratio)
        
        # Calculate costs
        cost1 = quantity1 * price1
        cost2 = quantity2 * price2
        
        # Enforce minimum notional
        total_cost = cost1 + cost2
        if total_cost > capital:
            scale_factor = capital / total_cost
            quantity1 *= scale_factor
            quantity2 *= scale_factor
            cost1 *= scale_factor
            cost2 *= scale_factor
        
        # Dynamic rounding based on price
        def smart_round(qty, price):
            if price < 0.01:
                return round(qty, 0)
            elif price < 1:
                return round(qty, 4)
            elif price < 100:
                return round(qty, 6)
            else:
                return round(qty, 8)
        
        quantity1 = smart_round(quantity1, price1)
        quantity2 = smart_round(quantity2, price2)
        
        final_cost1 = quantity1 * price1
        final_cost2 = quantity2 * price2
        
        return quantity1, quantity2, final_cost1, final_cost2
    
    def calculate_price_targets(self, current_price1, current_price2, 
                              target_zscore, spread_mean, spread_std, alpha, beta):
        """Calculate price targets for given Z-score"""
        target_spread = spread_mean + target_zscore * spread_std
        
        # Multiple scenarios for price targets
        scenarios = {}
        
        # If price1 moves ±10%, what should price2 be?
        for pct in [-10, -5, 0, 5, 10]:
            new_price1 = current_price1 * (1 + pct/100)
            required_price2 = alpha + beta * new_price1 + target_spread
            scenarios[f'price1_{pct:+d}pct'] = {
                'price1': new_price1,
                'required_price2': required_price2,
                'price2_change_pct': ((required_price2 - current_price2) / current_price2) * 100
            }
        
        return scenarios

class PairsBacktester:
    """Realistic pairs trading backtest engine"""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.05, 
                 position_size_pct=20, stop_loss_pct=5.0, take_profit_pct=8.0,
                 max_trade_days=30):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost / 100
        self.position_size_pct = position_size_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.max_trade_days = max_trade_days
        
    def run_backtest(self, df, entry_threshold, exit_threshold, lookback_window=20):
        """Execute realistic pairs trading backtest"""
        # Initialize tracking
        cash = self.initial_capital
        position_active = False
        position_data = {}
        trades = []
        daily_portfolio = []
        
        for i in range(lookback_window, len(df)):
            current_date = df.index[i]
            current_data = df.iloc[i]
            
            if pd.isna(current_data['zscore']):
                continue
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(
                cash, position_active, position_data, current_data
            )
            
            # Record daily data
            daily_portfolio.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'zscore': current_data['zscore'],
                'position_active': position_active
            })
            
            # Trading logic
            if not position_active:
                # Check for entry signals
                signal = self._check_entry_signals(current_data, entry_threshold)
                if signal:
                    position_data, cash = self._enter_position(
                        signal, current_data, current_date, cash, portfolio_value
                    )
                    position_active = True
            else:
                # Check for exit signals
                exit_signal = self._check_exit_signals(
                    current_data, position_data, exit_threshold
                )
                if exit_signal['should_exit']:
                    trade_result, cash = self._exit_position(
                        position_data, current_data, exit_signal['reason']
                    )
                    trades.append(trade_result)
                    position_active = False
                    position_data = {}
        
        return pd.DataFrame(daily_portfolio), pd.DataFrame(trades)
    
    def _calculate_portfolio_value(self, cash, position_active, position_data, current_data):
        """Calculate current portfolio value"""
        if not position_active:
            return cash
        
        position_value = (position_data['shares1'] * current_data['price1'] + 
                         position_data['shares2'] * current_data['price2'])
        return cash + position_value
    
    def _check_entry_signals(self, current_data, entry_threshold):
        """Check for entry signals"""
        if current_data['zscore'] <= -entry_threshold:
            return 'long_spread'
        elif current_data['zscore'] >= entry_threshold:
            return 'short_spread'
        return None
    
    def _check_exit_signals(self, current_data, position_data, exit_threshold):
        """Check for exit signals"""
        days_held = (current_data.name - position_data['entry_date']).days
        
        # P&L calculation
        current_value = (position_data['shares1'] * current_data['price1'] + 
                        position_data['shares2'] * current_data['price2'])
        pnl_pct = ((current_value - position_data['initial_value']) / 
                   position_data['initial_value']) * 100
        
        # Exit conditions
        if position_data['type'] == 'long_spread' and current_data['zscore'] >= -exit_threshold:
            return {'should_exit': True, 'reason': 'Mean reversion'}
        elif position_data['type'] == 'short_spread' and current_data['zscore'] <= exit_threshold:
            return {'should_exit': True, 'reason': 'Mean reversion'}
        elif pnl_pct <= -self.stop_loss_pct * 100:
            return {'should_exit': True, 'reason': 'Stop loss'}
        elif pnl_pct >= self.take_profit_pct * 100:
            return {'should_exit': True, 'reason': 'Take profit'}
        elif days_held >= self.max_trade_days:
            return {'should_exit': True, 'reason': 'Time limit'}
        
        return {'should_exit': False, 'reason': None}
    
    def _enter_position(self, signal_type, current_data, current_date, cash, portfolio_value):
        """Enter new position"""
        position_size = self.position_size_pct * portfolio_value
        capital_per_leg = position_size / 2
        
        if signal_type == 'long_spread':
            # Long asset1, short asset2
            shares1 = capital_per_leg / current_data['price1']
            shares2 = -shares1 * processor.beta  # Negative = short
        else:
            # Short asset1, long asset2
            shares1 = -capital_per_leg / current_data['price1']  # Negative = short
            shares2 = abs(shares1) * processor.beta
        
        initial_value = abs(shares1 * current_data['price1']) + abs(shares2 * current_data['price2'])
        transaction_costs = initial_value * self.transaction_cost
        
        position_data = {
            'type': signal_type,
            'entry_date': current_date,
            'entry_zscore': current_data['zscore'],
            'entry_price1': current_data['price1'],
            'entry_price2': current_data['price2'],
            'shares1': shares1,
            'shares2': shares2,
            'initial_value': initial_value,
            'hedge_ratio': processor.beta
        }
        
        return position_data, cash - transaction_costs
    
    def _exit_position(self, position_data, current_data, exit_reason):
        """Exit current position"""
        current_value = (position_data['shares1'] * current_data['price1'] + 
                        position_data['shares2'] * current_data['price2'])
        
        exit_costs = abs(current_value) * self.transaction_cost
        net_pnl = current_value - position_data['initial_value'] - exit_costs
        
        trade_result = {
            'entry_date': position_data['entry_date'],
            'exit_date': current_data.name,
            'position_type': position_data['type'],
            'days_held': (current_data.name - position_data['entry_date']).days,
            'entry_zscore': position_data['entry_zscore'],
            'exit_zscore': current_data['zscore'],
            'net_pnl': net_pnl,
            'pnl_percentage': (net_pnl / position_data['initial_value']) * 100,
            'exit_reason': exit_reason,
            'shares1': position_data['shares1'],
            'shares2': position_data['shares2']
        }
        
        return trade_result, current_value - exit_costs

@st.cache_data
def load_data(ticker_key, period, interval):
    """Load and cache market data"""
    try:
        ticker_symbol = tickers[ticker_key]
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].dropna()
        return data['Close'].dropna()
        
    except Exception as e:
        st.error(f"Error loading data for {ticker_key}: {e}")
        return pd.Series()

# Sidebar Configuration
with st.sidebar:
    st.header("Pair Selection")
    all_tickers = list(tickers.keys())
    
    name1 = st.selectbox("Asset 1", all_tickers, index=0)
    remaining_tickers = [t for t in all_tickers if t != name1]
    name2 = st.selectbox("Asset 2", remaining_tickers, index=0)
    
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

# Load and process data
data1 = load_data(name1, periode, interval)
data2 = load_data(name2, periode, interval)

if data1.empty or data2.empty:
    st.error("No data available for one or both assets")
    st.stop()

# Initialize processors
processor = PairsDataProcessor(zscore_window=corr_window, correlation_window=corr_window)
calculator = PairsTradingCalculator(leverage=leverage, risk_per_trade=risk_per_trade)

# Process data
df = processor.process_pair_data(data1, data2)

# Current market state
current_price1 = df['price1'].iloc[-1]
current_price2 = df['price2'].iloc[-1]
current_zscore = df['zscore'].iloc[-1]

# === STATISTICAL ANALYSIS SECTION ===
with st.expander("📊 Statistical Analysis", expanded=True):
    st.header("📊 Statistical Analysis")
    
    tab1, tab2 = st.tabs(["Price Analysis", "Correlation Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Dual Y-axis price chart
            fig_prices = go.Figure()
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price1'], name=name1, line=dict(color='blue')))
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price2'], name=name2, 
                line=dict(color='red'), yaxis='y2'))
            
            fig_prices.update_layout(
                title="Price Movement Comparison",
                xaxis_title="Date",
                yaxis_title=f"{name1} Price (USD)",
                yaxis2=dict(title=f"{name2} Price (USD)", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
            
            # Spread visualization
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=df.index, y=df['spread'], name='Spread',
                line=dict(color='green'), fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            fig_spread.update_layout(
                title="Regression Spread",
                xaxis_title="Date", yaxis_title="Spread Value",
                height=400
            )
            st.plotly_chart(fig_spread, use_container_width=True)
        
        with col2:
            # Price ratio
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=df.index, y=df['ratio'], name=f"{name1}/{name2}",
                line=dict(color='purple'), fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            fig_ratio.update_layout(
                title=f"{name1}/{name2} Price Ratio",
                xaxis_title="Date", yaxis_title="Ratio",
                height=400
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Z-score with trading zones
            fig_zscore = go.Figure()
            fig_zscore.add_trace(go.Scatter(
                x=df.index, y=df['zscore'], name='Z-score',
                line=dict(color='#2ca02c', width=2)
            ))
            
            # Trading thresholds
            for threshold, color, label in [
                (-zscore_entry_threshold, 'green', 'LONG ENTRY'),
                (-zscore_exit_threshold, 'blue', 'LONG EXIT'),
                (zscore_entry_threshold, 'red', 'SHORT ENTRY'),
                (zscore_exit_threshold, 'purple', 'SHORT EXIT')
            ]:
                fig_zscore.add_hline(y=threshold, line=dict(color=color, dash='dash'),
                                   annotation_text=label)
            
            fig_zscore.add_hline(y=0, line=dict(color='black', width=1))
            fig_zscore.update_layout(
                title="Z-score Trading Signals",
                xaxis_title="Date", yaxis_title="Z-score",
                height=400
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rolling correlation
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df.index, y=df['rolling_corr'], name='Rolling Correlation',
                line=dict(color='blue')
            ))
            fig_corr.update_layout(
                title=f"Rolling Correlation ({corr_window}d)",
                xaxis_title="Date", yaxis_title="Correlation",
                yaxis_range=[-1, 1], height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Key statistics
            st.subheader("Key Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Pearson Correlation", f"{processor.pearson_corr:.4f}")
                st.metric("Beta (Hedge Ratio)", f"{processor.beta:.4f}")
            with col_b:
                st.metric("Current Correlation", f"{df['rolling_corr'].iloc[-1]:.4f}")
                st.metric("R-squared", f"{processor.r_squared:.4f}")
        
        with col2:
            # Returns scatterplot
            fig_scatter = px.scatter(
                df.dropna(), x='returns1', y='returns2',
                title=f"Returns Correlation: {name1} vs {name2}",
                labels={'returns1': f'{name1} Returns', 'returns2': f'{name2} Returns'}
            )
            fig_scatter.update_traces(marker=dict(size=6, opacity=0.6))
            st.plotly_chart(fig_scatter, use_container_width=True)

# === PRACTICAL TRADING EXECUTION ===
with st.expander("🎯 Praktische Trade Uitvoering", expanded=True):
    st.header("🎯 Live Trading Execution Dashboard")
    
    # Capital input
    col1, col2, col3 = st.columns(3)
    with col1:
        trading_capital = st.number_input("💵 Trading Capital (USDT)", 
                                        min_value=50, max_value=1000000, 
                                        value=1000, step=50)
    with col2:
        max_risk_pct = st.slider("🎯 Max Risk per Trade (%)", 1.0, 10.0, 2.0, 0.5)
    with col3:
        max_risk_usdt = trading_capital * (max_risk_pct / 100)
        st.metric("Max Risk Amount", f"{max_risk_usdt:.2f} USDT")
    
    # Current market status
    st.subheader("📊 Current Market Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(f"{name1} Price", f"{current_price1:.8f} USDT")
    with col2:
        st.metric(f"{name2} Price", f"{current_price2:.8f} USDT")
    with col3:
        signal_status = "🚨 TRADE SIGNAL!" if abs(current_zscore) >= zscore_entry_threshold else "⏳ Wait"
        st.metric("Z-Score", f"{current_zscore:.2f}", delta=signal_status)
    with col4:
        st.metric("Hedge Ratio (β)", f"{processor.beta:.6f}")
    
    # Trading decision and execution
    st.markdown("---")
    
    if abs(current_zscore) >= zscore_entry_threshold:
        signal_type = "LONG SPREAD" if current_zscore <= -zscore_entry_threshold else "SHORT SPREAD"
        signal_color = "success" if current_zscore <= -zscore_entry_threshold else "error"
        
        getattr(st, signal_color)(f"🎯 **{signal_type} SIGNAL** - Z-Score: {current_zscore:.2f}")
        
        # Calculate positions
        qty1, qty2, cost1, cost2 = calculator.calculate_balanced_positions(
            trading_capital, current_price1, current_price2, processor.beta
        )
        
        total_cost = cost1 + cost2
        
        # Position details
        st.subheader("📋 Exact Trade Execution")
        
        col1, col2 = st.columns(2)
        
        if current_zscore <= -zscore_entry_threshold:
            # LONG SPREAD
            with col1:
                st.success(f"""
                #### 🟢 BUY {name1}
                - **Quantity**: {qty1:.6f} {name1}
                - **Price**: {current_price1:.8f} USDT
                - **Cost**: {cost1:.2f} USDT
                - **Order**: MARKET BUY
                """)
            with col2:
                st.error(f"""
                #### 🔴 SHORT {name2}
                - **Quantity**: {qty2:.0f} {name2}
                - **Price**: {current_price2:.8f} USDT
                - **Value**: {cost2:.2f} USDT
                - **Order**: MARKET SELL (FUTURES)
                """)
        else:
            # SHORT SPREAD
            with col1:
                st.error(f"""
                #### 🔴 SHORT {name1}
                - **Quantity**: {qty1:.6f} {name1}
                - **Price**: {current_price1:.8f} USDT
                - **Value**: {cost1:.2f} USDT
                - **Order**: MARKET SELL (FUTURES)
                """)
            with col2:
                st.success(f"""
                #### 🟢 BUY {name2}
                - **Quantity**: {qty2:.0f} {name2}
                - **Price**: {current_price2:.8f} USDT
                - **Cost**: {cost2:.2f} USDT
                - **Order**: MARKET BUY
                """)
        
        st.info(f"**Total Used**: {total_cost:.2f} USDT / {trading_capital:.2f} USDT "
                f"({(total_cost/trading_capital)*100:.1f}% efficiency)")
        
        # Price targets
        st.subheader("🎯 Price Targets & Alerts")
        
        exit_zscore = -zscore_exit_threshold if current_zscore <= -zscore_entry_threshold else zscore_exit_threshold
        stop_zscore = zscore_entry_threshold * 1.5 if current_zscore <= -zscore_entry_threshold else -zscore_entry_threshold * 1.5
        
        profit_targets = calculator.calculate_price_targets(
            current_price1, current_price2, exit_zscore,
            df['spread_mean'].iloc[-1], df['spread_std'].iloc[-1],
            processor.alpha, processor.beta
        )
        
        stop_targets = calculator.calculate_price_targets(
            current_price1, current_price2, stop_zscore,
            df['spread_mean'].iloc[-1], df['spread_std'].iloc[-1],
            processor.alpha, processor.beta
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("#### 🎯 PROFIT TARGETS")
            st.markdown(f"""
            **Target Z-score: {exit_zscore:.1f}**
            
            **Price Scenarios:**
            - If {name1} +5%: {name2} → {profit_targets['price1_+5pct']['required_price2']:.8f}
            - If {name1} -5%: {name2} → {profit_targets['price1_-5pct']['required_price2']:.8f}
            - If {name2} unchanged: Close at Z = {exit_zscore:.1f}
            """)
        
        with col2:
            st.error("#### 🛑 STOP LOSS")
            st.markdown(f"""
            **Stop Z-score: {stop_zscore:.1f}**
            
            **Emergency Exit:**
            - If {name1} moves against: {name2} → {stop_targets['price1_0pct']['required_price2']:.8f}
            - **Max Loss**: {max_risk_usdt:.2f} USDT
            - **Emergency close** if correlation breaks
            """)
    
    else:
        st.info(f"⏳ **NO SIGNAL** - Z-Score: {current_zscore:.2f}")
        
        distance_to_long = abs(current_zscore - (-zscore_entry_threshold))
        distance_to_short = abs(current_zscore - zscore_entry_threshold)
        
        st.markdown(f"""
        ### ⌛ Waiting for Signal
        
        **Entry Levels:**
        - 🟢 LONG SPREAD: Z ≤ -{zscore_entry_threshold:.1f} (need {distance_to_long:.2f} more)
        - 🔴 SHORT SPREAD: Z ≥ +{zscore_entry_threshold:.1f} (need {distance_to_short:.2f} more)
        """)

# === REALISTIC BACKTEST SECTION ===
with st.expander("🎯 Realistic Pairs Backtest", expanded=True):
    st.header("🎯 Professional Backtest Engine")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Capital Settings")
        initial_capital = st.number_input("Initial Capital ($)", 1000, 10000000, 100000, 1000)
        position_size_pct = st.slider("Position Size (%)", 5.0, 50.0, 20.0, 2.5)
        transaction_cost = st.slider("Transaction Cost (%)", 0.01, 0.5, 0.05, 0.01)
    
    with col2:
        st.subheader("🎯 Risk Management")
        stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 15.0, 5.0, 0.5)
        take_profit_pct = st.slider("Take Profit (%)", 2.0, 20.0, 8.0, 0.5)
        max_trade_days = st.slider("Max Days per Trade", 5, 90, 30, 5)
    
    with col3:
        st.subheader("📊 Technical Settings")
        lookback_window = st.slider("Z-score Lookback", 10, 100, 20, 5)
        st.metric("Entry Threshold", f"±{zscore_entry_threshold:.1f}")
        st.metric("Exit Threshold", f"±{zscore_exit_threshold:.1f}")
    
    # Run backtest
    if st.button("🚀 Run Professional Backtest", type="primary"):
        with st.spinner("Running comprehensive backtest..."):
            backtest_engine = PairsBacktester(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                position_size_pct=position_size_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_trade_days=max_trade_days
            )
            
            portfolio_df, trades_df = backtest_engine.run_backtest(
                df, zscore_entry_threshold, zscore_exit_threshold, lookback_window
            )
        
        if trades_df.empty:
            st.warning("⚠️ No trades generated. Adjust Z-score thresholds or time period.")
        else:
            # Performance metrics
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            num_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
            win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
            
            avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if winning_trades < num_trades else 0
            
            # Risk metrics
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            max_drawdown = 0
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min()) * 100
            
            volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else 0
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            st.success(f"✅ Backtest Complete: {num_trades} trades executed")
            
            # Performance summary
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Return", f"{total_return:.2f}%", 
                         delta=f"${final_value - initial_capital:,.0f}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                         delta=f"{winning_trades}/{num_trades}")
            with col3:
                st.metric("Avg Win/Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")
            with col4:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with col5:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            # Portfolio performance chart
            st.subheader("📈 Portfolio Performance Over Time")
            
            fig_performance = go.Figure()
            
            # Portfolio value line
            fig_performance.add_trace(go.Scatter(
                x=portfolio_df['date'],
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add trade markers
            for _, trade in trades_df.iterrows():
                color = 'green' if trade['position_type'] == 'long_spread' else 'red'
                symbol = 'triangle-up' if trade['net_pnl'] > 0 else 'triangle-down'
                
                # Entry marker
                entry_portfolio_value = portfolio_df[portfolio_df['date'] == trade['entry_date']]['portfolio_value'].iloc[0]
                fig_performance.add_trace(go.Scatter(
                    x=[trade['entry_date']],
                    y=[entry_portfolio_value],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='circle'),
                    name=f"Entry {trade['position_type']}",
                    showlegend=False,
                    hovertemplate=f"<b>Entry</b><br>Type: {trade['position_type']}<br>Z-score: {trade['entry_zscore']:.2f}<extra></extra>"
                ))
                
                # Exit marker
                exit_portfolio_value = portfolio_df[portfolio_df['date'] == trade['exit_date']]['portfolio_value'].iloc[0]
                fig_performance.add_trace(go.Scatter(
                    x=[trade['exit_date']],
                    y=[exit_portfolio_value],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol=symbol),
                    name=f"Exit {trade['position_type']}",
                    showlegend=False,
                    hovertemplate=f"<b>Exit</b><br>P&L: ${trade['net_pnl']:.0f}<br>Reason: {trade['exit_reason']}<extra></extra>"
                ))
            
            fig_performance.update_layout(
                title=f"Backtest Results: {name1} vs {name2}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Detailed trade analysis
            if not trades_df.empty:
                st.subheader("📋 Trade History & Analysis")
                
                # Format trades for display
                display_trades = trades_df.copy()
                
                # Format numeric columns
                for col in ['entry_zscore', 'exit_zscore', 'net_pnl', 'pnl_percentage']:
                    if col in display_trades.columns:
                        if 'pnl' in col and 'percentage' not in col:
                            display_trades[col] = display_trades[col].apply(lambda x: f"${x:,.2f}")
                        else:
                            display_trades[col] = display_trades[col].round(3)
                
                # Select key columns for display
                key_columns = [
                    'entry_date', 'exit_date', 'position_type', 'days_held',
                    'entry_zscore', 'exit_zscore', 'net_pnl', 'pnl_percentage', 'exit_reason'
                ]
                
                display_df = display_trades[key_columns].rename(columns={
                    'entry_date': 'Entry Date',
                    'exit_date': 'Exit Date',
                    'position_type': 'Position Type',
                    'days_held': 'Days Held',
                    'entry_zscore': 'Entry Z-Score',
                    'exit_zscore': 'Exit Z-Score',
                    'net_pnl': 'Net P&L',
                    'pnl_percentage': 'P&L %',
                    'exit_reason': 'Exit Reason'
                })
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Analysis charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # P&L distribution
                    fig_pnl = px.histogram(
                        trades_df, x='net_pnl', nbins=min(20, len(trades_df)),
                        title="P&L Distribution",
                        color_discrete_sequence=['lightblue']
                    )
                    fig_pnl.update_layout(
                        xaxis_title="Net P&L ($)",
                        yaxis_title="Number of Trades"
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                
                with col2:
                    # Performance by position type
                    if len(trades_df['position_type'].unique()) > 1:
                        fig_by_type = px.box(
                            trades_df, x='position_type', y='net_pnl',
                            title="P&L by Position Type",
                            color='position_type'
                        )
                        st.plotly_chart(fig_by_type, use_container_width=True)
                    else:
                        # Trade duration if only one position type
                        fig_duration = px.histogram(
                            trades_df, x='days_held', nbins=min(15, len(trades_df)),
                            title="Trade Duration Distribution",
                            color_discrete_sequence=['lightgreen']
                        )
                        st.plotly_chart(fig_duration, use_container_width=True)
                
                # Performance summary by position type
                if len(trades_df['position_type'].unique()) > 1:
                    st.subheader("📊 Performance by Strategy")
                    
                    perf_summary = trades_df.groupby('position_type').agg({
                        'net_pnl': ['count', 'mean', 'sum'],
                        'pnl_percentage': 'mean',
                        'days_held': 'mean'
                    }).round(2)
                    
                    perf_summary.columns = ['# Trades', 'Avg P&L ($)', 'Total P&L ($)', 'Avg P&L %', 'Avg Days']
                    st.dataframe(perf_summary, use_container_width=True)

# === RISK MANAGEMENT & ALERTS ===
with st.expander("⚠️ Risk Management & Alerts", expanded=False):
    st.header("⚠️ Advanced Risk Management")
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current Risk Metrics")
        
        # Correlation quality
        recent_corr = df['rolling_corr'].tail(10).mean()
        corr_stability = df['rolling_corr'].tail(20).std()
        
        st.metric("Recent Correlation", f"{recent_corr:.3f}")
        st.metric("Correlation Stability", f"{corr_stability:.3f}")
        st.metric("Spread Volatility", f"{df['spread'].std():.6f}")
        
        # Risk warnings
        if recent_corr < 0.5:
            st.error("🚨 LOW CORRELATION WARNING")
        elif corr_stability > 0.2:
            st.warning("⚠️ UNSTABLE CORRELATION")
        elif abs(current_zscore) > 4:
            st.error("🚨 EXTREME Z-SCORE - High Risk")
    
    with col2:
        st.subheader("🔔 Recommended Alerts")
        
        # Calculate alert levels
        alert_long_entry = -zscore_entry_threshold
        alert_short_entry = zscore_entry_threshold
        alert_correlation_break = 0.3
        
        st.markdown(f"""
        **Set these TradingView/Exchange alerts:**
        
        📱 **Entry Signals:**
        - Z-score ≤ {alert_long_entry:.1f} → LONG SPREAD
        - Z-score ≥ {alert_short_entry:.1f} → SHORT SPREAD
        
        📱 **Risk Alerts:**
        - Correlation < {alert_correlation_break:.1f} → EXIT ALL
        - Z-score > 4.0 → EXTREME RISK
        - {name1} daily change > 15% → REVIEW
        - {name2} daily change > 15% → REVIEW
        
        📱 **Exit Signals:**
        - Z-score crosses ±{zscore_exit_threshold:.1f} → CLOSE
        - P&L < -{risk_per_trade}% → STOP LOSS
        """)

# === EXCHANGE EXECUTION GUIDE ===
with st.expander("🏪 Exchange Execution Guide", expanded=False):
    st.header("🏪 Step-by-Step Exchange Orders")
    
    if abs(current_zscore) >= zscore_entry_threshold:
        signal_type = "LONG SPREAD" if current_zscore <= -zscore_entry_threshold else "SHORT SPREAD"
        
        # Calculate positions for guide
        qty1, qty2, cost1, cost2 = calculator.calculate_balanced_positions(
            trading_capital, current_price1, current_price2, processor.beta
        )
        
        st.subheader(f"🎯 {signal_type} Execution on Binance/Bybit")
        
        if current_zscore <= -zscore_entry_threshold:
            # LONG SPREAD instructions
            st.markdown(f"""
            ### 📋 LONG SPREAD Orders:
            
            **STEP 1: BUY {name1} (SPOT)**
            ```
            Exchange: Binance/Bybit Spot
            Pair: {name1}/USDT
            Order Type: MARKET BUY
            Quantity: {qty1:.6f} {name1}
            Estimated Cost: ~{cost1:.2f} USDT
            ```
            
            **STEP 2: SHORT {name2} (FUTURES)**
            ```
            Exchange: Binance/Bybit Futures
            Pair: {name2}/USDT (Perpetual)
            Order Type: MARKET SELL (Open Short)
            Quantity: {qty2:.0f} {name2}
            Leverage: 1x-3x (recommended)
            Margin Required: ~{cost2:.2f} USDT
            ```
            
            **STEP 3: SET CONDITIONAL ORDERS**
            ```
            Take Profit: Z-score ≥ -{zscore_exit_threshold:.1f}
            Stop Loss: Portfolio loss > {risk_per_trade}%
            Time Exit: Close after {max_trade_days} days
            ```
            """)
        else:
            # SHORT SPREAD instructions
            st.markdown(f"""
            ### 📋 SHORT SPREAD Orders:
            
            **STEP 1: SHORT {name1} (FUTURES)**
            ```
            Exchange: Binance/Bybit Futures
            Pair: {name1}/USDT (Perpetual)
            Order Type: MARKET SELL (Open Short)
            Quantity: {qty1:.6f} {name1}
            Leverage: 1x-3x (recommended)
            Margin Required: ~{cost1:.2f} USDT
            ```
            
            **STEP 2: BUY {name2} (SPOT)**
            ```
            Exchange: Binance/Bybit Spot
            Pair: {name2}/USDT
            Order Type: MARKET BUY
            Quantity: {qty2:.0f} {name2}
            Estimated Cost: ~{cost2:.2f} USDT
            ```
            
            **STEP 3: SET CONDITIONAL ORDERS**
            ```
            Take Profit: Z-score ≤ {zscore_exit_threshold:.1f}
            Stop Loss: Portfolio loss > {risk_per_trade}%
            Time Exit: Close after {max_trade_days} days
            ```
            """)
    else:
        st.info("⏳ No current signal. Set price alerts to be notified of entry opportunities.")
    
    # Universal warnings
    st.markdown("---")
    st.error(f"""
    ⚠️ **CRITICAL TRADING RISKS:**
    
    **Crypto-Specific Risks:**
    - **Correlation breakdown**: Crypto pairs can decorrelate rapidly
    - **Extreme volatility**: 50%+ moves possible in crypto
    - **24/7 markets**: No trading halts, gaps can occur
    - **Funding costs**: Daily funding fees on futures positions
    - **Liquidation risk**: Leverage amplifies both gains and losses
    
    **Risk Management:**
    - Max loss per trade: {trading_capital * (risk_per_trade/100):.2f} USDT
    - Never risk more than {risk_per_trade}% of total capital
    - Monitor correlation daily - exit if < 0.3
    - Use stop-loss orders on both legs
    - Start small: Test with 10% of intended size first
    
    **Technical Considerations:**
    - Use LIMIT orders for large positions to avoid slippage
    - Check funding rates before opening futures positions
    - Monitor both legs independently
    - Have exit plan ready before entering
    """)

# === ADVANCED ANALYTICS ===
with st.expander("📈 Advanced Analytics", expanded=False):
    st.header("📈 Advanced Pair Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Mean Reversion Analysis")
        
        # Half-life calculation
        spread_series = df['spread'].dropna()
        if len(spread_series) > 20:
            # Calculate half-life of mean reversion
            lagged_spread = spread_series.shift(1).dropna()
            current_spread = spread_series[1:].dropna()
            
            if len(lagged_spread) == len(current_spread):
                X_half = lagged_spread.values.reshape(-1, 1)
                y_half = current_spread.values
                half_life_model = LinearRegression().fit(X_half, y_half)
                theta = 1 - half_life_model.coef_[0]
                half_life = -np.log(2) / np.log(1 - theta) if theta > 0 else np.inf
                
                st.metric("Half-life (days)", f"{half_life:.1f}" if half_life != np.inf else "∞")
        
        # Volatility metrics
        spread_vol = df['spread'].std()
        zscore_vol = df['zscore'].std()
        
        st.metric("Spread Volatility", f"{spread_vol:.6f}")
        st.metric("Z-score Volatility", f"{zscore_vol:.3f}")
    
    with col2:
        st.subheader("📊 Trading Opportunity Frequency")
        
        # Signal frequency analysis
        long_signals = (df['zscore'] <= -zscore_entry_threshold).sum()
        short_signals = (df['zscore'] >= zscore_entry_threshold).sum()
        total_days = len(df)
        
        st.metric("Long Opportunities", f"{long_signals} ({long_signals/total_days*100:.1f}%)")
        st.metric("Short Opportunities", f"{short_signals} ({short_signals/total_days*100:.1f}%)")
        st.metric("Total Signals", f"{long_signals + short_signals}")
        
        # Optimal thresholds suggestion
        signal_freq = (long_signals + short_signals) / total_days
        if signal_freq < 0.05:
            st.warning("⚠️ Too few signals - consider lower Z-score threshold")
        elif signal_freq > 0.3:
            st.warning("⚠️ Too many signals - consider higher Z-score threshold")
        else:
            st.success("✅ Good signal frequency for pairs trading")

# === FOOTER INFO ===
st.markdown("---")
st.markdown("""
### 📚 Trading Notes:
- **Pairs trading** profits from relative price movements, not absolute direction
- **Hedge ratio (β)** determines position sizes for market-neutral exposure  
- **Z-score** measures how far spread deviates from historical mean
- **Risk management** is crucial - never risk more than you can afford to lose
- **Backtest results** don't guarantee future performance
""")

st.caption(f"""
Data: {name1} vs {name2} | Period: {periode} | Interval: {interval} | 
Current Z-score: {current_zscore:.2f} | Correlation: {processor.pearson_corr:.3f} | 
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")
