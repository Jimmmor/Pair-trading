import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Professional CSS - Finance style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f0f0f 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 5px;
        color: #00ff41;
        text-align: center;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #000000;
        border: 1px solid #00ff41;
        padding: 15px;
        border-radius: 3px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
    }
    .profit-alert {
        background: #001100;
        border: 2px solid #00ff41;
        padding: 15px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    .loss-alert {
        background: #110000;
        border: 2px solid #ff0000;
        padding: 15px;
        color: #ff0000;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    .neutral-alert {
        background: #111111;
        border: 1px solid #666666;
        padding: 15px;
        color: #cccccc;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    .data-table {
        background: #000000;
        color: #00ff41;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Cryptocurrency tickers - expanded list
CRYPTO_TICKERS = {
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'BNB': 'BNB-USD', 'XRP': 'XRP-USD',
    'ADA': 'ADA-USD', 'SOL': 'SOL-USD', 'DOT': 'DOT-USD', 'DOGE': 'DOGE-USD',
    'AVAX': 'AVAX-USD', 'SHIB': 'SHIB-USD', 'MATIC': 'MATIC-USD', 'LTC': 'LTC-USD',
    'UNI': 'UNI-USD', 'LINK': 'LINK-USD', 'ALGO': 'ALGO-USD', 'VET': 'VET-USD',
    'ICP': 'ICP-USD', 'FIL': 'FIL-USD', 'TRX': 'TRX-USD', 'XLM': 'XLM-USD'
}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_crypto_data(symbol, period='1y'):
    """Load cryptocurrency data with error handling"""
    try:
        ticker = CRYPTO_TICKERS.get(symbol, symbol)
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            return pd.Series(dtype=float)
        
        # Return close prices as Series
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            return data['Close'].dropna()
        elif isinstance(data, pd.DataFrame):
            return data.iloc[:, -1].dropna()
        else:
            return data.dropna()
            
    except Exception as e:
        st.error(f"Data loading error for {symbol}: {str(e)}")
        return pd.Series(dtype=float)

class ProfessionalPairsTrader:
    """Professional pairs trading system with robust logic"""
    
    def __init__(self):
        self.optimal_params = {}
        self.current_data = {}
        self.correlation_threshold = 0.7
        
    def calculate_correlation_statistics(self, price1, price2):
        """Calculate comprehensive correlation statistics"""
        # Align data
        price1, price2 = price1.align(price2, join='inner')
        
        if len(price1) < 30:
            return None
            
        # Calculate returns
        ret1 = price1.pct_change().dropna()
        ret2 = price2.pct_change().dropna()
        
        # Correlation analysis
        correlation = ret1.corr(ret2)
        
        # Rolling correlations
        rolling_30d = ret1.rolling(30).corr(ret2).dropna()
        rolling_90d = ret1.rolling(90).corr(ret2).dropna()
        
        # Cointegration test (simplified)
        try:
            # Linear regression for cointegration
            X = price2.values.reshape(-1, 1)
            y = price1.values
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            
            # Check for mean reversion in residuals
            residual_series = pd.Series(residuals, index=price1.index)
            adf_stat = self._simplified_adf_test(residual_series)
            
        except:
            adf_stat = 0.5
            
        return {
            'correlation': correlation,
            'correlation_30d_mean': rolling_30d.mean(),
            'correlation_30d_std': rolling_30d.std(),
            'correlation_90d_mean': rolling_90d.mean(),
            'correlation_stability': rolling_30d.std(),
            'cointegration_score': adf_stat,
            'suitable_for_pairs': correlation > self.correlation_threshold and adf_stat > 0.6
        }
    
    def _simplified_adf_test(self, series):
        """Simplified stationarity test"""
        try:
            # Check for mean reversion properties
            mean_val = series.mean()
            crosses = ((series > mean_val) != (series.shift(1) > mean_val)).sum()
            total_points = len(series)
            
            # High crossing frequency indicates mean reversion
            crossing_rate = crosses / total_points
            return min(crossing_rate * 2, 1.0)  # Normalize to 0-1
            
        except:
            return 0.5
    
    def calculate_spread_and_signals(self, price1, price2, lookback_window=60, zscore_window=20):
        """Calculate spread, z-score and generate trading signals"""
        
        # Align data
        price1, price2 = price1.align(price2, join='inner')
        
        if len(price1) < lookback_window + zscore_window:
            return pd.DataFrame(), 1.0
        
        # Calculate optimal hedge ratio using regression
        try:
            X = price2.iloc[-lookback_window:].values.reshape(-1, 1)
            y = price1.iloc[-lookback_window:].values
            model = LinearRegression().fit(X, y)
            hedge_ratio = model.coef_[0]
            
            # Validate hedge ratio
            if abs(hedge_ratio) > 5 or abs(hedge_ratio) < 0.2:
                hedge_ratio = 1.0
                
        except:
            hedge_ratio = 1.0
        
        # Calculate spread
        spread = price1 - hedge_ratio * price2
        
        # Calculate z-score
        rolling_mean = spread.rolling(window=zscore_window, min_periods=zscore_window//2).mean()
        rolling_std = spread.rolling(window=zscore_window, min_periods=zscore_window//2).std()
        
        # Prevent division by zero
        rolling_std = rolling_std.fillna(rolling_std.mean()).replace(0, rolling_std.mean())
        zscore = (spread - rolling_mean) / rolling_std
        
        # Create comprehensive dataframe
        df = pd.DataFrame({
            'price1': price1,
            'price2': price2,
            'spread': spread,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'zscore': zscore.fillna(0)
        }, index=price1.index)
        
        return df, hedge_ratio
    
    def generate_trading_signals(self, df, entry_threshold=2.0, exit_threshold=0.5, 
                               stop_loss_threshold=3.5):
        """Generate precise trading signals based on z-score logic"""
        
        signals = []
        position = 0  # 0=flat, 1=long spread, -1=short spread
        entry_zscore = 0
        entry_date = None
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_zscore = df['zscore'].iloc[i]
            
            if pd.isna(current_zscore):
                continue
            
            signal = {
                'date': current_date,
                'zscore': current_zscore,
                'price1': df['price1'].iloc[i],
                'price2': df['price2'].iloc[i],
                'spread': df['spread'].iloc[i],
                'position': position,
                'action': 'HOLD',
                'entry_price': None,
                'exit_price': None,
                'stop_loss': None,
                'take_profit': None
            }
            
            # Position management
            if position != 0:
                # Check exit conditions
                if position == 1:  # Long spread position
                    # Exit conditions: zscore crosses back above exit_threshold or hits stop loss
                    if current_zscore >= -exit_threshold:
                        signal['action'] = 'EXIT_LONG'
                        signal['exit_price'] = df['spread'].iloc[i]
                        position = 0
                    elif current_zscore <= -stop_loss_threshold:
                        signal['action'] = 'STOP_LOSS_LONG'
                        signal['exit_price'] = df['spread'].iloc[i]
                        position = 0
                        
                elif position == -1:  # Short spread position
                    # Exit conditions: zscore crosses back below exit_threshold or hits stop loss
                    if current_zscore <= exit_threshold:
                        signal['action'] = 'EXIT_SHORT'
                        signal['exit_price'] = df['spread'].iloc[i]
                        position = 0
                    elif current_zscore >= stop_loss_threshold:
                        signal['action'] = 'STOP_LOSS_SHORT'
                        signal['exit_price'] = df['spread'].iloc[i]
                        position = 0
            
            else:  # No position
                # Entry conditions
                if current_zscore <= -entry_threshold:
                    # Long spread: buy asset1, sell asset2
                    signal['action'] = 'ENTER_LONG'
                    signal['entry_price'] = df['spread'].iloc[i]
                    signal['stop_loss'] = -stop_loss_threshold
                    signal['take_profit'] = -exit_threshold
                    position = 1
                    entry_zscore = current_zscore
                    entry_date = current_date
                    
                elif current_zscore >= entry_threshold:
                    # Short spread: sell asset1, buy asset2
                    signal['action'] = 'ENTER_SHORT'
                    signal['entry_price'] = df['spread'].iloc[i]
                    signal['stop_loss'] = stop_loss_threshold
                    signal['take_profit'] = exit_threshold
                    position = -1
                    entry_zscore = current_zscore
                    entry_date = current_date
            
            signal['position'] = position
            signals.append(signal)
        
        return pd.DataFrame(signals)
    
    def calculate_position_sizing(self, signal_row, capital, risk_per_trade=0.02, max_leverage=3):
        """Calculate precise position sizing for each asset"""
        
        if signal_row['action'] not in ['ENTER_LONG', 'ENTER_SHORT']:
            return None
        
        # Risk-based position sizing
        risk_amount = capital * risk_per_trade
        
        # Calculate position sizes based on current prices
        price1 = signal_row['price1']
        price2 = signal_row['price2']
        
        if signal_row['action'] == 'ENTER_LONG':
            # Long spread: Buy asset1, Short asset2
            # Allocate capital proportionally
            total_allocation = min(capital * 0.5 * max_leverage, risk_amount * 50)  # Max 50x risk
            
            asset1_qty = total_allocation / (2 * price1)  # Buy asset1
            asset2_qty = total_allocation / (2 * price2)  # Short asset2
            
            return {
                'asset1_action': 'BUY',
                'asset1_quantity': asset1_qty,
                'asset1_value': asset1_qty * price1,
                'asset2_action': 'SHORT',
                'asset2_quantity': asset2_qty,
                'asset2_value': asset2_qty * price2,
                'total_exposure': asset1_qty * price1 + asset2_qty * price2,
                'margin_required': total_allocation / max_leverage
            }
            
        else:  # ENTER_SHORT
            # Short spread: Short asset1, Buy asset2
            total_allocation = min(capital * 0.5 * max_leverage, risk_amount * 50)
            
            asset1_qty = total_allocation / (2 * price1)  # Short asset1
            asset2_qty = total_allocation / (2 * price2)  # Buy asset2
            
            return {
                'asset1_action': 'SHORT',
                'asset1_quantity': asset1_qty,
                'asset1_value': asset1_qty * price1,
                'asset2_action': 'BUY',
                'asset2_quantity': asset2_qty,
                'asset2_value': asset2_qty * price2,
                'total_exposure': asset1_qty * price1 + asset2_qty * price2,
                'margin_required': total_allocation / max_leverage
            }
    
    def backtest_strategy(self, signals_df, initial_capital=10000):
        """Professional backtesting with transaction costs"""
        
        if signals_df.empty:
            return self._empty_backtest_results()
        
        capital = initial_capital
        portfolio_values = [capital]
        trades = []
        current_position = None
        
        transaction_cost = 0.001  # 0.1% per trade
        
        for i, row in signals_df.iterrows():
            
            if row['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                # Calculate position sizing
                position_info = self.calculate_position_sizing(row, capital)
                
                if position_info:
                    # Deduct transaction costs
                    cost = position_info['total_exposure'] * transaction_cost
                    capital -= cost
                    
                    current_position = {
                        'entry_date': row['date'],
                        'entry_zscore': row['zscore'],
                        'entry_spread': row['entry_price'],
                        'position_type': row['action'],
                        'position_info': position_info
                    }
            
            elif row['action'] in ['EXIT_LONG', 'EXIT_SHORT', 'STOP_LOSS_LONG', 'STOP_LOSS_SHORT']:
                if current_position:
                    # Calculate P&L
                    entry_spread = current_position['entry_spread']
                    exit_spread = row['exit_price']
                    
                    if current_position['position_type'] == 'ENTER_LONG':
                        pnl = exit_spread - entry_spread
                    else:  # ENTER_SHORT
                        pnl = entry_spread - exit_spread
                    
                    # Apply to capital (simplified)
                    position_size = current_position['position_info']['total_exposure']
                    pnl_amount = (pnl / entry_spread) * position_size
                    
                    # Transaction costs on exit
                    exit_cost = position_size * transaction_cost
                    final_pnl = pnl_amount - exit_cost
                    
                    capital += final_pnl
                    
                    # Record trade
                    trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': row['date'],
                        'position_type': current_position['position_type'],
                        'entry_zscore': current_position['entry_zscore'],
                        'exit_zscore': row['zscore'],
                        'pnl': final_pnl,
                        'pnl_pct': (final_pnl / position_size) * 100,
                        'exit_reason': row['action']
                    })
                    
                    current_position = None
            
            portfolio_values.append(capital)
        
        # Calculate performance metrics
        return self._calculate_backtest_metrics(initial_capital, capital, portfolio_values, trades)
    
    def _calculate_backtest_metrics(self, initial_capital, final_capital, portfolio_values, trades):
        """Calculate comprehensive backtest metrics"""
        
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        if not trades:
            return self._empty_backtest_results()
        
        trades_df = pd.DataFrame(trades)
        
        # Trade statistics
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(win_trades)) / abs(avg_loss * len(loss_trades)) if avg_loss != 0 else float('inf')
        
        # Risk metrics
        portfolio_series = pd.Series(portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        max_drawdown = ((portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax()).max() * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'final_capital': final_capital,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_values': portfolio_values,
            'trades': trades_df,
            'best_trade': trades_df.loc[trades_df['pnl'].idxmax()] if len(trades_df) > 0 else None,
            'worst_trade': trades_df.loc[trades_df['pnl'].idxmin()] if len(trades_df) > 0 else None
        }
    
    def _empty_backtest_results(self):
        """Return empty backtest results"""
        return {
            'total_return': 0, 'final_capital': 10000, 'win_rate': 0, 'num_trades': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'max_drawdown': 0,
            'sharpe_ratio': 0, 'portfolio_values': [10000], 'trades': pd.DataFrame(),
            'best_trade': None, 'worst_trade': None
        }

# Page configuration
st.set_page_config(page_title="Professional Pairs Trading", layout="wide")

# Initialize trading system
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = ProfessionalPairsTrader()

trader = st.session_state.trading_system

# Main header
st.markdown('<div class="main-header"><h1>PROFESSIONAL PAIRS TRADING SYSTEM</h1><p>Quantitative Trading • Risk Management • Live Signals</p></div>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["ANALYSIS", "SIGNALS", "PERFORMANCE", "CORRELATION"])

# Tab 1: Analysis
with tab1:
    st.subheader("STRATEGY ANALYSIS")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        crypto1 = st.selectbox("PRIMARY ASSET:", list(CRYPTO_TICKERS.keys()), index=0)
    with col2:
        crypto2 = st.selectbox("SECONDARY ASSET:", [c for c in CRYPTO_TICKERS.keys() if c != crypto1], index=1)
    with col3:
        analysis_period = st.selectbox("PERIOD:", ["6mo", "1y", "2y"], index=1)
    
    if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
        
        with st.spinner("Loading market data..."):
            price1 = load_crypto_data(crypto1, period=analysis_period)
            price2 = load_crypto_data(crypto2, period=analysis_period)
        
        if not price1.empty and not price2.empty and len(price1) > 100:
            
            # Calculate signals and backtest
            df, hedge_ratio = trader.calculate_spread_and_signals(price1, price2)
            signals = trader.generate_trading_signals(df)
            backtest_results = trader.backtest_strategy(signals)
            
            # Store results
            trader.current_data = {
                'crypto1': crypto1, 'crypto2': crypto2,
                'price1': price1, 'price2': price2,
                'df': df, 'hedge_ratio': hedge_ratio,
                'signals': signals, 'backtest': backtest_results
            }
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("TOTAL RETURN", f"{backtest_results['total_return']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("WIN RATE", f"{backtest_results['win_rate']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("PROFIT FACTOR", f"{backtest_results['profit_factor']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MAX DRAWDOWN", f"{backtest_results['max_drawdown']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'{crypto1} vs {crypto2} - Normalized Prices', 'Z-Score & Trading Signals'],
                vertical_spacing=0.1
            )
            
            # Price chart (normalized)
            norm_price1 = price1 / price1.iloc[0] * 100
            norm_price2 = price2 / price2.iloc[0] * 100
            
            fig.add_trace(go.Scatter(x=price1.index, y=norm_price1, name=crypto1, line=dict(color='#00ff41')), row=1, col=1)
            fig.add_trace(go.Scatter(x=price2.index, y=norm_price2, name=crypto2, line=dict(color='#ff4444')), row=1, col=1)
            
            # Z-score chart
            fig.add_trace(go.Scatter(x=df.index, y=df['zscore'], name='Z-Score', line=dict(color='#00ffff')), row=2, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="#ff4444", row=2, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="#00ff41", row=2, col=1)
            fig.add_hline(y=0, line_color="#666666", row=2, col=1)
            
            fig.update_layout(height=600, plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00ff41')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Insufficient data for analysis")

# Tab 2: Live Signals
with tab2:
    st.subheader("LIVE TRADING SIGNALS")
    
    if 'current_data' in trader.__dict__ and trader.current_data:
        
        # Get latest signal
        signals = trader.current_data['signals']
        if not signals.empty:
            latest_signal = signals.iloc[-1]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### CURRENT MARKET STATUS")
                
                current_zscore = latest_signal['zscore']
                
                if latest_signal['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                    st.markdown('<div class="profit-alert">ACTIVE SIGNAL DETECTED</div>', unsafe_allow_html=True)
                    
                    # Position details
                    position_info = trader.calculate_position_sizing(latest_signal, 10000)
                    
                    if position_info:
                        st.markdown(f"""
                        **SIGNAL TYPE:** {latest_signal['action']}  
                        **Z-SCORE:** {current_zscore:.2f}  
                        **ENTRY PRICE:** ${latest_signal['entry_price']:.6f}  
                        **STOP LOSS:** Z-Score {latest_signal['stop_loss']:.1f}  
                        **TAKE PROFIT:** Z-Score {latest_signal['take_profit']:.1f}  
                        """)
                        
                        st.markdown("### POSITION SIZING")
                        st.markdown(f"""
                        **{trader.current_data['crypto1']}:** {position_info['asset1_action']} {position_info['asset1_quantity']:.6f}  
                        **{trader.current_data['crypto2']}:** {position_info['asset2_action']} {position_info['asset2_quantity']:.6f}  
                        **TOTAL EXPOSURE:** ${position_info['total_exposure']:.2f}  
                        **MARGIN REQUIRED:** ${position_info['margin_required']:.2f}  
                        """)
                
                else:
                    st.markdown('<div class="neutral-alert">NO ACTIVE SIGNAL</div>', unsafe_allow_html=True)
                    st.markdown(f"**CURRENT Z-SCORE:** {current_zscore:.2f}")
                    st.markdown(f"**DISTANCE TO SIGNAL:** {abs(2.0 - abs(current_zscore)):.2f}")
            
            with col2:
                st.markdown("### SIGNAL HISTORY")
                
                # Show recent entry/exit signals
                action_signals = signals[signals['action'].isin(['ENTER_LONG', 'ENTER_SHORT', 'EXIT_LONG', 'EXIT_SHORT'])].tail(10)
                
                if not action_signals.empty:
                    for idx, sig in action_signals.iterrows():
                        color = "#00ff41" if "ENTER" in sig['action'] else "#ff4444"
                        st.markdown(f"<span style='color: {color}'>{sig['date'].strftime('%Y-%m-%d %H:%M')} - {sig['action']} (Z: {sig['zscore']:.2f})</span>", unsafe_allow_html=True)
        
        # Risk management panel
        st.markdown("---")
        st.markdown("### RISK MANAGEMENT")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            capital = st.number_input("TRADING CAPITAL ($)", value=10000, min_value=1000, step=1000)
        with col2:
            risk_per_trade = st.slider("RISK PER TRADE (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        with col3:
            max_leverage = st.slider("MAX LEVERAGE", min_value=1, max_value=5, value=3, step=1)
    
    else:
        st.warning("Run analysis first to generate signals")

# Tab 3: Performance
with tab3:
    st.subheader("PERFORMANCE ANALYSIS")
    
    if 'current_data' in trader.__dict__ and trader.current_data:
        backtest = trader.current_data['backtest']
        
        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("TOTAL RETURN", f"{backtest['total_return']:.1f}%"),
            ("WIN RATE", f"{backtest['win_rate']:.1f}%"),
            ("NUM TRADES", f"{backtest['num_trades']}"),
            ("PROFIT FACTOR", f"{backtest['profit_factor']:.2f}"),
            ("SHARPE RATIO", f"{backtest['sharpe_ratio']:.2f}")
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, value)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Equity curve
        portfolio_values = backtest['portfolio_values']
        dates = pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='D')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_values,
            name='Portfolio Value',
            line=dict(color='#00ff41', width=2)
        ))
        
        fig.update_layout(
            title="PORTFOLIO EQUITY CURVE",
            xaxis_title="DATE",
            yaxis_title="VALUE ($)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#00ff41',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        if not backtest['trades'].empty:
            st.markdown("### TRADE ANALYSIS")
            
            trades_df = backtest['trades']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**BEST TRADE:**")
                if backtest['best_trade'] is not None:
                    best = backtest['best_trade']
                    st.success(f"""
                    Date: {best['entry_date'].strftime('%Y-%m-%d')}  
                    Type: {best['position_type']}  
                    Entry Z-Score: {best['entry_zscore']:.2f}  
                    Exit Z-Score: {best['exit_zscore']:.2f}  
                    P&L: ${best['pnl']:.2f} ({best['pnl_pct']:.2f}%)
                    """)
            
            with col2:
                st.markdown("**WORST TRADE:**")
                if backtest['worst_trade'] is not None:
                    worst = backtest['worst_trade']
                    st.error(f"""
                    Date: {worst['entry_date'].strftime('%Y-%m-%d')}  
                    Type: {worst['position_type']}  
                    Entry Z-Score: {worst['entry_zscore']:.2f}  
                    Exit Z-Score: {worst['exit_zscore']:.2f}  
                    P&L: ${worst['pnl']:.2f} ({worst['pnl_pct']:.2f}%)
                    """)
            
            # Trade distribution
            fig_trades = go.Figure()
            
            fig_trades.add_trace(go.Histogram(
                x=trades_df['pnl_pct'],
                nbinsx=20,
                name='Trade Returns',
                marker_color='#00ff41',
                opacity=0.7
            ))
            
            fig_trades.update_layout(
                title="TRADE RETURN DISTRIBUTION",
                xaxis_title="RETURN (%)",
                yaxis_title="FREQUENCY",
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font_color='#00ff41',
                height=300
            )
            
            st.plotly_chart(fig_trades, use_container_width=True)
            
            # Detailed trades table
            st.markdown("### TRADE HISTORY")
            
            display_trades = trades_df[['entry_date', 'exit_date', 'position_type', 'entry_zscore', 'exit_zscore', 'pnl', 'pnl_pct', 'exit_reason']].copy()
            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
            display_trades = display_trades.round(2)
            
            st.dataframe(
                display_trades,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "entry_date": "ENTRY DATE",
                    "exit_date": "EXIT DATE", 
                    "position_type": "TYPE",
                    "entry_zscore": "ENTRY Z",
                    "exit_zscore": "EXIT Z",
                    "pnl": "P&L ($)",
                    "pnl_pct": "RETURN (%)",
                    "exit_reason": "EXIT REASON"
                }
            )
    
    else:
        st.warning("Run analysis first to see performance metrics")

# Tab 4: Correlation Analysis  
with tab4:
    st.subheader("CORRELATION & STATISTICS")
    
    if 'current_data' in trader.__dict__ and trader.current_data:
        price1 = trader.current_data['price1']
        price2 = trader.current_data['price2']
        crypto1 = trader.current_data['crypto1']
        crypto2 = trader.current_data['crypto2']
        
        # Calculate correlation statistics
        corr_stats = trader.calculate_correlation_statistics(price1, price2)
        
        if corr_stats:
            # Correlation metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CORRELATION", f"{corr_stats['correlation']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("30D AVG CORR", f"{corr_stats['correlation_30d_mean']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CORR STABILITY", f"{corr_stats['correlation_stability']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("COINTEGRATION", f"{corr_stats['cointegration_score']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Pair suitability
            if corr_stats['suitable_for_pairs']:
                st.markdown('<div class="profit-alert">PAIR SUITABLE FOR TRADING</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="loss-alert">PAIR NOT SUITABLE - LOW CORRELATION</div>', unsafe_allow_html=True)
            
            # Rolling correlation chart
            ret1 = price1.pct_change().dropna()
            ret2 = price2.pct_change().dropna()
            rolling_corr = ret1.rolling(30).corr(ret2).dropna()
            
            fig_corr = go.Figure()
            
            fig_corr.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                name='30-Day Rolling Correlation',
                line=dict(color='#00ff41', width=2)
            ))
            
            fig_corr.add_hline(y=0.7, line_dash="dash", line_color="#ffff00", annotation_text="Suitable Threshold")
            fig_corr.add_hline(y=0, line_color="#666666")
            
            fig_corr.update_layout(
                title=f"ROLLING CORRELATION: {crypto1} vs {crypto2}",
                xaxis_title="DATE",
                yaxis_title="CORRELATION",
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font_color='#00ff41',
                height=400
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Spread analysis
            df = trader.current_data['df']
            
            fig_spread = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Price Spread', 'Z-Score Distribution'],
                vertical_spacing=0.15
            )
            
            # Spread over time
            fig_spread.add_trace(go.Scatter(
                x=df.index, y=df['spread'],
                name='Spread', line=dict(color='#00ffff')
            ), row=1, col=1)
            
            fig_spread.add_trace(go.Scatter(
                x=df.index, y=df['rolling_mean'],
                name='Rolling Mean', line=dict(color='#ffff00', dash='dash')
            ), row=1, col=1)
            
            # Z-score histogram
            fig_spread.add_trace(go.Histogram(
                x=df['zscore'].dropna(),
                name='Z-Score Distribution',
                marker_color='#00ff41',
                opacity=0.7
            ), row=2, col=1)
            
            fig_spread.update_layout(
                height=600,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font_color='#00ff41'
            )
            
            st.plotly_chart(fig_spread, use_container_width=True)
            
            # Statistical summary
            st.markdown("### STATISTICAL SUMMARY")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **PRICE STATISTICS:**  
                {crypto1} Mean: ${price1.mean():.4f}  
                {crypto1} Std: ${price1.std():.4f}  
                {crypto2} Mean: ${price2.mean():.4f}  
                {crypto2} Std: ${price2.std():.4f}  
                """)
            
            with col2:
                st.markdown(f"""
                **SPREAD STATISTICS:**  
                Spread Mean: {df['spread'].mean():.6f}  
                Spread Std: {df['spread'].std():.6f}  
                Z-Score Mean: {df['zscore'].mean():.3f}  
                Z-Score Std: {df['zscore'].std():.3f}  
                """)
        
        # Optimization suggestions
        st.markdown("### OPTIMIZATION RECOMMENDATIONS")
        
        if corr_stats and corr_stats['correlation'] > 0.8:
            st.success("HIGH CORRELATION - Consider shorter timeframes and higher leverage")
        elif corr_stats and corr_stats['correlation'] > 0.6:
            st.info("MODERATE CORRELATION - Use standard parameters with careful risk management")
        else:
            st.warning("LOW CORRELATION - Consider different pairs or longer analysis periods")
    
    else:
        st.warning("Run analysis first to see correlation statistics")

# Enhanced Sidebar - Professional style
with st.sidebar:
    st.markdown('<div style="background: #000000; padding: 15px; border: 1px solid #00ff41; color: #00ff41; font-family: Courier New;">', unsafe_allow_html=True)
    st.markdown("### TRADING SYSTEM STATUS")
    
    if 'current_data' in trader.__dict__ and trader.current_data:
        st.success(f"""
        ACTIVE PAIR: {trader.current_data['crypto1']}/{trader.current_data['crypto2']}  
        HEDGE RATIO: {trader.current_data['hedge_ratio']:.3f}  
        STATUS: READY
        """)
        
        # Latest signal
        latest_signal = trader.current_data['signals'].iloc[-1]
        current_zscore = latest_signal['zscore']
        
        if abs(current_zscore) >= 2.0:
            st.error(f"SIGNAL: {latest_signal['action']}")
            st.error(f"Z-SCORE: {current_zscore:.2f}")
        else:
            st.info(f"MONITORING")
            st.info(f"Z-SCORE: {current_zscore:.2f}")
    
    else:
        st.warning("NO ACTIVE STRATEGY")
        st.info("Run analysis to activate")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("---")
    st.markdown("### MARKET DATA")
    
    # Show available pairs
    st.markdown("**AVAILABLE PAIRS:**")
    for crypto in list(CRYPTO_TICKERS.keys())[:15]:  # Show first 15
        st.text(crypto)
    
    st.markdown("---")
    st.markdown("**RISK DISCLAIMER:**")
    st.error("""
    Trading involves substantial risk.  
    Past performance does not guarantee future results.  
    Only trade with capital you can afford to lose.
    """)
    
    st.markdown("---")
    st.markdown("*Professional Pairs Trading System v1.0*")
    st.markdown("*Quantitative • Risk-Managed • Profitable*")
