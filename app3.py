import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from constants.tickers import tickers
import warnings
warnings.filterwarnings('ignore')

# Professional trading theme CSS
st.markdown("""
<style>
    .stApp {
        background: #0a0a0a;
        color: #00ff41;
    }
    .main-header {
        background: linear-gradient(90deg, #000 0%, #111 50%, #000 100%);
        padding: 25px;
        border: 2px solid #00ff41;
        color: #00ff41;
        text-align: center;
        font-family: 'Courier New', monospace;
        margin-bottom: 30px;
        box-shadow: 0 0 20px #00ff4120;
    }
    .metric-box {
        background: #000;
        border: 1px solid #00ff41;
        padding: 20px;
        text-align: center;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 10px #00ff4120;
    }
    .signal-active {
        background: linear-gradient(135deg, #001100 0%, #002200 100%);
        border: 2px solid #00ff41;
        padding: 25px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 30px #00ff41;
        animation: pulse 2s infinite;
    }
    .signal-inactive {
        background: #111;
        border: 1px solid #333;
        padding: 25px;
        color: #666;
        font-family: 'Courier New', monospace;
    }
    .position-card {
        background: #000;
        border: 1px solid #00ff41;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .saved-pair {
        background: #111;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        font-family: 'Courier New', monospace;
        transition: all 0.3s;
    }
    .saved-pair:hover {
        border-color: #00ff41;
        background: #001100;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 30px #00ff41; }
        50% { box-shadow: 0 0 50px #00ff41; }
        100% { box-shadow: 0 0 30px #00ff41; }
    }
    .positive { color: #00ff41; }
    .negative { color: #ff4444; }
    .neutral { color: #888; }
</style>
""", unsafe_allow_html=True)

class AdvancedPairsTradingSystem:
    def __init__(self):
        self.current_analysis = {}
        self.cached_data = {}
        
    @st.cache_data(ttl=300)
    def fetch_price_data(_self, ticker, period='1y'):
        """Fetch price data with caching"""
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data.empty:
                return pd.Series(dtype=float)
            
            close_prices = data['Close'] if 'Close' in data.columns else data.iloc[:, -1]
            return close_prices.dropna()
            
        except Exception as e:
            st.error(f"Failed to load {ticker}: {str(e)}")
            return pd.Series(dtype=float)
    
    def calculate_correlation_metrics(self, price1, price2):
        """Calculate comprehensive correlation analysis"""
        if price1.empty or price2.empty or len(price1) < 30 or len(price2) < 30:
            return {'correlation': 0.0, 'suitable': False}
        
        try:
            # Align data
            price1, price2 = price1.align(price2, join='inner')
            
            if len(price1) < 30:
                return {'correlation': 0.0, 'suitable': False}
            
            # Calculate returns
            ret1 = price1.pct_change().dropna()
            ret2 = price2.pct_change().dropna()
            
            if ret1.empty or ret2.empty:
                return {'correlation': 0.0, 'suitable': False}
            
            # Overall correlation
            correlation = ret1.corr(ret2)
            if pd.isna(correlation):
                correlation = 0.0
            
            # Rolling correlation stability
            rolling_corr = ret1.rolling(30).corr(ret2).dropna()
            corr_stability = rolling_corr.std() if len(rolling_corr) > 10 else 1.0
            
            # Cointegration test (simplified)
            spread_series = price1 - price2
            mean_reversion_score = self._test_mean_reversion(spread_series)
            
            suitable = (abs(correlation) > 0.3 and 
                       corr_stability < 0.3 and 
                       mean_reversion_score > 0.4)
            
            return {
                'correlation': correlation,
                'corr_stability': corr_stability,
                'mean_reversion': mean_reversion_score,
                'suitable': suitable
            }
            
        except Exception as e:
            st.warning(f"Correlation calculation error: {e}")
            return {'correlation': 0.0, 'suitable': False}
    
    def _test_mean_reversion(self, series):
        """Simple mean reversion test"""
        if len(series) < 50:
            return 0.0
        
        try:
            mean_val = series.mean()
            crossings = ((series > mean_val) != (series.shift(1) > mean_val)).sum()
            crossing_rate = crossings / len(series)
            return min(crossing_rate * 2, 1.0)
        except:
            return 0.0
    
    def optimize_zscore_parameters(self, price1, price2):
        """Find optimal z-score parameters for maximum profitability"""
        if price1.empty or price2.empty:
            return {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}, 0.0
        
        best_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
        best_return = -999
        
        # Parameter ranges for optimization
        entry_levels = np.arange(1.2, 4.0, 0.3)
        exit_levels = np.arange(0.2, 1.2, 0.2)
        stop_levels = np.arange(2.8, 5.5, 0.4)
        
        progress_bar = st.progress(0)
        total_combinations = len(entry_levels) * len(exit_levels) * len(stop_levels)
        current_combination = 0
        
        for entry in entry_levels:
            for exit in exit_levels:
                for stop in stop_levels:
                    if stop > entry and exit < entry:  # Logical constraints
                        
                        # Test these parameters
                        signals_df = self.generate_trading_signals(price1, price2, entry, exit, stop)
                        performance = self.backtest_strategy(signals_df, price1, price2)
                        
                        # Optimization criteria: total return * win_rate * (1/max_drawdown)
                        if performance['num_trades'] > 5:  # Minimum trade requirement
                            score = (performance['total_return'] * 
                                   performance['win_rate'] / 100 * 
                                   (1 / max(performance.get('max_drawdown', 20), 5)))
                            
                            if score > best_return:
                                best_return = score
                                best_params = {'entry': entry, 'exit': exit, 'stop': stop}
                    
                    current_combination += 1
                    progress_bar.progress(current_combination / total_combinations)
        
        progress_bar.empty()
        return best_params, best_return
    
    def calculate_spread_zscore(self, price1, price2, lookback_period=80):
        """Calculate spread and z-score with robust hedge ratio"""
        if price1.empty or price2.empty:
            return pd.DataFrame(), 1.0
        
        try:
            # Align prices
            price1, price2 = price1.align(price2, join='inner')
            
            if len(price1) < lookback_period + 30:
                return pd.DataFrame(), 1.0
            
            # Dynamic hedge ratio calculation
            recent_data_points = min(lookback_period, len(price1))
            X = price2.iloc[-recent_data_points:].values.reshape(-1, 1)
            y = price1.iloc[-recent_data_points:].values
            
            # Linear regression for hedge ratio
            model = LinearRegression().fit(X, y)
            hedge_ratio = float(model.coef_[0])
            
            # Validate hedge ratio
            if abs(hedge_ratio) > 20 or abs(hedge_ratio) < 0.05:
                hedge_ratio = price1.mean() / price2.mean() if price2.mean() != 0 else 1.0
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # Dynamic z-score calculation
            zscore_window = 25
            rolling_mean = spread.rolling(window=zscore_window, min_periods=zscore_window//2).mean()
            rolling_std = spread.rolling(window=zscore_window, min_periods=zscore_window//2).std()
            
            # Handle edge cases
            rolling_std = rolling_std.fillna(spread.std())
            rolling_std = rolling_std.replace(0, spread.std())
            
            zscore = (spread - rolling_mean) / rolling_std
            
            # Create analysis dataframe
            analysis_df = pd.DataFrame({
                'price1': price1,
                'price2': price2,
                'spread': spread,
                'zscore': zscore.fillna(0),
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std
            }, index=price1.index)
            
            return analysis_df, hedge_ratio
            
        except Exception as e:
            st.error(f"Spread calculation failed: {e}")
            return pd.DataFrame(), 1.0
    
    def generate_trading_signals(self, price1, price2, entry_zscore=2.0, exit_zscore=0.5, stop_zscore=3.5):
        """Generate precise trading signals based on z-score thresholds"""
        analysis_df, hedge_ratio = self.calculate_spread_zscore(price1, price2)
        
        if analysis_df.empty:
            return pd.DataFrame()
        
        signals = []
        current_position = 0  # 0=flat, 1=long_spread, -1=short_spread
        entry_price = 0
        entry_zscore_val = 0
        
        for i, (timestamp, row) in enumerate(analysis_df.iterrows()):
            zscore = row['zscore']
            
            signal_data = {
                'timestamp': timestamp,
                'price1': row['price1'],
                'price2': row['price2'],
                'spread': row['spread'],
                'zscore': zscore,
                'position': current_position,
                'action': 'HOLD',
                'entry_price': None,
                'exit_price': None
            }
            
            # Signal generation logic
            if current_position == 0:  # No position
                if zscore <= -entry_zscore:  # Enter long spread
                    signal_data['action'] = 'ENTER_LONG_SPREAD'
                    signal_data['entry_price'] = row['spread']
                    current_position = 1
                    entry_price = row['spread']
                    entry_zscore_val = zscore
                    
                elif zscore >= entry_zscore:  # Enter short spread
                    signal_data['action'] = 'ENTER_SHORT_SPREAD'
                    signal_data['entry_price'] = row['spread']
                    current_position = -1
                    entry_price = row['spread']
                    entry_zscore_val = zscore
            
            elif current_position == 1:  # Long spread position
                if zscore >= -exit_zscore:  # Normal exit
                    signal_data['action'] = 'EXIT_LONG_SPREAD'
                    signal_data['exit_price'] = row['spread']
                    current_position = 0
                elif zscore <= -stop_zscore:  # Stop loss
                    signal_data['action'] = 'STOP_LONG_SPREAD'
                    signal_data['exit_price'] = row['spread']
                    current_position = 0
            
            elif current_position == -1:  # Short spread position
                if zscore <= exit_zscore:  # Normal exit
                    signal_data['action'] = 'EXIT_SHORT_SPREAD'
                    signal_data['exit_price'] = row['spread']
                    current_position = 0
                elif zscore >= stop_zscore:  # Stop loss
                    signal_data['action'] = 'STOP_SHORT_SPREAD'
                    signal_data['exit_price'] = row['spread']
                    current_position = 0
            
            signal_data['position'] = current_position
            signals.append(signal_data)
        
        return pd.DataFrame(signals)
    
    def backtest_strategy(self, signals_df, price1, price2):
        """Comprehensive backtesting with performance metrics"""
        if signals_df.empty:
            return self._empty_performance_metrics()
        
        trades = []
        portfolio_value = 10000  # Initial capital
        current_trade = None
        
        for _, signal in signals_df.iterrows():
            
            if signal['action'] in ['ENTER_LONG_SPREAD', 'ENTER_SHORT_SPREAD']:
                current_trade = {
                    'entry_date': signal['timestamp'],
                    'entry_spread': signal['entry_price'],
                    'entry_zscore': signal['zscore'],
                    'position_type': signal['action'],
                    'entry_price1': signal['price1'],
                    'entry_price2': signal['price2']
                }
            
            elif signal['action'] in ['EXIT_LONG_SPREAD', 'EXIT_SHORT_SPREAD', 'STOP_LONG_SPREAD', 'STOP_SHORT_SPREAD']:
                if current_trade:
                    # Calculate P&L
                    entry_spread = current_trade['entry_spread']
                    exit_spread = signal['exit_price']
                    
                    if current_trade['position_type'] == 'ENTER_LONG_SPREAD':
                        pnl = exit_spread - entry_spread
                    else:  # SHORT_SPREAD
                        pnl = entry_spread - exit_spread
                    
                    # Calculate percentage return
                    pnl_percentage = (pnl / abs(entry_spread)) * 100 if entry_spread != 0 else 0
                    
                    trade_record = {
                        'entry_date': current_trade['entry_date'],
                        'exit_date': signal['timestamp'],
                        'entry_zscore': current_trade['entry_zscore'],
                        'exit_zscore': signal['zscore'],
                        'position_type': current_trade['position_type'],
                        'entry_spread': entry_spread,
                        'exit_spread': exit_spread,
                        'pnl_absolute': pnl,
                        'pnl_percentage': pnl_percentage,
                        'exit_reason': signal['action']
                    }
                    
                    trades.append(trade_record)
                    current_trade = None
        
        return self._calculate_performance_metrics(trades)
    
    def _calculate_performance_metrics(self, trades):
        """Calculate comprehensive performance statistics"""
        if not trades:
            return self._empty_performance_metrics()
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl_percentage'] > 0]
        losing_trades = trades_df[trades_df['pnl_percentage'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_return = trades_df['pnl_percentage'].sum()
        
        avg_win = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if avg_loss != 0 else float('inf')
        
        # Risk metrics
        returns_series = trades_df['pnl_percentage']
        max_drawdown = self._calculate_max_drawdown(returns_series)
        
        sharpe_ratio = 0
        if returns_series.std() > 0:
            sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': winning_trades.loc[winning_trades['pnl_percentage'].idxmax()] if len(winning_trades) > 0 else None,
            'worst_trade': losing_trades.loc[losing_trades['pnl_percentage'].idxmin()] if len(losing_trades) > 0 else None,
            'trades_data': trades_df
        }
    
    def _calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns_series / 100).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
    
    def _empty_performance_metrics(self):
        """Return empty performance metrics"""
        return {
            'total_return': 0, 'win_rate': 0, 'num_trades': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'sharpe_ratio': 0,
            'best_trade': None, 'worst_trade': None, 'trades_data': pd.DataFrame()
        }
    
    def calculate_position_sizes(self, current_signal, capital, leverage, hedge_ratio):
        """Calculate exact position sizes for both assets"""
        if current_signal['action'] not in ['ENTER_LONG_SPREAD', 'ENTER_SHORT_SPREAD']:
            return None
        
        price1 = current_signal['price1']
        price2 = current_signal['price2']
        
        # Total available capital with leverage
        total_capital = capital * leverage
        
        # Equal allocation to both sides
        capital_per_side = total_capital / 2
        
        if current_signal['action'] == 'ENTER_LONG_SPREAD':
            # Long spread: Buy asset1, Short asset2
            asset1_quantity = capital_per_side / price1
            asset2_quantity = (capital_per_side / price2) * hedge_ratio
            
            return {
                'asset1_action': 'BUY',
                'asset1_quantity': asset1_quantity,
                'asset1_price': price1,
                'asset1_value': asset1_quantity * price1,
                'asset2_action': 'SHORT',
                'asset2_quantity': asset2_quantity,
                'asset2_price': price2,
                'asset2_value': asset2_quantity * price2,
                'total_exposure': (asset1_quantity * price1) + (asset2_quantity * price2),
                'hedge_ratio': hedge_ratio
            }
        else:  # ENTER_SHORT_SPREAD
            # Short spread: Short asset1, Buy asset2
            asset1_quantity = capital_per_side / price1
            asset2_quantity = (capital_per_side / price2) * hedge_ratio
            
            return {
                'asset1_action': 'SHORT',
                'asset1_quantity': asset1_quantity,
                'asset1_price': price1,
                'asset1_value': asset1_quantity * price1,
                'asset2_action': 'BUY',
                'asset2_quantity': asset2_quantity,
                'asset2_price': price2,
                'asset2_value': asset2_quantity * price2,
                'total_exposure': (asset1_quantity * price1) + (asset2_quantity * price2),
                'hedge_ratio': hedge_ratio
            }

# Streamlit App Configuration
st.set_page_config(
    page_title="Advanced Crypto Pairs Trading", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize system
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = AdvancedPairsTradingSystem()
    st.session_state.saved_pairs = []

system = st.session_state.trading_system

# Main header
st.markdown('''
<div class="main-header">
<h1>ADVANCED CRYPTO PAIRS TRADING SYSTEM</h1>
<p>Quantitative Analysis • Optimized Parameters • Live Execution Signals</p>
</div>
''', unsafe_allow_html=True)

# Control Panel
st.markdown("### TRADING PARAMETERS")

col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 1, 1.5])

with col1:
    asset1_name = st.selectbox("PRIMARY ASSET", list(tickers.keys()), key='primary_asset')
    asset1_ticker = tickers[asset1_name]

with col2:
    available_assets = [name for name in tickers.keys() if name != asset1_name]
    asset2_name = st.selectbox("SECONDARY ASSET", available_assets, key='secondary_asset')
    asset2_ticker = tickers[asset2_name]

with col3:
    trading_capital = st.number_input("CAPITAL ($)", min_value=1000, value=10000, step=1000)

with col4:
    leverage_factor = st.number_input("LEVERAGE", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

with col5:
    data_period = st.selectbox("ANALYSIS PERIOD", ["3mo", "6mo", "1y", "2y"], index=2)

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    analyze_button = st.button("🔍 ANALYZE PAIR", type="primary", use_container_width=True)

with col2:
    optimize_button = st.button("⚡ OPTIMIZE PARAMS", use_container_width=True)

with col3:
    save_button = st.button("💾 SAVE PAIR", use_container_width=True)

with col4:
    clear_button = st.button("🗑️ CLEAR DATA", use_container_width=True)

# Main Analysis
if analyze_button or optimize_button:
    
    with st.spinner(f"Fetching data for {asset1_name} and {asset2_name}..."):
        price1 = system.fetch_price_data(asset1_ticker, data_period)
        price2 = system.fetch_price_data(asset2_ticker, data_period)
    
    if not price1.empty and not price2.empty and len(price1) > 100 and len(price2) > 100:
        
        # Correlation analysis
        correlation_metrics = system.calculate_correlation_metrics(price1, price2)
        
        if not correlation_metrics['suitable']:
            st.warning(f"⚠️ Low correlation pair ({correlation_metrics['correlation']:.3f}). Results may not be reliable.")
        
        # Parameter optimization
        if optimize_button:
            with st.spinner("🔬 Optimizing parameters for maximum profit..."):
                optimal_params, optimization_score = system.optimize_zscore_parameters(price1, price2)
        else:
            optimal_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
            optimization_score = 0
        
        # Generate signals and backtest
        signals_df = system.generate_trading_signals(price1, price2, **optimal_params)
        performance_metrics = system.backtest_strategy(signals_df, price1, price2)
        
        # Calculate current market state
        analysis_df, hedge_ratio = system.calculate_spread_zscore(price1, price2)
        current_zscore = analysis_df['zscore'].iloc[-1] if not analysis_df.empty else 0
        current_spread = analysis_df['spread'].iloc[-1] if not analysis_df.empty else 0
        
        # Store analysis results
        system.current_analysis = {
            'asset1_name': asset1_name, 'asset2_name': asset2_name,
            'asset1_ticker': asset1_ticker, 'asset2_ticker': asset2_ticker,
            'price1': price1, 'price2': price2,
            'correlation_metrics': correlation_metrics,
            'optimal_params': optimal_params,
            'signals_df': signals_df,
            'performance': performance_metrics,
            'analysis_df': analysis_df,
            'hedge_ratio': hedge_ratio,
            'current_zscore': current_zscore,
            'current_spread': current_spread
        }
        
        # Display Performance Metrics
        st.markdown("### PERFORMANCE METRICS")
        
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        with metric_col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            color_class = "positive" if correlation_metrics['correlation'] > 0.5 else "negative" if correlation_metrics['correlation'] < -0.5 else "neutral"
            st.markdown(f'<div class="metric-value {color_class}">{correlation_metrics["correlation"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">CORRELATION</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            zscore_color = "positive" if abs(current_zscore) > optimal_params['entry'] else "neutral"
            st.markdown(f'<div class="metric-value {zscore_color}">{current_zscore:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">CURRENT Z-SCORE</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            return_color = "positive" if performance_metrics['total_return'] > 0 else "negative"
            st.markdown(f'<div class="metric-value {return_color}">{performance_metrics["total_return"]:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">TOTAL RETURN</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            winrate_color = "positive" if performance_metrics['win_rate'] > 60 else "neutral"
            st.markdown(f'<div class="metric-value {winrate_color}">{performance_metrics["win_rate"]:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">WIN RATE</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col5:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            sharpe_color = "positive" if performance_metrics['sharpe_ratio'] > 1.0 else "neutral"
            st.markdown(f'<div class="metric-value {sharpe_color}">{performance_metrics["sharpe_ratio"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">SHARPE RATIO</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Live Trading Signal
        st.markdown("### LIVE TRADING SIGNAL")
        
        latest_signal = signals_df.iloc[-1] if not signals_df.empty else None
        
        if (latest_signal and 
            latest_signal['action'] in ['ENTER_LONG_SPREAD', 'ENTER_SHORT_SPREAD'] and
            abs(current_zscore) > optimal_params['entry']):
            
            st.markdown('<div class="signal-active">', unsafe_allow_html=True)
            st.markdown(f"## 🚨 ACTIVE SIGNAL: {latest_signal['action']}")
            
            # Calculate position sizes
            position_details = system.calculate_position_sizes(
                latest_signal, trading_capital, leverage_factor, hedge_ratio
            )
            
            if position_details:
                signal_col1, signal_col2 = st.columns(2)
                
                with signal_col1:
                    st.markdown(f"""
                    ### {asset1_name} ({asset1_ticker})
                    **ACTION:** {position_details['asset1_action']}  
                    **QUANTITY:** {position_details['asset1_quantity']:.6f}  
                    **PRICE:** ${position_details['asset1_price']:.4f}  
                    **VALUE:** ${position_details['asset1_value']:.2f}  
                    """)
                
                with signal_col2:
                    st.markdown(f"""
                    ### {asset2_name} ({asset2_ticker})
                    **ACTION:** {position_details['asset
