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
        """Fetch price data with caching and better error handling"""
        try:
            # Download data with specific parameters to avoid issues
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True, prepost=True, threads=True)
            
            if data is None or data.empty:
                st.error(f"No data returned for {ticker}")
                return pd.Series(dtype=float)
            
            # Handle different column structures from yfinance
            close_prices = None
            
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns (when downloading multiple tickers or certain cases)
                try:
                    # Try to get Close column from level 0
                    if ('Close', ticker) in data.columns:
                        close_prices = data[('Close', ticker)]
                    elif 'Close' in [col[0] for col in data.columns]:
                        # Find Close in level 0
                        close_col = [col for col in data.columns if col[0] == 'Close'][0]
                        close_prices = data[close_col]
                    else:
                        # Fallback to last column
                        close_prices = data.iloc[:, -1]
                except:
                    # If all else fails, try flattening the MultiIndex
                    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
                    if 'Close' in data.columns:
                        close_prices = data['Close']
                    else:
                        close_prices = data.iloc[:, -1]
            else:
                # Single-level columns (normal case)
                if 'Close' in data.columns:
                    close_prices = data['Close']
                elif 'Adj Close' in data.columns:
                    close_prices = data['Adj Close']
                else:
                    # Fallback to the last column (usually Close)
                    close_prices = data.iloc[:, -1]
            
            # Ensure we have a Series, not DataFrame
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
            
            # Clean the data
            close_prices = close_prices.dropna()
            
            if len(close_prices) < 30:
                st.warning(f"Insufficient data for {ticker}: only {len(close_prices)} data points")
                return pd.Series(dtype=float)
            
            st.success(f"✅ Loaded {ticker}: {len(close_prices)} data points")
            return close_prices
            
        except Exception as e:
            st.error(f"Failed to load {ticker}: {str(e)}")
            # Try alternative approach with Ticker object
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty and 'Close' in hist.columns:
                    close_prices = hist['Close'].dropna()
                    if len(close_prices) >= 30:
                        st.success(f"✅ Loaded {ticker} (alternative method): {len(close_prices)} data points")
                        return close_prices
            except:
                pass
            
            return pd.Series(dtype=float)
    
    def calculate_correlation_metrics(self, price1, price2):
        """Calculate comprehensive correlation analysis with better error handling"""
        # Input validation
        if price1.empty or price2.empty:
            return {'correlation': 0.0, 'suitable': False, 'error': 'Empty price data'}
        
        if len(price1) < 30 or len(price2) < 30:
            return {'correlation': 0.0, 'suitable': False, 'error': 'Insufficient data points'}
        
        try:
            # Align data properly
            price1, price2 = price1.align(price2, join='inner')
            
            if len(price1) < 30:
                return {'correlation': 0.0, 'suitable': False, 'error': 'Insufficient aligned data'}
            
            # Calculate returns with proper handling
            ret1 = price1.pct_change().dropna()
            ret2 = price2.pct_change().dropna()
            
            if ret1.empty or ret2.empty or len(ret1) < 20 or len(ret2) < 20:
                return {'correlation': 0.0, 'suitable': False, 'error': 'Insufficient return data'}
            
            # Align returns
            ret1, ret2 = ret1.align(ret2, join='inner')
            
            if len(ret1) < 20:
                return {'correlation': 0.0, 'suitable': False, 'error': 'Insufficient aligned returns'}
            
            # Calculate correlation
            correlation = ret1.corr(ret2)
            if pd.isna(correlation):
                correlation = 0.0
            
            # Rolling correlation stability
            min_window = min(30, len(ret1) // 3)
            if len(ret1) >= min_window * 2:
                rolling_corr = ret1.rolling(min_window).corr(ret2).dropna()
                corr_stability = rolling_corr.std() if len(rolling_corr) > 5 else 1.0
            else:
                corr_stability = 1.0
            
            # Mean reversion test
            spread_series = price1 - price2
            mean_reversion_score = self._test_mean_reversion(spread_series)
            
            # Adjusted suitability criteria
            suitable = (abs(correlation) > 0.2 and  # Lowered threshold
                       corr_stability < 0.5 and      # More lenient
                       mean_reversion_score > 0.2)    # Lower threshold
            
            return {
                'correlation': correlation,
                'corr_stability': corr_stability,
                'mean_reversion': mean_reversion_score,
                'suitable': suitable,
                'data_points': len(ret1)
            }
            
        except Exception as e:
            st.error(f"Correlation calculation error: {e}")
            return {'correlation': 0.0, 'suitable': False, 'error': str(e)}
    
    def _test_mean_reversion(self, series):
        """Simple mean reversion test with better error handling"""
        if len(series) < 20:
            return 0.0
        
        try:
            # Remove NaN values
            series = series.dropna()
            if len(series) < 20:
                return 0.0
            
            mean_val = series.mean()
            if pd.isna(mean_val):
                return 0.0
            
            # Count mean crossings
            above_mean = series > mean_val
            crossings = (above_mean != above_mean.shift(1)).sum()
            crossing_rate = crossings / len(series) if len(series) > 0 else 0
            
            return min(crossing_rate * 2, 1.0)
        except Exception:
            return 0.0
    
    def optimize_zscore_parameters(self, price1, price2):
        """Find optimal z-score parameters with better error handling"""
        if price1.empty or price2.empty or len(price1) < 50 or len(price2) < 50:
            return {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}, 0.0
        
        # Quick optimization with fewer combinations for speed
        best_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
        best_return = -999
        
        # Reduced parameter ranges for faster optimization
        entry_levels = [1.5, 2.0, 2.5, 3.0]
        exit_levels = [0.3, 0.5, 0.8]
        stop_levels = [3.0, 3.5, 4.0]
        
        progress_bar = st.progress(0)
        total_combinations = len(entry_levels) * len(exit_levels) * len(stop_levels)
        current_combination = 0
        
        try:
            for entry in entry_levels:
                for exit in exit_levels:
                    for stop in stop_levels:
                        if stop > entry and exit < entry:  # Logical constraints
                            
                            # Test these parameters
                            signals_df = self.generate_trading_signals(price1, price2, entry, exit, stop)
                            if not signals_df.empty:
                                performance = self.backtest_strategy(signals_df, price1, price2)
                                
                                # Simple optimization criteria
                                if performance['num_trades'] > 2:  # Lower minimum trade requirement
                                    score = performance['total_return'] * (performance['win_rate'] / 100)
                                    
                                    if score > best_return:
                                        best_return = score
                                        best_params = {'entry': entry, 'exit': exit, 'stop': stop}
                        
                        current_combination += 1
                        progress_bar.progress(current_combination / total_combinations)
        
        except Exception as e:
            st.warning(f"Optimization error: {e}")
        finally:
            progress_bar.empty()
        
        return best_params, best_return
    
    def calculate_spread_zscore(self, price1, price2, lookback_period=60):
        """Calculate spread and z-score with robust error handling"""
        if price1.empty or price2.empty:
            return pd.DataFrame(), 1.0
        
        try:
            # Align prices
            price1, price2 = price1.align(price2, join='inner')
            
            if len(price1) < max(20, lookback_period // 2):
                st.warning("Insufficient data for spread calculation")
                return pd.DataFrame(), 1.0
            
            # Dynamic hedge ratio calculation
            recent_data_points = min(lookback_period, len(price1), 100)
            
            # Ensure we have enough data for regression
            if recent_data_points < 10:
                hedge_ratio = 1.0
            else:
                try:
                    X = price2.iloc[-recent_data_points:].values.reshape(-1, 1)
                    y = price1.iloc[-recent_data_points:].values
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                    if mask.sum() < 5:  # Need at least 5 valid points
                        hedge_ratio = 1.0
                    else:
                        X_clean = X[mask].reshape(-1, 1)
                        y_clean = y[mask]
                        
                        model = LinearRegression().fit(X_clean, y_clean)
                        hedge_ratio = float(model.coef_[0])
                        
                        # Validate hedge ratio
                        if abs(hedge_ratio) > 20 or abs(hedge_ratio) < 0.01 or np.isnan(hedge_ratio):
                            hedge_ratio = price1.mean() / price2.mean() if price2.mean() != 0 and not np.isnan(price2.mean()) else 1.0
                
                except Exception:
                    hedge_ratio = 1.0
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            spread = spread.dropna()
            
            if len(spread) < 10:
                return pd.DataFrame(), hedge_ratio
            
            # Dynamic z-score calculation
            zscore_window = min(20, len(spread) // 3)
            if zscore_window < 5:
                zscore_window = 5
            
            rolling_mean = spread.rolling(window=zscore_window, min_periods=zscore_window//2).mean()
            rolling_std = spread.rolling(window=zscore_window, min_periods=zscore_window//2).std()
            
            # Handle edge cases for std
            rolling_std = rolling_std.fillna(spread.std())
            rolling_std = rolling_std.replace(0, spread.std() if spread.std() > 0 else 0.01)
            
            # Calculate z-score
            zscore = (spread - rolling_mean) / rolling_std
            zscore = zscore.fillna(0)
            
            # Align all series
            aligned_data = pd.DataFrame({
                'price1': price1,
                'price2': price2,
                'spread': spread,
                'zscore': zscore,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std
            }).dropna()
            
            return aligned_data, hedge_ratio
            
        except Exception as e:
            st.error(f"Spread calculation failed: {e}")
            return pd.DataFrame(), 1.0
    
    def generate_trading_signals(self, price1, price2, entry_zscore=2.0, exit_zscore=0.5, stop_zscore=3.5):
        """Generate trading signals with better error handling"""
        analysis_df, hedge_ratio = self.calculate_spread_zscore(price1, price2)
        
        if analysis_df.empty or len(analysis_df) < 10:
            return pd.DataFrame()
        
        try:
            signals = []
            current_position = 0  # 0=flat, 1=long_spread, -1=short_spread
            
            for timestamp, row in analysis_df.iterrows():
                zscore = row['zscore']
                
                # Skip if zscore is NaN or infinite
                if pd.isna(zscore) or np.isinf(zscore):
                    continue
                
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
                        
                    elif zscore >= entry_zscore:  # Enter short spread
                        signal_data['action'] = 'ENTER_SHORT_SPREAD'
                        signal_data['entry_price'] = row['spread']
                        current_position = -1
                
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
        
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, signals_df, price1, price2):
        """Comprehensive backtesting with better error handling"""
        if signals_df.empty:
            return self._empty_performance_metrics()
        
        try:
            trades = []
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
                    if current_trade and signal['exit_price'] is not None:
                        # Calculate P&L
                        entry_spread = current_trade['entry_spread']
                        exit_spread = signal['exit_price']
                        
                        if pd.isna(entry_spread) or pd.isna(exit_spread):
                            continue
                        
                        if current_trade['position_type'] == 'ENTER_LONG_SPREAD':
                            pnl = exit_spread - entry_spread
                        else:  # SHORT_SPREAD
                            pnl = entry_spread - exit_spread
                        
                        # Calculate percentage return
                        pnl_percentage = (pnl / abs(entry_spread)) * 100 if abs(entry_spread) > 0 else 0
                        
                        # Skip trades with extreme or invalid returns
                        if abs(pnl_percentage) < 100:  # Sanity check
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
        
        except Exception as e:
            st.error(f"Backtesting failed: {e}")
            return self._empty_performance_metrics()
    
    def _calculate_performance_metrics(self, trades):
        """Calculate comprehensive performance statistics"""
        if not trades:
            return self._empty_performance_metrics()
        
        try:
            trades_df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl_percentage'] > 0]
            losing_trades = trades_df[trades_df['pnl_percentage'] <= 0]
            
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            total_return = trades_df['pnl_percentage'].sum()
            
            avg_win = winning_trades['pnl_percentage'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl_percentage'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if avg_loss != 0 and len(losing_trades) > 0 else float('inf')
            
            # Risk metrics
            returns_series = trades_df['pnl_percentage']
            max_drawdown = self._calculate_max_drawdown(returns_series)
            
            sharpe_ratio = 0
            if returns_series.std() > 0 and not pd.isna(returns_series.std()):
                sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
            
            # Handle NaN values
            for metric in ['total_return', 'win_rate', 'avg_win', 'avg_loss', 'max_drawdown', 'sharpe_ratio']:
                value = locals()[metric]
                if pd.isna(value) or np.isinf(value):
                    locals()[metric] = 0
            
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
        
        except Exception as e:
            st.error(f"Performance calculation failed: {e}")
            return self._empty_performance_metrics()
    
    def _calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown"""
        try:
            if returns_series.empty or returns_series.std() == 0:
                return 0.0
            
            cumulative = (1 + returns_series / 100).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
            
            return max_dd if not pd.isna(max_dd) else 0.0
        except:
            return 0.0
    
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
        
        try:
            price1 = current_signal['price1']
            price2 = current_signal['price2']
            
            if pd.isna(price1) or pd.isna(price2) or price1 <= 0 or price2 <= 0:
                return None
            
            # Total available capital with leverage
            total_capital = capital * leverage
            
            # Equal allocation to both sides
            capital_per_side = total_capital / 2
            
            if current_signal['action'] == 'ENTER_LONG_SPREAD':
                # Long spread: Buy asset1, Short asset2
                asset1_quantity = capital_per_side / price1
                asset2_quantity = (capital_per_side / price2) * abs(hedge_ratio)
                
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
                asset2_quantity = (capital_per_side / price2) * abs(hedge_ratio)
                
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
        except Exception as e:
            st.error(f"Position calculation failed: {e}")
            return None

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
    
    # Improved data validation
    min_data_points = 50
    if (not price1.empty and not price2.empty and 
        len(price1) >= min_data_points and len(price2) >= min_data_points):
        
        st.success(f"✅ Data loaded: {asset1_name} ({len(price1)} points), {asset2_name} ({len(price2)} points)")
        
        # Correlation analysis
        with st.spinner("Calculating correlation metrics..."):
            correlation_metrics = system.calculate_correlation_metrics(price1, price2)
        
        # Show correlation warning but continue
        if not correlation_metrics.get('suitable', False):
            correlation_value = correlation_metrics.get('correlation', 0)
            st.warning(f"⚠️ Correlation: {correlation_value:.3f}. This pair may have limited trading opportunities.")
        
        # Parameter optimization
        if optimize_button:
            with st.spinner("🔬 Optimizing parameters for maximum profit..."):
                optimal_params, optimization_score = system.optimize_zscore_parameters(price1, price2)
                st.info(f"✅ Optimization completed. Score: {optimization_score:.3f}")
        else:
            optimal_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
            optimization_score = 0
        
        # Generate signals and backtest
        with st.spinner("Generating trading signals..."):
            signals_df = system.generate_trading_signals(price1, price2, **optimal_params)
        
        if signals_df.empty:
            st.error("❌ Could not generate trading signals. Try different parameters or time period.")
        else:
            with st.spinner("Running backtest..."):
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
                correlation_value = correlation_metrics.get('correlation', 0)
                color_class = "positive" if correlation_value > 0.5 else "negative" if correlation_value < -0.5 else "neutral"
                st.markdown(f'<div class="metric-value {color_class}">{correlation_value:.3f}</div>', unsafe_allow_html=True)
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
                        **ACTION:** {position_details['asset2_action']}  
                        **QUANTITY:** {position_details['asset2_quantity']:.6f}  
                        **PRICE:** ${position_details['asset2_price']:.4f}  
                        **VALUE:** ${position_details['asset2_value']:.2f}  
                        """)
                    
                    st.markdown(f"""
                    ---
                    **TOTAL EXPOSURE:** ${position_details['total_exposure']:.2f}  
                    **HEDGE RATIO:** {position_details['hedge_ratio']:.4f}  
                    **ENTRY Z-SCORE:** {latest_signal['zscore']:.2f}  
                    **EXIT TARGET:** ±{optimal_params['exit']:.1f}  
                    **STOP LOSS:** ±{optimal_params['stop']:.1f}  
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown('<div class="signal-inactive">', unsafe_allow_html=True)
                st.markdown("## 📊 MONITORING MODE")
                st.markdown(f"""
                **CURRENT Z-SCORE:** {current_zscore:.2f}  
                **ENTRY THRESHOLD:** ±{optimal_params['entry']:.1f}  
                **DISTANCE TO SIGNAL:** {abs(optimal_params['entry'] - abs(current_zscore)):.2f}  
                **OPTIMAL PARAMETERS:** Entry: {optimal_params['entry']:.1f} | Exit: {optimal_params['exit']:.1f} | Stop: {optimal_params['stop']:.1f}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts
            st.markdown("### MARKET ANALYSIS CHARTS")
            
            try:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=[
                        f'{asset1_name} vs {asset2_name} - Normalized Price Comparison',
                        'Z-Score Analysis & Signal Levels'
                    ],
                    vertical_spacing=0.12,
                    row_heights=[0.6, 0.4]
                )
                
                # Normalize prices for comparison
                if not price1.empty and not price2.empty:
                    norm_price1 = (price1 / price1.iloc[0]) * 100
                    norm_price2 = (price2 / price2.iloc[0]) * 100
                    
                    fig.add_trace(
                        go.Scatter(x=price1.index, y=norm_price1, name=asset1_name, 
                                  line=dict(color='#00ff41', width=2)), row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=price2.index, y=norm_price2, name=asset2_name, 
                                  line=dict(color='#ff4444', width=2)), row=1, col=1
                    )
                
                # Z-score chart
                if not analysis_df.empty:
                    fig.add_trace(
                        go.Scatter(x=analysis_df.index, y=analysis_df['zscore'], 
                                  name='Z-Score', line=dict(color='#00ffff', width=2)), row=2, col=1
                    )
                    
                    # Signal levels
                    fig.add_hline(y=optimal_params['entry'], line_dash="dash", 
                                 line_color="#ff4444", row=2, col=1, 
                                 annotation_text="Entry Level")
                    fig.add_hline(y=-optimal_params['entry'], line_dash="dash", 
                                 line_color="#00ff41", row=2, col=1)
                    fig.add_hline(y=optimal_params['exit'], line_dash="dot", 
                                 line_color="#ffff00", row=2, col=1, 
                                 annotation_text="Exit Level")
                    fig.add_hline(y=-optimal_params['exit'], line_dash="dot", 
                                 line_color="#ffff00", row=2, col=1)
                    fig.add_hline(y=0, line_color="#666666", row=2, col=1)
                
                fig.update_layout(
                    height=800,
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font_color='#00ff41',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Chart generation failed: {e}")
            
            # Trading Statistics
            if not performance_metrics['trades_data'].empty:
                st.markdown("### DETAILED PERFORMANCE STATISTICS")
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.markdown("#### Trade Summary")
                    st.markdown(f"""
                    **Total Trades:** {performance_metrics['num_trades']}  
                    **Winning Trades:** {len(performance_metrics['trades_data'][performance_metrics['trades_data']['pnl_percentage'] > 0])}  
                    **Losing Trades:** {len(performance_metrics['trades_data'][performance_metrics['trades_data']['pnl_percentage'] <= 0])}  
                    **Win Rate:** {performance_metrics['win_rate']:.1f}%  
                    **Profit Factor:** {performance_metrics['profit_factor']:.2f}  
                    """)
                
                with stat_col2:
                    st.markdown("#### Return Analysis")
                    st.markdown(f"""
                    **Total Return:** {performance_metrics['total_return']:.2f}%  
                    **Average Win:** {performance_metrics['avg_win']:.2f}%  
                    **Average Loss:** {performance_metrics['avg_loss']:.2f}%  
                    **Max Drawdown:** {performance_metrics['max_drawdown']:.2f}%  
                    **Sharpe Ratio:** {performance_metrics['sharpe_ratio']:.2f}  
                    """)
                
                with stat_col3:
                    st.markdown("#### Best/Worst Trades")
                    if performance_metrics['best_trade'] is not None:
                        best_trade = performance_metrics['best_trade']
                        st.markdown(f"""
                        **Best Trade:** +{best_trade['pnl_percentage']:.2f}%  
                        **Date:** {best_trade['entry_date'].strftime('%Y-%m-%d')}  
                        **Type:** {best_trade['position_type']}  
                        """)
                    
                    if performance_metrics['worst_trade'] is not None:
                        worst_trade = performance_metrics['worst_trade']
                        st.markdown(f"""
                        **Worst Trade:** {worst_trade['pnl_percentage']:.2f}%  
                        **Date:** {worst_trade['entry_date'].strftime('%Y-%m-%d')}  
                        **Type:** {worst_trade['position_type']}  
                        """)
                
                # Trade history table
                st.markdown("#### Complete Trade History")
                try:
                    trades_display = performance_metrics['trades_data'].copy()
                    trades_display['entry_date'] = trades_display['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
                    trades_display['exit_date'] = trades_display['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
                    trades_display = trades_display.round(4)
                    
                    st.dataframe(
                        trades_display[['entry_date', 'exit_date', 'position_type', 'entry_zscore', 
                                       'exit_zscore', 'pnl_percentage', 'exit_reason']],
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.warning(f"Trade history display failed: {e}")
            
            else:
                st.info("📊 No trades generated with current parameters. Try adjusting the entry threshold or time period.")
    
    else:
        # Better error messaging
        if price1.empty:
            st.error(f"❌ No data available for {asset1_name} ({asset1_ticker})")
        elif price2.empty:
            st.error(f"❌ No data available for {asset2_name} ({asset2_ticker})")
        else:
            st.error(f"❌ Insufficient data: {asset1_name} ({len(price1)} points), {asset2_name} ({len(price2)} points). Need at least {min_data_points} points each.")
        
        st.info("💡 Try selecting different assets or a longer time period (1y or 2y)")

# Save pair functionality
if save_button and hasattr(system, 'current_analysis') and system.current_analysis:
    try:
        pair_identifier = f"{system.current_analysis['asset1_name']}/{system.current_analysis['asset2_name']}"
        
        pair_data = {
            'pair_name': pair_identifier,
            'asset1_name': system.current_analysis['asset1_name'],
            'asset2_name': system.current_analysis['asset2_name'],
            'asset1_ticker': system.current_analysis['asset1_ticker'],
            'asset2_ticker': system.current_analysis['asset2_ticker'],
            'correlation': system.current_analysis['correlation_metrics'].get('correlation', 0),
            'total_return': system.current_analysis['performance']['total_return'],
            'win_rate': system.current_analysis['performance']['win_rate'],
            'sharpe_ratio': system.current_analysis['performance']['sharpe_ratio'],
            'optimal_params': system.current_analysis['optimal_params'],
            'current_zscore': system.current_analysis['current_zscore']
        }
        
        # Check if pair already exists
        existing_pair = next((p for p in st.session_state.saved_pairs if p['pair_name'] == pair_identifier), None)
        
        if existing_pair:
            st.warning(f"⚠️ Pair {pair_identifier} already saved. Updating with new analysis.")
            st.session_state.saved_pairs = [p for p in st.session_state.saved_pairs if p['pair_name'] != pair_identifier]
        
        st.session_state.saved_pairs.append(pair_data)
        st.success(f"✅ Saved pair: {pair_identifier}")
    
    except Exception as e:
        st.error(f"Failed to save pair: {e}")

elif save_button:
    st.warning("⚠️ No analysis data to save. Run analysis first.")

# Clear data
if clear_button:
    system.current_analysis = {}
    st.success("🗑️ Analysis data cleared")

# Saved Pairs Management
if st.session_state.saved_pairs:
    st.markdown("### SAVED TRADING PAIRS")
    
    for idx, pair in enumerate(st.session_state.saved_pairs):
        with st.expander(f"📊 {pair['pair_name']} | Return: {pair['total_return']:.1f}% | Win Rate: {pair['win_rate']:.1f}%"):
            
            pair_col1, pair_col2, pair_col3, pair_col4 = st.columns([2, 2, 1, 1])
            
            with pair_col1:
                st.markdown(f"""
                **Assets:** {pair['asset1_name']} / {pair['asset2_name']}  
                **Tickers:** {pair['asset1_ticker']} / {pair['asset2_ticker']}  
                **Correlation:** {pair['correlation']:.3f}  
                """)
            
            with pair_col2:
                st.markdown(f"""
                **Total Return:** {pair['total_return']:.2f}%  
                **Win Rate:** {pair['win_rate']:.1f}%  
                **Sharpe Ratio:** {pair['sharpe_ratio']:.2f}  
                """)
            
            with pair_col3:
                params = pair['optimal_params']
                st.markdown(f"""
                **Entry:** ±{params['entry']:.1f}  
                **Exit:** ±{params['exit']:.1f}  
                **Stop:** ±{params['stop']:.1f}  
                """)
            
            with pair_col4:
                if st.button(f"📈 Load Pair", key=f"load_{idx}"):
                    # Load this pair for analysis
                    st.session_state.primary_asset = pair['asset1_name']
                    st.session_state.secondary_asset = pair['asset2_name']
                    st.rerun()
                
                if st.button(f"🗑️ Delete", key=f"delete_{idx}"):
                    st.session_state.saved_pairs.pop(idx)
                    st.rerun()

# Sidebar Status Panel
with st.sidebar:
    st.markdown("### 🎯 SYSTEM STATUS")
    
    if hasattr(system, 'current_analysis') and system.current_analysis:
        current = system.current_analysis
        
        st.success(f"✅ ACTIVE PAIR")
        st.markdown(f"**{current['asset1_name']} / {current['asset2_name']}**")
        
        # Current metrics
        st.metric("Correlation", f"{current['correlation_metrics'].get('correlation', 0):.3f}")
        st.metric("Current Z-Score", f"{current['current_zscore']:.2f}")
        st.metric("Backtest Return", f"{current['performance']['total_return']:.1f}%")
        st.metric("Win Rate", f"{current['performance']['win_rate']:.1f}%")
        
        # Optimal parameters
        params = current['optimal_params']
        st.markdown("**Optimal Parameters:**")
        st.markdown(f"Entry: ±{params['entry']:.1f}")
        st.markdown(f"Exit: ±{params['exit']:.1f}")
        st.markdown(f"Stop: ±{params['stop']:.1f}")
        
        # Signal status
        if abs(current['current_zscore']) > params['entry']:
            st.error("🚨 SIGNAL ACTIVE")
        else:
            st.info("📊 MONITORING")
    
    else:
        st.info("💤 NO ACTIVE PAIR")
        st.markdown("Select assets and run analysis")
    
    st.markdown("---")
    st.markdown("### 📊 QUICK STATS")
    st.metric("Saved Pairs", len(st.session_state.saved_pairs))
    st.metric("Trading Capital", f"${trading_capital:,}")
    st.metric("Leverage", f"{leverage_factor}x")
    
    st.markdown("---")
    st.markdown("### 🔧 TOOLS")
    
    if st.button("📥 Export Data", use_container_width=True):
        if hasattr(system, 'current_analysis') and system.current_analysis and not system.current_analysis.get('signals_df', pd.DataFrame()).empty:
            try:
                # Export current analysis data
                st.download_button(
                    "Download Analysis",
                    data=system.current_analysis['signals_df'].to_csv(),
                    file_name=f"{system.current_analysis['asset1_name']}_{system.current_analysis['asset2_name']}_analysis.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
        else:
            st.warning("No data to export")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared")
    
    st.markdown("---")
    st.markdown("*Advanced Crypto Pairs Trading v2.1*")
