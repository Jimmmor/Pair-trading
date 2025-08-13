import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import tickers from constants file (same as original)
from constants.tickers import tickers

# Page configuration
st.set_page_config(page_title="AI Pairs Trading System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .signal-buy {
        background: linear-gradient(90deg, #00C851, #007E33);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(90deg, #ff4444, #CC0000);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .no-signal {
        background: linear-gradient(90deg, #666666, #444444);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Move the cached function outside of the class
@st.cache_data
def load_crypto_data(symbol, period='1y'):
    """Load cryptocurrency data using imported tickers"""
    try:
        # Use the imported tickers dictionary
        ticker_symbol = tickers.get(symbol, symbol)
        st.write(f"Debug - Loading {symbol} with ticker {ticker_symbol}")
        
        # Download data
        data = yf.download(ticker_symbol, period=period, progress=False)
        
        st.write(f"Debug - Raw data type: {type(data)}")
        st.write(f"Debug - Raw data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        
        if data.empty:
            st.error(f"No data returned for {symbol}")
            return pd.Series(dtype=float)
        
        # Extract Close price
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                close_data = data['Close']
            else:
                # If single column, use it
                close_data = data.iloc[:, -1]  # Last column
        else:
            # If it's already a Series
            close_data = data
        
        # Clean data
        close_data = close_data.dropna()
        
        st.write(f"Debug - Final data type: {type(close_data)}")
        st.write(f"Debug - Final data length: {len(close_data)}")
        
        return close_data
        
    except Exception as e:
        st.error(f"Error loading {symbol}: {str(e)}")
        return pd.Series(dtype=float)

class MLPairsTradingSystem:
    """Advanced ML-Optimized Pairs Trading System"""
    
    def __init__(self):
        self.optimal_params = {}
        self.best_performance = None
        
    def load_data(self, symbol, period='1y'):
        """Load cryptocurrency data using imported tickers"""
        return load_crypto_data(symbol, period)
        
    def calculate_spread_and_zscore(self, price1, price2, zscore_window=30):
        # Zorg dat inputs Pandas Series zijn
        if not isinstance(price1, pd.Series):
            if isinstance(price1, (int, float)):
                raise ValueError("price1 moet een tijdreeks zijn, geen enkele waarde.")
            price1 = pd.Series(price1)
    
        if not isinstance(price2, pd.Series):
            if isinstance(price2, (int, float)):
                raise ValueError("price2 moet een tijdreeks zijn, geen enkele waarde.")
            price2 = pd.Series(price2)
    
        # Align data op gemeenschappelijke index
        common_index = price1.index.intersection(price2.index)
        if len(common_index) < zscore_window:
            raise ValueError(
                f"Niet genoeg overlappende data: {len(common_index)} punten gevonden, "
                f"maar {zscore_window} nodig voor de z-score berekening."
            )
    
        price1_aligned = price1.loc[common_index]
        price2_aligned = price2.loc[common_index]
    
        # Bereken spread
        spread = price1_aligned - price2_aligned
    
        # Bereken z-score
        rolling_mean = spread.rolling(window=zscore_window).mean()
        rolling_std = spread.rolling(window=zscore_window).std()
        zscore = (spread - rolling_mean) / rolling_std
    
        # Bouw DataFrame (nu altijd met Series, geen scalars)
        spread_df = pd.DataFrame({
            'price1': price1_aligned,
            'price2': price2_aligned,
            'spread': spread,
            'zscore': zscore
        })
    
        return spread_df

    
    def backtest_strategy(self, df, entry_zscore, exit_zscore, stop_loss_pct, 
                         take_profit_pct, leverage, max_hold_days=30):
        """Comprehensive backtesting with realistic trading costs"""
        capital = 10000
        position = 0  # 0 = no position, 1 = long spread, -1 = short spread
        trades = []
        portfolio_value = [capital]
        entry_price = 0
        entry_date = None
        transaction_cost = 0.001  # 0.1% per trade
        
        for i in range(50, len(df)):  # Start after warmup period
            current_date = df.index[i]
            current_zscore = df['zscore'].iloc[i]
            
            if pd.isna(current_zscore):
                portfolio_value.append(portfolio_value[-1])
                continue
            
            # Check for position exit first
            if position != 0:
                days_held = (current_date - entry_date).days
                
                # Calculate current P&L
                if position == 1:  # Long spread
                    pnl_pct = ((df['spread'].iloc[i] - entry_price) / entry_price) * leverage
                else:  # Short spread
                    pnl_pct = ((entry_price - df['spread'].iloc[i]) / entry_price) * leverage
                
                current_value = capital * (1 + pnl_pct)
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                if position == 1 and current_zscore >= -exit_zscore:
                    should_exit, exit_reason = True, "mean_reversion"
                elif position == -1 and current_zscore <= exit_zscore:
                    should_exit, exit_reason = True, "mean_reversion"
                elif pnl_pct <= -stop_loss_pct/100:
                    should_exit, exit_reason = True, "stop_loss"
                elif pnl_pct >= take_profit_pct/100:
                    should_exit, exit_reason = True, "take_profit"
                elif days_held >= max_hold_days:
                    should_exit, exit_reason = True, "time_exit"
                
                if should_exit:
                    # Execute exit
                    final_value = current_value * (1 - transaction_cost)
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position_type': 'long_spread' if position == 1 else 'short_spread',
                        'entry_zscore': entry_zscore if position == 1 else -entry_zscore,
                        'exit_zscore': current_zscore,
                        'days_held': days_held,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': exit_reason
                    })
                    
                    capital = final_value
                    position = 0
                
                portfolio_value.append(current_value if position != 0 else capital)
            
            # Check for new entry signals (only if no position)
            elif position == 0:
                if current_zscore <= -entry_zscore:
                    # Enter long spread
                    position = 1
                    entry_price = df['spread'].iloc[i]
                    entry_date = current_date
                    capital *= (1 - transaction_cost)  # Entry transaction cost
                    
                elif current_zscore >= entry_zscore:
                    # Enter short spread
                    position = -1
                    entry_price = df['spread'].iloc[i]
                    entry_date = current_date
                    capital *= (1 - transaction_cost)  # Entry transaction cost
                
                portfolio_value.append(capital)
            else:
                portfolio_value.append(portfolio_value[-1])
        
        # Calculate performance metrics
        final_value = portfolio_value[-1]
        total_return = (final_value - 10000) / 10000 * 100
        
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
            avg_trade_pct = trades_df['pnl_pct'].mean()
            max_drawdown = self._calculate_max_drawdown(portfolio_value)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_value)
        else:
            win_rate = 0
            avg_trade_pct = 0
            max_drawdown = 0
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_trade_pct': avg_trade_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_values': portfolio_value,
            'trades': trades_df
        }
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100
    
    def _calculate_sharpe_ratio(self, portfolio_values):
        """Calculate Sharpe ratio"""
        returns = pd.Series(portfolio_values).pct_change().dropna()
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    def optimize_parameters(self, price1, price2, trading_timeframe_days=30):
        """ML-powered parameter optimization"""
        
        # Validate inputs
        if len(price1) < 50 or len(price2) < 50:
            raise ValueError("Insufficient data for optimization. Need at least 50 data points.")
        
        st.info("🤖 AI is optimizing parameters for maximum profit...")
        
        # Define parameter grid based on trading timeframe
        if trading_timeframe_days <= 7:  # Short-term
            param_grid = {
                'entry_zscore': [1.5, 2.0, 2.5, 3.0],
                'exit_zscore': [0.2, 0.5, 0.8, 1.0],
                'zscore_window': [10, 15, 20],
                'stop_loss_pct': [3, 5, 8],
                'take_profit_pct': [6, 10, 15],
                'leverage': [2, 3, 5],
                'hedge_method': ['dollar_neutral', 'regression']
            }
        elif trading_timeframe_days <= 30:  # Medium-term
            param_grid = {
                'entry_zscore': [2.0, 2.5, 3.0],
                'exit_zscore': [0.5, 0.8, 1.0],
                'zscore_window': [15, 20, 30],
                'stop_loss_pct': [5, 8, 12],
                'take_profit_pct': [10, 15, 20],
                'leverage': [2, 3, 5, 8],
                'hedge_method': ['dollar_neutral', 'regression']
            }
        else:  # Long-term
            param_grid = {
                'entry_zscore': [2.5, 3.0, 3.5],
                'exit_zscore': [0.8, 1.0, 1.5],
                'zscore_window': [20, 30, 40],
                'stop_loss_pct': [8, 12, 15],
                'take_profit_pct': [15, 20, 25],
                'leverage': [2, 3, 5],
                'hedge_method': ['dollar_neutral', 'regression']
            }
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Create progress bar
        param_combinations = list(ParameterGrid(param_grid))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, params in enumerate(param_combinations[:50]):  # Limit to 50 combinations for speed
            try:
                # Calculate spread and z-score
                df, hedge_ratio = self.calculate_spread_and_zscore(
                    price1, price2, 
                    zscore_window=params['zscore_window'],
                    hedge_method=params['hedge_method']
                )
                
                # Backtest with current parameters
                results_dict = self.backtest_strategy(
                    df, 
                    entry_zscore=params['entry_zscore'],
                    exit_zscore=params['exit_zscore'],
                    stop_loss_pct=params['stop_loss_pct'],
                    take_profit_pct=params['take_profit_pct'],
                    leverage=params['leverage'],
                    max_hold_days=trading_timeframe_days
                )
                
                # Scoring function (weighted combination of metrics)
                score = (
                    results_dict['total_return'] * 0.4 +
                    results_dict['win_rate'] * 0.3 +
                    results_dict['sharpe_ratio'] * 20 * 0.2 +
                    (100 - results_dict['max_drawdown']) * 0.1
                )
                
                results.append({**params, **results_dict, 'score': score})
                
                if score > best_score and results_dict['num_trades'] >= 3:
                    best_score = score
                    best_params = params.copy()
                    best_params['hedge_ratio'] = hedge_ratio
                
                # Update progress
                progress_bar.progress((i + 1) / min(50, len(param_combinations)))
                status_text.text(f"Testing combination {i+1}/{min(50, len(param_combinations))} - Best return: {results_dict['total_return']:.2f}%")
                
            except Exception as e:
                # Skip this parameter combination if it fails
                st.warning(f"Skipping parameter combination due to error: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found. Check your data quality.")
        
        self.optimal_params = best_params
        self.best_performance = [r for r in results if r['score'] == best_score][0]
        
        return best_params, pd.DataFrame(results)

# Initialize the system (removed @st.cache_resource decorator as it can cause issues)
def get_trading_system():
    return MLPairsTradingSystem()

# Initialize session state to store the trading system
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = get_trading_system()

trading_system = st.session_state.trading_system

# Main App Interface
st.title("🤖 AI-Powered Pairs Trading System")
st.markdown("*Professional cryptocurrency pairs trading with machine learning optimization*")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Pair Analysis & Optimization", "🎯 Live Trading Signals", "📈 Performance Analytics"])

# Tab 1: Pair Analysis & Optimization
with tab1:
    st.header("🔍 Pair Selection & AI Optimization")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        crypto1 = st.selectbox("Select Crypto 1:", list(tickers.keys()), index=0)
    with col2:
        remaining_cryptos = [c for c in tickers.keys() if c != crypto1]
        crypto2 = st.selectbox("Select Crypto 2:", remaining_cryptos, index=0)
    with col3:
        timeframe_days = st.selectbox("Trading Timeframe:", [7, 14, 30, 90], index=2)
    
    # Load data
    if st.button("🚀 Analyze Pair & Optimize with AI", type="primary"):
        try:
            with st.spinner("Loading market data..."):
                price1 = trading_system.load_data(crypto1)
                price2 = trading_system.load_data(crypto2)
            
            # Debug: Check what we actually loaded
            st.write(f"Debug - price1 type: {type(price1)}, length: {len(price1) if hasattr(price1, '__len__') else 'N/A'}")
            st.write(f"Debug - price2 type: {type(price2)}, length: {len(price2) if hasattr(price2, '__len__') else 'N/A'}")
            st.write(f"Debug - price1 empty: {price1.empty if hasattr(price1, 'empty') else 'N/A'}")
            st.write(f"Debug - price2 empty: {price2.empty if hasattr(price2, 'empty') else 'N/A'}")
            
            if not price1.empty and not price2.empty and len(price1) > 50 and len(price2) > 50:
                st.success(f"✅ Data loaded successfully! {crypto1}: {len(price1)} points, {crypto2}: {len(price2)} points")
                
                # Run ML optimization
                with st.spinner("Running AI optimization..."):
                    optimal_params, all_results = trading_system.optimize_parameters(
                        price1, price2, timeframe_days
                    )
                
                # Display results
                st.success("✅ AI Optimization Complete!")
                
                # Show optimal parameters
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Optimal Entry Z-Score", f"±{optimal_params['entry_zscore']:.1f}")
                    st.metric("Optimal Exit Z-Score", f"±{optimal_params['exit_zscore']:.1f}")
                with col2:
                    st.metric("Best Leverage", f"{optimal_params['leverage']}x")
                    st.metric("Hedge Method", optimal_params['hedge_method'].replace('_', ' ').title())
                with col3:
                    st.metric("Stop Loss", f"{optimal_params['stop_loss_pct']}%")
                    st.metric("Take Profit", f"{optimal_params['take_profit_pct']}%")
                with col4:
                    st.metric("Expected Return", f"{trading_system.best_performance['total_return']:.1f}%")
                    st.metric("Win Rate", f"{trading_system.best_performance['win_rate']:.1f}%")
                
                # Create visualizations
                df, _ = trading_system.calculate_spread_and_zscore(
                    price1, price2, 
                    optimal_params['zscore_window'], 
                    optimal_params['hedge_method']
                )
                
                # Price chart with trading zones
                fig = go.Figure()
                
                # Add price lines
                fig.add_trace(go.Scatter(x=df.index, y=df['price1'], name=f'{crypto1} Price', 
                                       line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['price2'], name=f'{crypto2} Price', 
                                       line=dict(color='red'), yaxis='y2'))
                
                fig.update_layout(
                    title=f"{crypto1} vs {crypto2} - Price Movement",
                    xaxis_title="Date",
                    yaxis_title=f"{crypto1} Price (USD)",
                    yaxis2=dict(title=f"{crypto2} Price (USD)", overlaying='y', side='right'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Z-score chart with optimal entry/exit levels
                fig_zscore = go.Figure()
                fig_zscore.add_trace(go.Scatter(x=df.index, y=df['zscore'], name='Z-Score',
                                              line=dict(color='green', width=2)))
                
                # Add optimal trading thresholds
                fig_zscore.add_hline(y=optimal_params['entry_zscore'], line_dash="dash", 
                                   line_color="red", annotation_text="Short Entry")
                fig_zscore.add_hline(y=-optimal_params['entry_zscore'], line_dash="dash", 
                                   line_color="blue", annotation_text="Long Entry")
                fig_zscore.add_hline(y=optimal_params['exit_zscore'], line_dash="dot", 
                                   line_color="purple", annotation_text="Exit Zone")
                fig_zscore.add_hline(y=-optimal_params['exit_zscore'], line_dash="dot", 
                                   line_color="purple", annotation_text="Exit Zone")
                fig_zscore.add_hline(y=0, line_color="black", line_width=1)
                
                fig_zscore.update_layout(
                    title="Z-Score with Optimal Entry/Exit Levels",
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    height=400
                )
                
                st.plotly_chart(fig_zscore, use_container_width=True)
                
                # Performance comparison table
                st.subheader("🏆 Top 10 Parameter Combinations")
                top_results = all_results.nlargest(10, 'total_return')[
                    ['entry_zscore', 'exit_zscore', 'leverage', 'hedge_method', 
                     'total_return', 'win_rate', 'num_trades', 'max_drawdown']
                ]
                st.dataframe(top_results.round(2), use_container_width=True)
            else:
                st.error("❌ Failed to load data for one or both cryptocurrencies!")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)  # This will show the full traceback for debugging

# Tab 2: Live Trading Signals
with tab2:
    st.header("🎯 Live Trading Signals & Execution")
    
    if hasattr(trading_system, 'optimal_params') and trading_system.optimal_params:
        # Capital and leverage settings
        col1, col2, col3 = st.columns(3)
        with col1:
            capital = st.number_input("💰 Trading Capital (USDT):", 
                                    min_value=100, max_value=1000000, value=5000, step=100)
        with col2:
            leverage = st.slider("🔥 Leverage Multiplier:", 
                               min_value=1, max_value=50, 
                               value=trading_system.optimal_params['leverage'], step=1)
        with col3:
            risk_per_trade = st.slider("🎯 Risk per Trade (%):", 
                                     min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        
        # Get current market data
        current_price1 = trading_system.load_data(crypto1, period='5d').iloc[-1]
        current_price2 = trading_system.load_data(crypto2, period='5d').iloc[-1]
        
        # Calculate current z-score
        recent_data1 = trading_system.load_data(crypto1, period='3mo')
        recent_data2 = trading_system.load_data(crypto2, period='3mo')
        df_current, hedge_ratio = trading_system.calculate_spread_and_zscore(
            recent_data1, recent_data2, 
            trading_system.optimal_params['zscore_window'],
            trading_system.optimal_params['hedge_method']
        )
        current_zscore = df_current['zscore'].iloc[-1]
        
        # Display current market status
        st.subheader("📊 Current Market Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{crypto1} Price", f"${current_price1:.4f}")
        with col2:
            st.metric(f"{crypto2} Price", f"${current_price2:.4f}")
        with col3:
            zscore_delta = "Strong Signal" if abs(current_zscore) >= trading_system.optimal_params['entry_zscore'] else "No Signal"
            st.metric("Current Z-Score", f"{current_zscore:.2f}", delta=zscore_delta)
        with col4:
            st.metric("Optimal Entry Level", f"±{trading_system.optimal_params['entry_zscore']:.1f}")
        
        # Trading signal logic
        signal_type = None
        if current_zscore <= -trading_system.optimal_params['entry_zscore']:
            signal_type = "LONG_SPREAD"
        elif current_zscore >= trading_system.optimal_params['entry_zscore']:
            signal_type = "SHORT_SPREAD"
        
        st.markdown("---")
        
        if signal_type:
            # Calculate position sizes
            position_value = capital * (risk_per_trade / 100) * leverage
            
            if trading_system.optimal_params['hedge_method'] == 'dollar_neutral':
                # Equal dollar amounts
                crypto1_amount = (position_value / 2) / current_price1
                crypto2_amount = (position_value / 2) / current_price2
            else:
                # Regression-based
                crypto1_amount = (position_value / 2) / current_price1
                crypto2_amount = crypto1_amount * abs(hedge_ratio)
            
            crypto1_value = crypto1_amount * current_price1
            crypto2_value = crypto2_amount * current_price2
            
            if signal_type == "LONG_SPREAD":
                st.markdown('<div class="signal-buy">🚀 LONG SPREAD SIGNAL ACTIVE</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"""
                    ### 🟢 BUY {crypto1}
                    **Action:** Market Buy Order
                    **Amount:** {crypto1_amount:.6f} {crypto1}
                    **Value:** ${crypto1_value:.2f} USDT
                    **Price:** ${current_price1:.4f}
                    """)
                
                with col2:
                    st.error(f"""
                    ### 🔴 SELL {crypto2} (Short)
                    **Action:** Market Sell (Futures)
                    **Amount:** {crypto2_amount:.6f} {crypto2}
                    **Value:** ${crypto2_value:.2f} USDT
                    **Price:** ${current_price2:.4f}
                    **Leverage:** {leverage}x
                    """)
            
            else:  # SHORT_SPREAD
                st.markdown('<div class="signal-sell">🔥 SHORT SPREAD SIGNAL ACTIVE</div>', 
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"""
                    ### 🔴 SELL {crypto1} (Short)
                    **Action:** Market Sell (Futures)
                    **Amount:** {crypto1_amount:.6f} {crypto1}
                    **Value:** ${crypto1_value:.2f} USDT
                    **Price:** ${current_price1:.4f}
                    **Leverage:** {leverage}x
                    """)
                
                with col2:
                    st.success(f"""
                    ### 🟢 BUY {crypto2}
                    **Action:** Market Buy Order
                    **Amount:** {crypto2_amount:.6f} {crypto2}
                    **Value:** ${crypto2_value:.2f} USDT
                    **Price:** ${current_price2:.4f}
                    """)
            
            # Risk management levels
            st.markdown("---")
            st.subheader("🛡️ Risk Management Levels")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"""
                **🎯 Take Profit**
                Exit when Z-Score reaches: **±{trading_system.optimal_params['exit_zscore']:.1f}**
                Expected profit: **{trading_system.optimal_params['take_profit_pct']}%**
                """)
            
            with col2:
                st.warning(f"""
                **🛑 Stop Loss**
                Exit if loss exceeds: **{trading_system.optimal_params['stop_loss_pct']}%**
                Max loss amount: **${(capital * risk_per_trade/100):.2f}**
                """)
            
            with col3:
                st.error(f"""
                **⏰ Time Exit**
                Close position after: **{timeframe_days} days**
                Reason: Prevent correlation breakdown
                """)
        
        else:
            st.markdown('<div class="no-signal">⏳ NO TRADING SIGNAL - WAIT FOR OPPORTUNITY</div>', 
                       unsafe_allow_html=True)
            
            distance_to_long = abs(current_zscore + trading_system.optimal_params['entry_zscore'])
            distance_to_short = abs(current_zscore - trading_system.optimal_params['entry_zscore'])
            
            st.info(f"""
            ### 📊 Signal Status
            **Current Z-Score:** {current_zscore:.2f}
            
            **Distance to Signals:**
            - 🟢 Long Signal: {distance_to_long:.2f} points away
            - 🔴 Short Signal: {distance_to_short:.2f} points away
            
            **Set Price Alerts:**
            - Alert when Z-Score ≤ -{trading_system.optimal_params['entry_zscore']:.1f}
            - Alert when Z-Score ≥ +{trading_system.optimal_params['entry_zscore']:.1f}
            """)
    
    else:
        st.warning("⚠️ Please complete the Pair Analysis & Optimization first!")

# Tab 3: Performance Analytics
with tab3:
    st.header("📈 Performance Analytics & Comparison")
    
    if hasattr(trading_system, 'best_performance') and trading_system.best_performance:
        # Performance metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Expected Return", 
                     f"{trading_system.best_performance['total_return']:.1f}%",
                     delta=f"vs Buy & Hold")
        
        with col2:
            st.metric("🎲 Win Rate", 
                     f"{trading_system.best_performance['win_rate']:.1f}%")
        
        with col3:
            st.metric("📊 Sharpe Ratio", 
                     f"{trading_system.best_performance['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("📉 Max Drawdown", 
                     f"{trading_system.best_performance['max_drawdown']:.1f}%")
        
        # Strategy vs Buy & Hold comparison
        st.subheader("⚖️ Strategy Performance vs Buy & Hold")
        
        if 'price1' in locals() and 'price2' in locals():
            # Calculate buy & hold returns
            crypto1_return = ((price1.iloc[-1] - price1.iloc[0]) / price1.iloc[0]) * 100
            crypto2_return = ((price2.iloc[-1] - price2.iloc[0]) / price2.iloc[0]) * 100
            portfolio_return = (crypto1_return + crypto2_return) / 2  # 50/50 portfolio
            
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                **🤖 AI Pairs Strategy**
                - Return: {trading_system.best_performance['total_return']:.1f}%
                - Trades: {trading_system.best_performance['num_trades']}
                - Win Rate: {trading_system.best_performance['win_rate']:.1f}%
                - Max DD: {trading_system.best_performance['max_drawdown']:.1f}%
                """)
            
            with col2:
                st.info(f"""
                **📈 Buy & Hold Portfolio**
                - Return: {portfolio_return:.1f}%
                - {crypto1}: {crypto1_return:.1f}%
                - {crypto2}: {crypto2_return:.1f}%
                - Max DD: ~30-50% (typical)
                """)
            
            with col3:
                outperformance = trading_system.best_performance['total_return'] - portfolio_return
                if outperformance > 0:
                    st.success(f"""
                    **🏆 Strategy Advantage**
                    - Outperforms by: {outperformance:.1f}%
                    - Lower volatility
                    - Market neutral exposure
                    - Controlled risk
                    """)
                else:
                    st.warning(f"""
                    **⚠️ Strategy Underperformance**
                    - Underperforms by: {abs(outperformance):.1f}%
                    - But lower risk
                    - Market neutral
                    - Consistent returns
                    """)
        
        # Portfolio equity curve
        if trading_system.best_performance.get('portfolio_values'):
            st.subheader("📊 Strategy Equity Curve")
            
            portfolio_values = trading_system.best_performance['portfolio_values']
            dates = pd.date_range(start='2023-01-01', periods=len(portfolio_values), freq='D')
            
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                name='AI Pairs Strategy',
                line=dict(color='green', width=3),
                fill='tonexty' if len(portfolio_values) > 1 else None,
                fillcolor='rgba(0,255,0,0.1)'
            ))
            
            # Add benchmark line (buy & hold)
            if 'portfolio_return' in locals():
                benchmark_values = [10000 * (1 + i * portfolio_return/100/len(portfolio_values)) 
                                  for i in range(len(portfolio_values))]
                fig_equity.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_values,
                    name='Buy & Hold Benchmark',
                    line=dict(color='blue', width=2, dash='dash')
                ))
            
            fig_equity.update_layout(
                title="Strategy Performance vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_equity, use_container_width=True)
        
        # Trade analysis
        if trading_system.best_performance.get('trades') is not None and not trading_system.best_performance['trades'].empty:
            st.subheader("📋 Detailed Trade Analysis")
            
            trades_df = trading_system.best_performance['trades']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L distribution
                fig_pnl = px.histogram(
                    trades_df, x='pnl_pct', nbins=20,
                    title="Trade P&L Distribution",
                    labels={'pnl_pct': 'P&L (%)', 'count': 'Number of Trades'},
                    color_discrete_sequence=['lightgreen']
                )
                fig_pnl.update_layout(height=400)
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Win/Loss by position type
                fig_pos = px.box(
                    trades_df, x='position_type', y='pnl_pct',
                    title="P&L by Position Type",
                    labels={'pnl_pct': 'P&L (%)', 'position_type': 'Position Type'}
                )
                fig_pos.update_layout(height=400)
                st.plotly_chart(fig_pos, use_container_width=True)
            
            # Trade statistics table
            st.subheader("📊 Trade Statistics")
            
            stats_summary = trades_df.groupby('position_type').agg({
                'pnl_pct': ['count', 'mean', 'std', 'min', 'max'],
                'days_held': 'mean'
            }).round(2)
            
            stats_summary.columns = ['Count', 'Avg P&L%', 'Std Dev', 'Min P&L%', 'Max P&L%', 'Avg Days']
            st.dataframe(stats_summary, use_container_width=True)
            
            # Recent trades table
            st.subheader("🔍 Recent Trade History")
            recent_trades = trades_df.tail(10)[
                ['entry_date', 'exit_date', 'position_type', 'entry_zscore', 
                 'exit_zscore', 'pnl_pct', 'days_held', 'exit_reason']
            ].round(2)
            
            # Color code the P&L
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: gray'
                    return color
                return ''
            
            styled_trades = recent_trades.style.applymap(color_pnl, subset=['pnl_pct'])
            st.dataframe(styled_trades, use_container_width=True)
        
        # Risk analysis
        st.subheader("⚠️ Risk Analysis & Warnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"""
            **🚨 High Risk Factors:**
            - Leverage up to {trading_system.optimal_params.get('leverage', 'N/A')}x amplifies losses
            - Crypto volatility can exceed 50% daily
            - Correlation breakdown risk in volatile markets
            - 24/7 market with no circuit breakers
            - Liquidation risk with high leverage
            """)
        
        with col2:
            st.info(f"""
            **🛡️ Risk Mitigation:**
            - Max {trading_system.optimal_params.get('stop_loss_pct', 'N/A')}% stop loss per trade
            - Position sizing based on risk tolerance
            - Market neutral strategy reduces directional risk
            - Time-based exits prevent correlation breakdown
            - AI optimization for optimal parameters
            """)
        
        # Performance summary
        st.markdown("---")
        st.subheader("🎯 Strategy Summary")
        
        summary_text = f"""
        ## 🤖 AI-Optimized Pairs Trading Strategy Summary
        
        **Pair:** {crypto1}/{crypto2}  
        **Timeframe:** {timeframe_days} days per trade  
        **Expected Annual Return:** {trading_system.best_performance['total_return']:.1f}%  
        **Win Rate:** {trading_system.best_performance['win_rate']:.1f}%  
        **Sharpe Ratio:** {trading_system.best_performance['sharpe_ratio']:.2f}  
        **Maximum Drawdown:** {trading_system.best_performance['max_drawdown']:.1f}%  
        
        **Optimal Parameters (AI-Selected):**
        - Entry Z-Score: ±{trading_system.optimal_params.get('entry_zscore', 'N/A')}
        - Exit Z-Score: ±{trading_system.optimal_params.get('exit_zscore', 'N/A')}
        - Recommended Leverage: {trading_system.optimal_params.get('leverage', 'N/A')}x
        - Stop Loss: {trading_system.optimal_params.get('stop_loss_pct', 'N/A')}%
        - Take Profit: {trading_system.optimal_params.get('take_profit_pct', 'N/A')}%
        - Hedge Method: {trading_system.optimal_params.get('hedge_method', 'N/A').replace('_', ' ').title()}
        
        **Risk Warning:** Past performance does not guarantee future results. 
        Cryptocurrency trading involves substantial risk of loss.
        """
        
        st.markdown(summary_text)
    
    else:
        st.warning("⚠️ Please complete the Pair Analysis & Optimization first to see performance analytics!")

# Sidebar with quick stats and alerts
with st.sidebar:
    st.header("🚀 Quick Stats")
    
    if hasattr(trading_system, 'optimal_params') and trading_system.optimal_params:
        st.success(f"""
        **Current Pair:** {crypto1}/{crypto2}
        **Status:** Optimized ✅
        **Expected Return:** {trading_system.best_performance['total_return']:.1f}%
        **Win Rate:** {trading_system.best_performance['win_rate']:.1f}%
        """)
        
        # Current signal status
        if 'current_zscore' in locals():
            if abs(current_zscore) >= trading_system.optimal_params['entry_zscore']:
                signal_emoji = "🚨" if current_zscore > 0 else "🟢"
                signal_text = "SHORT SIGNAL" if current_zscore > 0 else "LONG SIGNAL"
                st.error(f"{signal_emoji} {signal_text}")
            else:
                st.info("⏳ No Signal")
    
