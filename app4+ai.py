import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #000 0%, #111 50%, #000 100%);
        padding: 20px;
        border-radius: 5px;
        border: 2px solid #00ff41;
        color: #00ff41;
        text-align: center;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #000;
        border: 1px solid #00ff41;
        padding: 15px;
        border-radius: 3px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        text-align: center;
    }
    .signal-active {
        background: #001100;
        border: 2px solid #00ff41;
        padding: 20px;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 20px #00ff41;
    }
    .signal-inactive {
        background: #111;
        border: 1px solid #333;
        padding: 20px;
        color: #888;
        font-family: 'Courier New', monospace;
    }
    .position-box {
        background: #000;
        border: 1px solid #333;
        padding: 15px;
        margin: 10px 0;
        color: #00ff41;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Crypto tickers from original code
CRYPTO_TICKERS = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD', 
    'Binance Coin': 'BNB-USD',
    'Cardano': 'ADA-USD',
    'Solana': 'SOL-USD',
    'XRP': 'XRP-USD',
    'Polkadot': 'DOT-USD',
    'Dogecoin': 'DOGE-USD',
    'Avalanche': 'AVAX-USD',
    'Chainlink': 'LINK-USD',
    'Polygon': 'MATIC-USD',
    'Litecoin': 'LTC-USD',
    'Bitcoin Cash': 'BCH-USD',
    'Stellar': 'XLM-USD',
    'VeChain': 'VET-USD'
}

class ProfitMaximizer:
    def __init__(self):
        self.optimal_params = {}
        self.current_data = {}
        
    @st.cache_data(ttl=300)
    def load_data(_self, symbol, period='1y'):
        try:
            ticker = CRYPTO_TICKERS.get(symbol, symbol)
            data = yf.download(ticker, period=period, progress=False)
            if data.empty:
                return pd.Series(dtype=float)
            return data['Close'].dropna() if 'Close' in data.columns else data.iloc[:, -1].dropna()
        except Exception as e:
            st.error(f"Data error for {symbol}: {str(e)}")
            return pd.Series(dtype=float)
    
    def calculate_correlation(self, price1, price2):
        ret1 = price1.pct_change().dropna()
        ret2 = price2.pct_change().dropna()
        ret1, ret2 = ret1.align(ret2, join='inner')
        return ret1.corr(ret2)
    
    def find_optimal_params(self, price1, price2):
        """Find optimal z-score parameters for maximum profit"""
        best_return = -999
        best_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
        
        # Parameter grid search
        entry_range = np.arange(1.0, 4.0, 0.2)
        exit_range = np.arange(0.2, 1.5, 0.2) 
        stop_range = np.arange(2.5, 5.0, 0.3)
        
        for entry in entry_range:
            for exit in exit_range:
                for stop in stop_range:
                    if stop > entry and exit < entry:
                        signals = self.generate_signals(price1, price2, entry, exit, stop)
                        backtest = self.backtest_signals(signals, price1, price2)
                        
                        if backtest['total_return'] > best_return:
                            best_return = backtest['total_return']
                            best_params = {'entry': entry, 'exit': exit, 'stop': stop}
        
        return best_params, best_return
    
    def calculate_spread_zscore(self, price1, price2, lookback=60):
        """Calculate spread and z-score"""
        # Align prices
        price1, price2 = price1.align(price2, join='inner')
        
        if len(price1) < lookback + 20:
            return pd.DataFrame(), 1.0
        
        # Calculate hedge ratio
        X = price2.iloc[-lookback:].values.reshape(-1, 1)
        y = price1.iloc[-lookback:].values
        model = LinearRegression().fit(X, y)
        hedge_ratio = float(model.coef_[0])
        
        # Calculate spread
        spread = price1 - hedge_ratio * price2
        
        # Calculate rolling z-score
        rolling_mean = spread.rolling(window=20).mean()
        rolling_std = spread.rolling(window=20).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        df = pd.DataFrame({
            'price1': price1,
            'price2': price2, 
            'spread': spread,
            'zscore': zscore.fillna(0)
        }, index=price1.index)
        
        return df, hedge_ratio
    
    def generate_signals(self, price1, price2, entry_z=2.0, exit_z=0.5, stop_z=3.5):
        """Generate trading signals based on z-score"""
        df, hedge_ratio = self.calculate_spread_zscore(price1, price2)
        
        if df.empty:
            return pd.DataFrame()
        
        signals = []
        position = 0  # 0=flat, 1=long spread, -1=short spread
        
        for i in range(len(df)):
            row = df.iloc[i]
            zscore = row['zscore']
            
            signal = {
                'date': row.name,
                'zscore': zscore,
                'price1': row['price1'],
                'price2': row['price2'], 
                'spread': row['spread'],
                'position': position,
                'action': 'HOLD'
            }
            
            # Position logic
            if position == 0:  # No position
                if zscore <= -entry_z:
                    signal['action'] = 'ENTER_LONG'
                    position = 1
                elif zscore >= entry_z:
                    signal['action'] = 'ENTER_SHORT'
                    position = -1
            
            elif position == 1:  # Long spread
                if zscore >= -exit_z:
                    signal['action'] = 'EXIT_LONG'
                    position = 0
                elif zscore <= -stop_z:
                    signal['action'] = 'STOP_LONG'
                    position = 0
            
            elif position == -1:  # Short spread
                if zscore <= exit_z:
                    signal['action'] = 'EXIT_SHORT'
                    position = 0
                elif zscore >= stop_z:
                    signal['action'] = 'STOP_SHORT'
                    position = 0
            
            signal['position'] = position
            signals.append(signal)
        
        return pd.DataFrame(signals)
    
    def backtest_signals(self, signals, price1, price2):
        """Backtest the trading signals"""
        if signals.empty:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0, 'sharpe': 0}
        
        trades = []
        current_position = None
        
        for i, row in signals.iterrows():
            if row['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                current_position = {
                    'entry_price': row['spread'],
                    'entry_date': row['date'],
                    'position_type': row['action']
                }
            
            elif row['action'] in ['EXIT_LONG', 'EXIT_SHORT', 'STOP_LONG', 'STOP_SHORT']:
                if current_position:
                    entry_spread = current_position['entry_price']
                    exit_spread = row['spread']
                    
                    if current_position['position_type'] == 'ENTER_LONG':
                        pnl = exit_spread - entry_spread
                    else:
                        pnl = entry_spread - exit_spread
                    
                    trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': row['date'],
                        'pnl': pnl,
                        'entry_price': entry_spread,
                        'exit_price': exit_spread
                    })
                    current_position = None
        
        if not trades:
            return {'total_return': 0, 'win_rate': 0, 'num_trades': 0, 'sharpe': 0}
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = trades_df['pnl'].sum() * 100  # Percentage return
        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(win_trades) / len(trades_df) * 100
        
        returns = trades_df['pnl']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades_df),
            'sharpe': sharpe,
            'trades': trades_df
        }
    
    def calculate_position_sizes(self, signal, capital, leverage):
        """Calculate exact position sizes for both assets"""
        if signal['action'] not in ['ENTER_LONG', 'ENTER_SHORT']:
            return None
        
        price1 = signal['price1']
        price2 = signal['price2']
        
        total_capital = capital * leverage
        allocation_per_side = total_capital / 2
        
        if signal['action'] == 'ENTER_LONG':
            # Long spread: Buy asset1, Short asset2
            asset1_qty = allocation_per_side / price1
            asset2_qty = allocation_per_side / price2
            
            return {
                'asset1_action': 'BUY',
                'asset1_qty': asset1_qty,
                'asset1_price': price1,
                'asset1_value': asset1_qty * price1,
                'asset2_action': 'SHORT',
                'asset2_qty': asset2_qty,
                'asset2_price': price2,
                'asset2_value': asset2_qty * price2
            }
        else:
            # Short spread: Short asset1, Buy asset2
            asset1_qty = allocation_per_side / price1
            asset2_qty = allocation_per_side / price2
            
            return {
                'asset1_action': 'SHORT',
                'asset1_qty': asset1_qty,
                'asset1_price': price1,
                'asset1_value': asset1_qty * price1,
                'asset2_action': 'BUY',
                'asset2_qty': asset2_qty,
                'asset2_price': price2,
                'asset2_value': asset2_qty * price2
            }

# Initialize system
st.set_page_config(page_title="Crypto Pairs Trading", layout="wide")

if 'trading_system' not in st.session_state:
    st.session_state.trading_system = ProfitMaximizer()
    st.session_state.saved_pairs = []

system = st.session_state.trading_system

# Header
st.markdown('<div class="main-header"><h1>CRYPTO PAIRS TRADING SYSTEM</h1><p>Advanced Correlation • Live Signals • Maximum Profit</p></div>', unsafe_allow_html=True)

# Controls
col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 2])

with col1:
    crypto1 = st.selectbox("ASSET A:", list(CRYPTO_TICKERS.keys()), key='asset1')

with col2:
    available_cryptos = [c for c in CRYPTO_TICKERS.keys() if c != crypto1]
    crypto2 = st.selectbox("ASSET B:", available_cryptos, key='asset2')

with col3:
    capital = st.number_input("CAPITAL ($)", value=10000, min_value=1000, step=1000)

with col4:
    leverage = st.number_input("LEVERAGE", value=3.0, min_value=1.0, max_value=10.0, step=0.5)

with col5:
    period = st.selectbox("PERIOD", ["6mo", "1y", "2y"], index=1)

with col6:
    col6a, col6b = st.columns(2)
    with col6a:
        analyze_btn = st.button("ANALYZE", type="primary", use_container_width=True)
    with col6b:
        optimize_btn = st.button("OPTIMIZE", use_container_width=True)

# Analysis
if analyze_btn or optimize_btn:
    with st.spinner("Loading data..."):
        price1 = system.load_data(crypto1, period)
        price2 = system.load_data(crypto2, period)
    
    if not price1.empty and not price2.empty and len(price1) > 100:
        
        # Calculate correlation
        correlation = system.calculate_correlation(price1, price2)
        
        if optimize_btn:
            with st.spinner("Optimizing parameters..."):
                optimal_params, best_return = system.find_optimal_params(price1, price2)
        else:
            optimal_params = {'entry': 2.0, 'exit': 0.5, 'stop': 3.5}
        
        # Generate signals with optimal parameters
        signals = system.generate_signals(price1, price2, **optimal_params)
        backtest = system.backtest_signals(signals, price1, price2)
        
        # Calculate current z-score
        df, hedge_ratio = system.calculate_spread_zscore(price1, price2)
        current_zscore = df['zscore'].iloc[-1] if not df.empty else 0
        
        # Store data
        system.current_data = {
            'crypto1': crypto1, 'crypto2': crypto2,
            'price1': price1, 'price2': price2,
            'df': df, 'signals': signals,
            'correlation': correlation, 'current_zscore': current_zscore,
            'backtest': backtest, 'optimal_params': optimal_params,
            'hedge_ratio': hedge_ratio
        }
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            corr_color = "positive" if correlation > 0.7 else "negative" if correlation < 0.3 else "neutral"
            st.metric("CORRELATION", f"{correlation:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            zscore_color = "positive" if abs(current_zscore) > 2 else "neutral"
            st.metric("Z-SCORE", f"{current_zscore:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            return_color = "positive" if backtest['total_return'] > 0 else "negative"
            st.metric("BACKTEST RETURN", f"{backtest['total_return']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("WIN RATE", f"{backtest['win_rate']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("SHARPE RATIO", f"{backtest['sharpe']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Signal panel
        latest_signal = signals.iloc[-1] if not signals.empty else None
        
        if latest_signal and latest_signal['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
            st.markdown('<div class="signal-active">', unsafe_allow_html=True)
            st.markdown(f"### ACTIVE SIGNAL: {latest_signal['action']}")
            
            position_info = system.calculate_position_sizes(latest_signal, capital, leverage)
            
            if position_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **{crypto1}:** {position_info['asset1_action']} {position_info['asset1_qty']:.6f}  
                    **Price:** ${position_info['asset1_price']:.4f}  
                    **Value:** ${position_info['asset1_value']:.2f}  
                    """)
                
                with col2:
                    st.markdown(f"""
                    **{crypto2}:** {position_info['asset2_action']} {position_info['asset2_qty']:.6f}  
                    **Price:** ${position_info['asset2_price']:.4f}  
                    **Value:** ${position_info['asset2_value']:.2f}  
                    """)
                
                st.markdown(f"""
                **ENTRY Z-SCORE:** {latest_signal['zscore']:.2f}  
                **EXIT TARGET:** {optimal_params['exit']:.1f}  
                **STOP LOSS:** {optimal_params['stop']:.1f}  
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown('<div class="signal-inactive">', unsafe_allow_html=True)
            st.markdown("### NO ACTIVE SIGNAL")
            st.markdown(f"Current Z-Score: {current_zscore:.2f}")
            st.markdown(f"Entry threshold: ±{optimal_params['entry']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{crypto1} vs {crypto2} - Normalized Prices', 'Z-Score & Signals'],
            vertical_spacing=0.1
        )
        
        # Normalize prices
        norm_price1 = (price1 / price1.iloc[0]) * 100
        norm_price2 = (price2 / price2.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(x=price1.index, y=norm_price1, name=crypto1, line=dict(color='#00ff41')), row=1, col=1)
        fig.add_trace(go.Scatter(x=price2.index, y=norm_price2, name=crypto2, line=dict(color='#ff4444')), row=1, col=1)
        
        # Z-score chart
        if not df.empty:
            fig.add_trace(go.Scatter(x=df.index, y=df['zscore'], name='Z-Score', line=dict(color='#00ffff')), row=2, col=1)
            fig.add_hline(y=optimal_params['entry'], line_dash="dash", line_color="#ff4444", row=2, col=1)
            fig.add_hline(y=-optimal_params['entry'], line_dash="dash", line_color="#00ff41", row=2, col=1)
            fig.add_hline(y=0, line_color="#666666", row=2, col=1)
        
        fig.update_layout(height=600, plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='#00ff41')
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade history
        if 'trades' in backtest and not backtest['trades'].empty:
            st.markdown("### TRADE HISTORY")
            trades = backtest['trades'].copy()
            trades['entry_date'] = trades['entry_date'].dt.strftime('%Y-%m-%d')
            trades['exit_date'] = trades['exit_date'].dt.strftime('%Y-%m-%d')
            trades = trades.round(6)
            st.dataframe(trades, use_container_width=True, hide_index=True)
        
        # Save pair functionality
        if st.button("SAVE PAIR", use_container_width=True):
            pair_name = f"{crypto1}/{crypto2}"
            pair_data = {
                'name': pair_name,
                'crypto1': crypto1,
                'crypto2': crypto2,
                'correlation': correlation,
                'backtest_return': backtest['total_return'],
                'optimal_params': optimal_params
            }
            
            # Check if already saved
            existing = [p for p in st.session_state.saved_pairs if p['name'] == pair_name]
            if not existing:
                st.session_state.saved_pairs.append(pair_data)
                st.success(f"Saved pair: {pair_name}")
            else:
                st.warning("Pair already saved")

# Saved pairs section
if st.session_state.saved_pairs:
    st.markdown("### SAVED PAIRS")
    
    for pair in st.session_state.saved_pairs:
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{pair['name']}**")
        with col2:
            st.markdown(f"Corr: {pair['correlation']:.3f}")
        with col3:
            st.markdown(f"Return: {pair['backtest_return']:.1f}%")
        with col4:
            if st.button(f"Load", key=f"load_{pair['name']}"):
                # Set the selectboxes to this pair
                st.session_state['asset1'] = pair['crypto1']
                st.session_state['asset2'] = pair['crypto2']
                st.rerun()
        with col5:
            if st.button(f"Delete", key=f"del_{pair['name']}"):
                st.session_state.saved_pairs = [p for p in st.session_state.saved_pairs if p['name'] != pair['name']]
                st.rerun()

# Current status in sidebar
with st.sidebar:
    st.markdown("### SYSTEM STATUS")
    
    if hasattr(system, 'current_data') and system.current_data:
        st.success(f"ACTIVE: {system.current_data['crypto1']}/{system.current_data['crypto2']}")
        st.metric("Correlation", f"{system.current_data['correlation']:.3f}")
        st.metric("Z-Score", f"{system.current_data['current_zscore']:.2f}")
        st.metric("Backtest Return", f"{system.current_data['backtest']['total_return']:.1f}%")
        
        params = system.current_data['optimal_params']
        st.markdown(f"""
        **PARAMETERS:**  
        Entry: ±{params['entry']:.1f}  
        Exit: ±{params['exit']:.1f}  
        Stop: ±{params['stop']:.1f}  
        """)
    else:
        st.info("No active pair")
    
    st.markdown("---")
    st.markdown("### QUICK STATS")
    st.metric("Saved Pairs", len(st.session_state.saved_pairs))
    st.metric("Capital", f"${capital:,}")
    st.metric("Leverage", f"{leverage}x")
