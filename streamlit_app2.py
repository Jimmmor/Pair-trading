import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pairs Trading Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-long {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .signal-short {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .signal-exit {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .signal-none {
        background-color: #e2e3e5;
        color: #383d41;
        border: 2px solid #d6d8db;
    }
    .stoploss-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stoploss-level {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Pairs Trading Monitor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p>Vergelijk twee crypto assets, bereken de spread, Z-score en statistieken (alpha, beta, R², Pearson R).</p>
    <p>Gebruik dit voor pairs trading met geavanceerde analyse en aanbevelingen.</p>
</div>
""", unsafe_allow_html=True)

# Available tickers
tickers = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Solana (SOL)": "SOL-USD",
    "XRP (XRP)": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Shiba Inu (SHIB)": "SHIB-USD",
    "TRON (TRX)": "TRX-USD",
    "Uniswap (UNI)": "UNI-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Stellar (XLM)": "XLM-USD",
    "VeChain (VET)": "VET-USD",
    "NEAR Protocol (NEAR)": "NEAR-USD",
    "Aptos (APT)": "APT-USD",
    "Filecoin (FIL)": "FIL-USD",
    "The Graph (GRT)": "GRT-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "Tezos (XTZ)": "XTZ-USD",
    "Hedera (HBAR)": "HBAR-USD",
    "Fantom (FTM)": "FTM-USD",
    "EOS (EOS)": "EOS-USD",
    "Zcash (ZEC)": "ZEC-USD",
    "Dash (DASH)": "DASH-USD",
    "Chiliz (CHZ)": "CHZ-USD",
    "THETA (THETA)": "THETA-USD",
    "Internet Computer (ICP)": "ICP-USD",
    "Arbitrum (ARB)": "ARB-USD",
    "Optimism (OP)": "OP-USD",
    "Injective (INJ)": "INJ-USD",
    "SUI (SUI)": "SUI-USD",
    "Lido DAO (LDO)": "LDO-USD",
    "Aave (AAVE)": "AAVE-USD",
    "Maker (MKR)": "MKR-USD",
    "Curve DAO (CRV)": "CRV-USD",
    "1inch (1INCH)": "1INCH-USD",
    "Gala (GALA)": "GALA-USD",
    "Render (RNDR)": "RNDR-USD"
}

# Enhanced sidebar with expanders
with st.sidebar:
    st.header("🔍 Trading Configuration")
    
    # Asset selection
    with st.expander("🏷️ Asset Selection", expanded=True):
        name1 = st.selectbox("🥇 Primary Asset", list(tickers.keys()), index=0)
        remaining = [k for k in tickers.keys() if k != name1]
        name2 = st.selectbox("🥈 Secondary Asset", remaining, index=1)
    
    # Time period settings
    with st.expander("⏰ Time Settings", expanded=True):
        periode = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval_options = ["1d"] if periode in ["1y", "2y"] else ["1d", "1h", "30m"]
        interval = st.selectbox("Data Interval", interval_options, index=0)
    
    # Analysis parameters
    with st.expander("📊 Analysis Parameters", expanded=True):
        corr_window = st.slider("Rolling correlation window (days)", 
                               min_value=5, max_value=90, value=20, step=5)
        zscore_entry_threshold = st.slider("Z-score entry threshold", 
                                          min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        zscore_exit_threshold = st.slider("Z-score exit threshold", 
                                         min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    
    # Stop loss settings
    with st.expander("🛑 Risk Management", expanded=True):
        stoploss_pct = st.slider("Stop loss percentage", 
                                min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100
        volatility_window = st.slider("Volatility calculation window", 
                                    min_value=5, max_value=50, value=20, step=5)
        volatility_multiplier = st.slider("Volatility multiplier", 
                                        min_value=1.0, max_value=5.0, value=2.5, step=0.1)
        atr_period = st.slider("ATR period", 
                             min_value=5, max_value=30, value=14, step=1)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        confidence_interval = st.slider("Confidence interval (%)", 
                                       min_value=90, max_value=99, value=95, step=1)
        lookback_period = st.slider("Lookback period for signals", 
                                   min_value=1, max_value=10, value=3, step=1)
        extreme_zscore = st.slider("Extreme Z-score threshold", 
                                 min_value=3.0, max_value=6.0, value=4.0, step=0.1)
        min_correlation = st.slider("Minimum correlation threshold", 
                                  min_value=0.1, max_value=0.8, value=0.3, step=0.05)
        max_holding_days = st.slider("Maximum holding period (days)", 
                                   min_value=1, max_value=90, value=30, step=1)

# Convert names to ticker symbols
coin1 = tickers[name1]
coin2 = tickers[name2]

# Data loading functions
@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

def process_data(data1, data2):
    # Handle multi-level columns
    if isinstance(data1.columns, pd.MultiIndex):
        df1 = data1['Close'].iloc[:, 0].dropna()
        df2 = data2['Close'].iloc[:, 0].dropna()
        vol1 = data1['Volume'].iloc[:, 0]
        vol2 = data2['Volume'].iloc[:, 0]
    else:
        df1 = data1['Close'].dropna()
        df2 = data2['Close'].dropna()
        vol1 = data1['Volume']
        vol2 = data2['Volume']
    
    # Ensure we have Series
    if not isinstance(df1, pd.Series):
        df1 = pd.Series(df1)
    if not isinstance(df2, pd.Series):
        df2 = pd.Series(df2)
    
    # Align data
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    vol1_aligned, vol2_aligned = vol1.align(vol2, join='inner')
    
    # Create DataFrame
    df = pd.DataFrame({
        'price1': df1_aligned,
        'price2': df2_aligned,
        'volume1': vol1_aligned,
        'volume2': vol2_aligned
    }).dropna()
    
    return df

def calculate_enhanced_stats(df):
    # Linear regression
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]
    r_squared = model.score(X, y)
    
    # Calculate spread
    df['spread'] = df['price2'] - (alpha + beta * df['price1'])
    
    # Spread statistics
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    spread_skew = df['spread'].skew()
    spread_kurtosis = df['spread'].kurtosis()
    
    # Z-score calculation
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    # Rolling statistics
    df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
    df['rolling_mean'] = df['spread'].rolling(window=corr_window).mean()
    df['rolling_std'] = df['spread'].rolling(window=corr_window).std()
    df['rolling_zscore'] = (df['spread'] - df['rolling_mean']) / df['rolling_std']
    
    # Calculate returns
    df['returns1'] = df['price1'].pct_change()
    df['returns2'] = df['price2'].pct_change()
    
    # Additional statistics
    pearson_corr = df['price1'].corr(df['price2'])
    spearman_corr = df['price1'].corr(df['price2'], method='spearman')
    
    # Volatility metrics
    vol1 = df['returns1'].std() * np.sqrt(252)
    vol2 = df['returns2'].std() * np.sqrt(252)
    vol_ratio = vol2 / vol1
    
    # FIXED: Half-life of mean reversion
    y_lag = df['spread'].shift(1)
    y_diff = df['spread'].diff()
    
    # Remove NaN values and ensure same length
    mask = ~(y_lag.isna() | y_diff.isna())
    y_lag_clean = y_lag[mask]
    y_diff_clean = y_diff[mask]
    
    if len(y_lag_clean) > 0:
        half_life_model = LinearRegression()
        half_life_model.fit(y_lag_clean.values.reshape(-1, 1), y_diff_clean.values)
        half_life = -np.log(2) / half_life_model.coef_[0] if half_life_model.coef_[0] != 0 else np.inf
    else:
        half_life = np.inf
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'spread_skew': spread_skew,
        'spread_kurtosis': spread_kurtosis,
        'vol1': vol1,
        'vol2': vol2,
        'vol_ratio': vol_ratio,
        'half_life': half_life
    }

def calculate_dynamic_stoploss(df, stoploss_params):
    """
    Calculate dynamic stop loss levels based on volatility and ATR
    """
    # Extract parameters
    stoploss_pct = stoploss_params.get('stoploss_pct', 0.05)
    volatility_window = stoploss_params.get('volatility_window', 20)
    volatility_multiplier = stoploss_params.get('volatility_multiplier', 2.5)
    atr_period = stoploss_params.get('atr_period', 14)
    
    # Calculate ATR (Average True Range) for both assets
    def calculate_atr(df, period=14):
        high_low = df['price1'].rolling(window=period).max() - df['price1'].rolling(window=period).min()
        high_close = np.abs(df['price1'] - df['price1'].shift(1))
        low_close = np.abs(df['price1'].rolling(window=period).min() - df['price1'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    # Calculate rolling volatility
    df['volatility'] = df['spread'].rolling(window=volatility_window).std()
    
    # Calculate ATR-based stop loss
    atr = calculate_atr(df, atr_period)
    
    # Dynamic stop loss levels
    df['stoploss_upper'] = df['spread'] + (df['volatility'] * volatility_multiplier)
    df['stoploss_lower'] = df['spread'] - (df['volatility'] * volatility_multiplier)
    
    # Alternative ATR-based stop loss
    df['atr_stoploss_upper'] = df['spread'] + (atr * volatility_multiplier)
    df['atr_stoploss_lower'] = df['spread'] - (atr * volatility_multiplier)
    
    # Percentage-based stop loss
    df['pct_stoploss_upper'] = df['spread'] * (1 + stoploss_pct)
    df['pct_stoploss_lower'] = df['spread'] * (1 - stoploss_pct)
    
    return df

def generate_signals(df, params):
    """
    Generate trading signals based on Z-score and additional filters
    """
    zscore_entry = params.get('zscore_entry_threshold', 2.0)
    zscore_exit = params.get('zscore_exit_threshold', 0.5)
    min_correlation = params.get('min_correlation', 0.3)
    extreme_zscore = params.get('extreme_zscore', 4.0)
    lookback_period = params.get('lookback_period', 3)
    
    # Initialize signals
    df['signal'] = 0  # 0: No signal, 1: Long, -1: Short, 2: Exit
    df['signal_strength'] = 0.0
    df['signal_type'] = 'NONE'
    
    # Generate basic signals
    for i in range(len(df)):
        zscore = df.iloc[i]['zscore']
        rolling_zscore = df.iloc[i]['rolling_zscore']
        correlation = df.iloc[i]['rolling_corr']
        
        # Skip if correlation is too low
        if pd.isna(correlation) or abs(correlation) < min_correlation:
            continue
        
        # Use rolling Z-score if available, otherwise use static Z-score
        current_zscore = rolling_zscore if not pd.isna(rolling_zscore) else zscore
        
        # Signal generation logic
        if current_zscore > zscore_entry:
            # Short signal (spread is high, expect reversion)
            df.iloc[i, df.columns.get_loc('signal')] = -1
            df.iloc[i, df.columns.get_loc('signal_strength')] = min(abs(current_zscore) / zscore_entry, 3.0)
            df.iloc[i, df.columns.get_loc('signal_type')] = 'SHORT'
            
        elif current_zscore < -zscore_entry:
            # Long signal (spread is low, expect reversion)
            df.iloc[i, df.columns.get_loc('signal')] = 1
            df.iloc[i, df.columns.get_loc('signal_strength')] = min(abs(current_zscore) / zscore_entry, 3.0)
            df.iloc[i, df.columns.get_loc('signal_type')] = 'LONG'
            
        elif abs(current_zscore) < zscore_exit:
            # Exit signal (spread near mean)
            df.iloc[i, df.columns.get_loc('signal')] = 2
            df.iloc[i, df.columns.get_loc('signal_strength')] = 1.0 - abs(current_zscore) / zscore_exit
            df.iloc[i, df.columns.get_loc('signal_type')] = 'EXIT'
        
        # Extreme Z-score warning
        if abs(current_zscore) > extreme_zscore:
            df.iloc[i, df.columns.get_loc('signal_type')] = 'EXTREME'
    
    # Add signal confirmation based on lookback period
    df['signal_confirmed'] = False
    for i in range(lookback_period, len(df)):
        if df.iloc[i]['signal'] != 0:
            # Check if signal is consistent over lookback period
            recent_signals = df.iloc[i-lookback_period:i]['signal'].values
            if len(recent_signals) > 0 and all(s == df.iloc[i]['signal'] or s == 0 for s in recent_signals):
                df.iloc[i, df.columns.get_loc('signal_confirmed')] = True
    
    return df

def calculate_position_sizing(df, params, current_price1, current_price2):
    """
    Calculate optimal position sizing based on volatility and correlation
    """
    # Get current statistics
    current_vol1 = df['returns1'].tail(20).std() * np.sqrt(252)
    current_vol2 = df['returns2'].tail(20).std() * np.sqrt(252)
    current_corr = df['rolling_corr'].iloc[-1]
    
    # Risk parameters
    max_position_size = params.get('max_position_size', 10000)  # USD
    risk_per_trade = params.get('risk_per_trade', 0.02)  # 2% of portfolio
    
    # Calculate hedge ratio (beta)
    beta = df['returns2'].cov(df['returns1']) / df['returns1'].var()
    
    # Position sizing based on volatility
    vol_adjusted_size = max_position_size / (current_vol1 + current_vol2)
    
    # Calculate number of shares/units
    shares_asset1 = vol_adjusted_size / current_price1
    shares_asset2 = vol_adjusted_size / current_price2 * beta
    
    return {
        'shares_asset1': shares_asset1,
        'shares_asset2': shares_asset2,
        'hedge_ratio': beta,
        'position_value': vol_adjusted_size,
        'risk_estimate': vol_adjusted_size * risk_per_trade
    }

def create_enhanced_plots(df, stats, coin1, coin2):
    """
    Create comprehensive plots for pairs trading analysis
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            f'{coin1} vs {coin2} Prices',
            'Spread & Z-Score',
            'Rolling Correlation',
            'Signal Strength',
            'Price Ratio',
            'Volume Analysis',
            'Stop Loss Levels',
            'Returns Distribution'
        ],
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Plot 1: Price comparison
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price1'], name=coin1, line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price2'], name=coin2, line=dict(color='red', width=2)),
        row=1, col=1, secondary_y=True
    )
    
    # Plot 2: Spread and Z-score
    fig.add_trace(
        go.Scatter(x=df.index, y=df['spread'], name='Spread', line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['zscore'], name='Z-Score', line=dict(color='purple', width=2)),
        row=1, col=2, secondary_y=True
    )
    
    # Add Z-score thresholds
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=1, col=2, secondary_y=True)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", row=1, col=2, secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2, secondary_y=True)
    
    # Plot 3: Rolling correlation
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rolling_corr'], name='Rolling Correlation', 
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", row=2, col=1)
    
    # Plot 4: Signal strength
    signal_colors = ['red' if s == -1 else 'green' if s == 1 else 'yellow' if s == 2 else 'gray' 
                     for s in df['signal']]
    fig.add_trace(
        go.Scatter(x=df.index, y=df['signal_strength'], name='Signal Strength',
                  mode='markers', marker=dict(color=signal_colors, size=8)),
        row=2, col=2
    )
    
    # Plot 5: Price ratio
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price1']/df['price2'], name='Price Ratio',
                  line=dict(color='navy', width=2)),
        row=3, col=1
    )
    
    # Plot 6: Volume analysis
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volume1'], name=f'{coin1} Volume',
                  line=dict(color='lightblue', width=1)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volume2'], name=f'{coin2} Volume',
                  line=dict(color='lightcoral', width=1)),
        row=3, col=2, secondary_y=True
    )
    
    # Plot 7: Stop loss levels
    if 'stoploss_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['stoploss_upper'], name='Stop Loss Upper',
                      line=dict(color='red', width=1, dash='dash')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['stoploss_lower'], name='Stop Loss Lower',
                      line=dict(color='red', width=1, dash='dash')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['spread'], name='Spread',
                      line=dict(color='black', width=2)),
            row=4, col=1
        )
    
    # Plot 8: Returns distribution
    fig.add_trace(
        go.Histogram(x=df['returns1'].dropna(), name=f'{coin1} Returns',
                    opacity=0.7, nbinsx=50),
        row=4, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['returns2'].dropna(), name=f'{coin2} Returns',
                    opacity=0.7, nbinsx=50),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Pairs Trading Analysis Dashboard",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def display_trading_signals(df, coin1, coin2):
    """
    Display current trading signals and recommendations
    """
    if df.empty:
        return
    
    # Get latest signal
    latest_signal = df.iloc[-1]
    zscore = latest_signal['zscore']
    rolling_zscore = latest_signal.get('rolling_zscore', zscore)
    signal_type = latest_signal.get('signal_type', 'NONE')
    signal_strength = latest_signal.get('signal_strength', 0)
    correlation = latest_signal.get('rolling_corr', 0)
    
    # Create signal display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Current Signal")
        if signal_type == 'LONG':
            st.markdown(f'<div class="signal-box signal-long">LONG SIGNAL<br/>Strength: {signal_strength:.2f}</div>', 
                       unsafe_allow_html=True)
            st.write(f"📈 **Action**: Long {coin1}, Short {coin2}")
        elif signal_type == 'SHORT':
            st.markdown(f'<div class="signal-box signal-short">SHORT SIGNAL<br/>Strength: {signal_strength:.2f}</div>', 
                       unsafe_allow_html=True)
            st.write(f"📉 **Action**: Short {coin1}, Long {coin2}")
        elif signal_type == 'EXIT':
            st.markdown(f'<div class="signal-box signal-exit">EXIT SIGNAL<br/>Strength: {signal_strength:.2f}</div>', 
                       unsafe_allow_html=True)
            st.write("🚪 **Action**: Close positions")
        elif signal_type == 'EXTREME':
            st.markdown(f'<div class="signal-box signal-short">EXTREME Z-SCORE<br/>Caution Required!</div>', 
                       unsafe_allow_html=True)
            st.write("⚠️ **Action**: Extreme deviation - consider larger position or wait")
        else:
            st.markdown(f'<div class="signal-box signal-none">NO SIGNAL<br/>Hold Current Position</div>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Key Metrics")
        st.metric("Z-Score", f"{zscore:.3f}")
        st.metric("Rolling Z-Score", f"{rolling_zscore:.3f}" if not pd.isna(rolling_zscore) else "N/A")
        st.metric("Correlation", f"{correlation:.3f}" if not pd.isna(correlation) else "N/A")
        
        # Signal quality indicator
        if abs(correlation) > 0.7:
            st.success("🟢 High correlation - Good for pairs trading")
        elif abs(correlation) > 0.3:
            st.warning("🟡 Moderate correlation - Proceed with caution")
        else:
            st.error("🔴 Low correlation - Not suitable for pairs trading")
    
    with col3:
        st.subheader("🛡️ Risk Management")
        if 'stoploss_upper' in df.columns:
            current_spread = latest_signal['spread']
            stoploss_upper = latest_signal.get('stoploss_upper', 0)
            stoploss_lower = latest_signal.get('stoploss_lower', 0)
            
            st.metric("Current Spread", f"{current_spread:.4f}")
            st.metric("Stop Loss Upper", f"{stoploss_upper:.4f}")
            st.metric("Stop Loss Lower", f"{stoploss_lower:.4f}")
            
            # Distance to stop loss
            if signal_type == 'LONG':
                distance_to_stop = abs(current_spread - stoploss_lower) / abs(current_spread) * 100
                st.metric("Distance to Stop Loss", f"{distance_to_stop:.1f}%")
            elif signal_type == 'SHORT':
                distance_to_stop = abs(stoploss_upper - current_spread) / abs(current_spread) * 100
                st.metric("Distance to Stop Loss", f"{distance_to_stop:.1f}%")

def display_statistics_table(stats, coin1, coin2):
    """
    Display comprehensive statistics in a formatted table
    """
    st.subheader("📈 Statistical Analysis")
    
    # Create two columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Regression Statistics**")
        st.write(f"**Alpha (α)**: {stats['alpha']:.6f}")
        st.write(f"**Beta (β)**: {stats['beta']:.6f}")
        st.write(f"**R-squared (R²)**: {stats['r_squared']:.4f}")
        st.write(f"**Pearson Correlation**: {stats['pearson_corr']:.4f}")
        st.write(f"**Spearman Correlation**: {stats['spearman_corr']:.4f}")
        
        # R-squared interpretation
        if stats['r_squared'] > 0.7:
            st.success("🟢 Strong relationship between assets")
        elif stats['r_squared'] > 0.4:
            st.warning("🟡 Moderate relationship between assets")
        else:
            st.error("🔴 Weak relationship between assets")
    
    with col2:
        st.markdown("**📉 Spread Statistics**")
        st.write(f"**Spread Mean**: {stats['spread_mean']:.6f}")
        st.write(f"**Spread Std Dev**: {stats['spread_std']:.6f}")
        st.write(f"**Spread Skewness**: {stats['spread_skew']:.4f}")
        st.write(f"**Spread Kurtosis**: {stats['spread_kurtosis']:.4f}")
        st.write(f"**Half-life**: {stats['half_life']:.2f} days" if stats['half_life'] != np.inf else "**Half-life**: ∞ (no mean reversion)")
        
        # Half-life interpretation
        if stats['half_life'] < 10:
            st.success("🟢 Fast mean reversion")
        elif stats['half_life'] < 30:
            st.warning("🟡 Moderate mean reversion")
        else:
            st.error("🔴 Slow/no mean reversion")
    
    # Volatility analysis
    st.markdown("**📊 Volatility Analysis**")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write(f"**{coin1} Volatility**: {stats['vol1']:.2%}")
        st.write(f"**{coin2} Volatility**: {stats['vol2']:.2%}")
        st.write(f"**Volatility Ratio**: {stats['vol_ratio']:.4f}")
    
    with col4:
        vol_diff = abs(stats['vol1'] - stats['vol2']) / max(stats['vol1'], stats['vol2'])
        st.write(f"**Volatility Difference**: {vol_diff:.2%}")
        
        if vol_diff < 0.2:
            st.success("🟢 Similar volatility levels")
        elif vol_diff < 0.5:
            st.warning("🟡 Moderate volatility difference")
        else:
            st.error("🔴 High volatility difference")

def display_performance_metrics(df):
    """
    Display performance metrics and backtesting results
    """
    if df.empty or 'signal' not in df.columns:
        return
    
    st.subheader("📊 Performance Analysis")
    
    # Calculate basic performance metrics
    signals = df['signal'].value_counts()
    total_signals = len(df[df['signal'] != 0])
    
    if total_signals > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", total_signals)
        
        with col2:
            long_signals = signals.get(1, 0)
            st.metric("Long Signals", long_signals)
        
        with col3:
            short_signals = signals.get(-1, 0)
            st.metric("Short Signals", short_signals)
        
        with col4:
            exit_signals = signals.get(2, 0)
            st.metric("Exit Signals", exit_signals)
        
        # Signal frequency analysis
        st.markdown("**Signal Frequency Analysis**")
        signal_freq = df.groupby(df.index.date)['signal'].apply(lambda x: (x != 0).sum())
        avg_signals_per_day = signal_freq.mean()
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Avg Signals/Day", f"{avg_signals_per_day:.2f}")
        
        with col6:
            days_with_signals = (signal_freq > 0).sum()
            total_days = len(signal_freq)
            signal_coverage = days_with_signals / total_days * 100
            st.metric("Signal Coverage", f"{signal_coverage:.1f}%")

def create_correlation_heatmap(df):
    """
    Create correlation heatmap for various metrics
    """
    if df.empty:
        return None
    
    # Select relevant columns for correlation analysis
    corr_columns = ['price1', 'price2', 'spread', 'zscore', 'rolling_corr', 'volume1', 'volume2']
    available_columns = [col for col in corr_columns if col in df.columns]
    
    if len(available_columns) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[available_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        width=600,
        height=500
    )
    
    return fig

# Main execution
def main():
    try:
        # Load data
        with st.spinner(f"Loading data for {coin1} and {coin2}..."):
            data1 = load_data(coin1, periode, interval)
            data2 = load_data(coin2, periode, interval)
        
        if data1.empty or data2.empty:
            st.error("Failed to load data. Please check your internet connection and try again.")
            return
        
        # Process data
        df = process_data(data1, data2)
        
        if df.empty:
            st.error("No overlapping data found between the selected assets.")
            return
        
        # Calculate statistics
        with st.spinner("Calculating statistics..."):
            stats_dict = calculate_enhanced_stats(df)
        
        # Calculate stop loss parameters
        stoploss_params = {
            'stoploss_pct': stoploss_pct,
            'volatility_window': volatility_window,
            'volatility_multiplier': volatility_multiplier,
            'atr_period': atr_period
        }
        
        # Add stop loss calculations
        df = calculate_dynamic_stoploss(df, stoploss_params)
        
        # Generate signals
        signal_params = {
            'zscore_entry_threshold': zscore_entry_threshold,
            'zscore_exit_threshold': zscore_exit_threshold,
            'min_correlation': min_correlation,
            'extreme_zscore': extreme_zscore,
            'lookback_period': lookback_period
        }
        
        df = generate_signals(df, signal_params)
        
        # Display current trading signals
        display_trading_signals(df, name1, name2)
        
        # Display statistics
        display_statistics_table(stats_dict, name1, name2)
        
        # Display performance metrics
        display_performance_metrics(df)
        
        # Create and display plots
        with st.spinner("Creating visualizations..."):
            fig = create_enhanced_plots(df, stats_dict, name1, name2)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Correlation Analysis")
        heatmap_fig = create_correlation_heatmap(df)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Position sizing calculator
        st.subheader("💼 Position Sizing Calculator")
        with st.expander("Calculate Position Size", expanded=False):
            portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=10000, step=1000)
            risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5) / 100
            
            if st.button("Calculate Position Size"):
                current_price1 = df['price1'].iloc[-1]
                current_price2 = df['price2'].iloc[-1]
                
                position_params = {
                    'max_position_size': portfolio_value * risk_per_trade,
                    'risk_per_trade': risk_per_trade
                }
                
                position_info = calculate_position_sizing(df, position_params, current_price1, current_price2)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{name1} Position**: {position_info['shares_asset1']:.4f} units")
                    st.write(f"**Position Value**: ${position_info['position_value']:.2f}")
                
                with col2:
                    st.write(f"**{name2} Position**: {position_info['shares_asset2']:.4f} units")
                    st.write(f"**Hedge Ratio**: {position_info['hedge_ratio']:.4f}")
                
                st.write(f"**Estimated Risk**: ${position_info['risk_estimate']:.2f}")
        
        # Data export
        st.subheader("📥 Data Export")
        with st.expander("Export Data", expanded=False):
            if st.button("Download CSV"):
                csv = df.to_csv()
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"pairs_trading_data_{coin1}_{coin2}.csv",
                    mime="text/csv"
                )
        
        # Footer with additional information
        st.markdown("---")
        st.markdown("""
        ### 📚 Trading Guidelines
        - **Long Signal**: Enter when Z-score < -2 (spread below mean)
        - **Short Signal**: Enter when Z-score > 2 (spread above mean)  
        - **Exit Signal**: Close positions when Z-score approaches 0
        - **Risk Management**: Always use stop losses and position sizing
        - **Correlation**: Ensure correlation > 0.3 for reliable signals
        
        ### ⚠️ Disclaimer
        This tool is for educational purposes only. Always do your own research and consider consulting with a financial advisor before making trading decisions.
        """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page or selecting different assets.")

# Run the main application
if __name__ == "__main__":
    main()
