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
    
    # Half-life of mean reversion
    y_lag = df['spread'].shift(1).dropna()
    y_diff = df['spread'].diff().dropna()
    y_lag = y_lag[1:]
    
    if len(y_lag) > 0:
        half_life_model = LinearRegression()
        half_life_model.fit(y_lag.values.reshape(-1, 1), y_diff.values)
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
    current_spread = df['spread'].iloc[-1]
    current_zscore = df['zscore'].iloc[-1]
    
    # Volatility-based stop loss
    current_vol = df['spread'].rolling(stoploss_params['volatility_window']).std().iloc[-1]
    
    # ATR-based stop loss
    spread_high = df['spread'].rolling(stoploss_params['atr_period']).max()
    spread_low = df['spread'].rolling(stoploss_params['atr_period']).min()
    spread_range = spread_high - spread_low
    current_atr = spread_range.rolling(stoploss_params['atr_period']).mean().iloc[-1]
    
    # Calculate stop loss levels
    volatility_stoploss_distance = current_vol * stoploss_params['volatility_multiplier']
    atr_stoploss_distance = current_atr * 0.5
    
    combined_stoploss_distance = max(volatility_stoploss_distance, atr_stoploss_distance)
    
    return {
        'volatility_upper': df['spread'].mean() + volatility_stoploss_distance,
        'volatility_lower': df['spread'].mean() - volatility_stoploss_distance,
        'atr_upper': df['spread'].mean() + atr_stoploss_distance,
        'atr_lower': df['spread'].mean() - atr_stoploss_distance,
        'combined_upper': df['spread'].mean() + combined_stoploss_distance,
        'combined_lower': df['spread'].mean() - combined_stoploss_distance,
        'current_volatility': current_vol,
        'current_atr': current_atr
    }

def check_stoploss_triggers(df, stoploss_levels, position_type):
    current_spread = df['spread'].iloc[-1]
    current_zscore = df['zscore'].iloc[-1]
    current_corr = df['rolling_corr'].iloc[-1]
    
    triggers = []
    
    # Check volatility-based stop loss
    if position_type == "LONG":
        if current_spread <= stoploss_levels['combined_lower']:
            triggers.append("Volatility Stop Loss (Long)")
    elif position_type == "SHORT":
        if current_spread >= stoploss_levels['combined_upper']:
            triggers.append("Volatility Stop Loss (Short)")
    
    # Check extreme z-score
    if abs(current_zscore) > extreme_zscore:
        triggers.append(f"Extreme Z-Score ({current_zscore:.2f})")
    
    # Check correlation breakdown
    if current_corr < min_correlation:
        triggers.append(f"Correlation Breakdown ({current_corr:.3f})")
    
    return triggers

def generate_signals(df, zscore_entry, zscore_exit, lookback):
    last_zscore = df['zscore'].iloc[-1]
    
    # Create stop loss parameters
    stoploss_params = {
        'volatility_window': volatility_window,
        'volatility_multiplier': volatility_multiplier,
        'atr_period': atr_period
    }
    
    # Determine position type
    if last_zscore < -zscore_entry:
        position_type = "LONG"
        signal = "LONG"
        signal_class = "signal-long"
        position = f"Long Spread: Buy {name2}, Sell {name1}"
        confidence = min(abs(last_zscore) / zscore_entry, 3.0)
    elif last_zscore > zscore_entry:
        position_type = "SHORT"
        signal = "SHORT"
        signal_class = "signal-short"
        position = f"Short Spread: Sell {name2}, Buy {name1}"
        confidence = min(abs(last_zscore) / zscore_entry, 3.0)
    elif abs(last_zscore) < zscore_exit:
        position_type = "EXIT"
        signal = "EXIT"
        signal_class = "signal-exit"
        position = "Exit Position"
        confidence = 1.0 - abs(last_zscore) / zscore_exit
    else:
        position_type = "HOLD"
        signal = "HOLD"
        signal_class = "signal-none"
        position = "No Clear Signal"
        confidence = 0.0
    
    # Calculate stop loss levels
    stoploss_levels = calculate_dynamic_stoploss(df, stoploss_params)
    
    # Check for stop loss triggers
    stoploss_triggers = check_stoploss_triggers(df, stoploss_levels, position_type)
    
    # Override signal if stop loss is triggered
    if stoploss_triggers:
        signal = "STOP_LOSS"
        signal_class = "signal-short"
        position = f"STOP LOSS TRIGGERED: {', '.join(stoploss_triggers)}"
        confidence = 1.0
    
    return signal, signal_class, position, confidence, stoploss_levels, stoploss_triggers

# Main execution
with st.spinner(f"Loading data for {name1} and {name2}..."):
    data1 = load_data(coin1, periode, interval)
    data2 = load_data(coin2, periode, interval)

if data1.empty or data2.empty:
    st.error("❌ Unable to load data for one or both assets.")
    st.stop()

df = process_data(data1, data2)

if df.empty:
    st.error("❌ No overlapping data available for both assets.")
    st.stop()

stats_dict = calculate_enhanced_stats(df)
signal, signal_class, position, confidence, stoploss_levels, stoploss_triggers = generate_signals(
    df, zscore_entry_threshold, zscore_exit_threshold, lookback_period
)

# Display sections with expanders
st.markdown("## 📊 Trading Dashboard")

# Current signal section
with st.expander("🚦 Current Trading Signal", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="{signal_class} signal-box">
        <h3>{signal}</h3>
        <p>{position}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Current Z-Score", f"{df['zscore'].iloc[-1]:.2f}")
        st.metric("Signal Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("Half-Life (days)", f"{stats_dict['half_life']:.1f}" if stats_dict['half_life'] != np.inf else "∞")
        st.metric("Spread Volatility", f"{stats_dict['spread_std']:.4f}")

# Stop loss information
with st.expander("🛑 Stop Loss Management", expanded=bool(stoploss_triggers)):
    if stoploss_triggers:
        st.markdown(f"""
        <div class="stoploss-box">
        <h4>⚠️ STOP LOSS TRIGGERED</h4>
        <p><strong>Triggers:</strong> {', '.join(stoploss_triggers)}</p>
        <p><strong>Action:</strong> Exit position immediately</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Volatility-Based**")
        st.markdown(f"""
        <div class="stoploss-level">
        <strong>Upper:</strong> {stoploss_levels['volatility_upper']:.4f}<br>
        <strong>Lower:</strong> {stoploss_levels['volatility_lower']:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**ATR-Based**")
        st.markdown(f"""
        <div class="stoploss-level">
        <strong>Upper:</strong> {stoploss_levels['atr_upper']:.4f}<br>
        <strong>Lower:</strong> {stoploss_levels['atr_lower']:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Combined (Active)**")
        st.markdown(f"""
        <div class="stoploss-level">
        <strong>Upper:</strong> {stoploss_levels['combined_upper']:.4f}<br>
        <strong>Lower:</strong> {stoploss_levels['combined_lower']:.4f}<br>
        <strong>Current:</strong> {df['spread'].iloc[-1]:.4f}
        </div>
        """, unsafe_allow_html=True)

# Risk monitoring
with st.expander("⚠️ Risk Monitoring"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zscore_risk = "🔴 HIGH" if abs(df['zscore'].iloc[-1]) > extreme_zscore else "🟡 MEDIUM" if abs(df['zscore'].iloc[-1]) > 2.5 else "🟢 LOW"
        st.metric("Z-Score Risk", zscore_risk)
    
    with col2:
        corr_risk = "🔴 HIGH" if df['rolling_corr'].iloc[-1] < min_correlation else "🟡 MEDIUM" if df['rolling_corr'].iloc[-1] < 0.5 else "🟢 LOW"
        st.metric("Correlation Risk", corr_risk)
    
    with col3:
        vol_risk = "🔴 HIGH" if stoploss_levels['current_volatility'] > stats_dict['spread_std'] * 1.5 else "🟡 MEDIUM" if stoploss_levels['current_volatility'] > stats_dict['spread_std'] * 1.2 else "🟢 LOW"
        st.metric("Volatility Risk", vol_risk)

# Charts section
with st.expander("📈 Analysis Charts", expanded=True):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Price Comparison', 'Z-Score with Signals',
                       'Spread Analysis', 'Rolling Correlation',
                       'Returns Scatter Plot', 'Volume Analysis'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]],
        vertical_spacing=0.08
    )
    
    # Price comparison
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price1'], name=name1, line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price2'], name=name2, line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # Z-score with signals
    fig.add_trace(
        go.Scatter(x=df.index, y=df['zscore'], name='Z-Score', line=dict(color='purple')),
        row=1, col=2
    )
    fig.add_hline(y=zscore_entry_threshold, line=dict(color='red', dash='dash'), row=1, col=2)
    fig.add_hline(y=-zscore_entry_threshold, line=dict(color='green', dash='dash'), row=1, col=2)
    fig.add_hline(y=zscore_exit_threshold, line=dict(color='blue', dash='dot'), row=1, col=2)
    fig.add_hline(y=-zscore_exit_threshold, line=dict(color='blue', dash='dot'), row=1, col=2)
    
    # Spread analysis
    fig.add_trace(
        go.Scatter(x=df.index, y=df['spread'], name='Spread', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_hline(y=stats_dict['spread_mean'], line=dict(color='black', dash='dash'), row=2, col=1)
    fig.add_hline(y=stoploss_levels['combined_upper'], line=dict(color='red', dash='dot', width=2), row=2, col=1)
    fig.add_hline(y=stoploss_levels['combined_lower'], line=dict(color='red', dash='dot', width=2), row=2, col=1)
    
    # Rolling correlation
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rolling_corr'], name='Rolling Correlation',
                  fill='tozeroy', fillcolor='rgba(0,255,0,0.3)', line=dict(color='green')),
        row=2, col=2
    )
    
    # Returns scatter plot
    returns_clean = df[['returns1', 'returns2']].dropna()
    fig.add_trace(
        go.Scatter(x=returns_clean['returns1']*100, y=returns_clean['returns2']*100,
                  mode='markers', name='Returns', marker=dict(color='purple', size=4, opacity=0.6)),
        row=3, col=1
    )
    
    # Volume analysis
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volume1'], name=f'{name1} Volume', line=dict(color='lightblue')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volume2'], name=f'{name2} Volume', line=dict(color='lightcoral')),
        row=3, col=2, secondary_y=True
    )
    
    fig.update_layout(height=1200, title_text="Comprehensive Pairs Trading Analysis", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Statistics section
with st.expander("📊 Detailed Statistics"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Regression Analysis**")
        st.metric("Alpha (α)", f"{stats_dict['alpha']:.6f}")
        st.metric("Beta (β)", f"{stats_dict['beta']:.4f}")
        st.metric("R-squared", f"{stats_dict['r_squared']:.4f}")
    
    with col2:
        st.markdown("**Correlation Metrics**")
        st.metric("Pearson Correlation", f"{stats_dict['pearson_corr']:.4f}")
        st.metric("Spearman Correlation", f"{stats_dict['spearman_corr']:.4f}")
        st.metric("Current Rolling Corr", f"{df['rolling_corr'].iloc[-1]:.4f}")
    
    with col3:
        st.markdown("**Spread Statistics**")
        st.metric("Mean", f"{stats_dict['spread_mean']:.4f}")
        st.metric("Std Deviation", f"{stats_dict['spread_std']:.4f}")
        st.metric("Skewness", f"{stats_dict['spread_skew']:.4f}")
        st.metric("Kurtosis", f"{stats_dict['spread_kurtosis']:.4f}")
    
    with col4:
        st.markdown("**Volatility Analysis**")
        st.metric(f"{name1} Vol (Ann.)", f"{stats_dict['vol1']:.2%}")
        st.metric(f"{name2} Vol (Ann.)", f"{stats_dict['vol2']:.2%}")
        st.metric("Volatility Ratio", f"{stats_dict['vol_ratio']:.4f}")

# Position management
with st.expander("📋 Position Management"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Entry Checklist:**")
        checklist_items = [
            ("Z-score within entry range", abs(df['zscore'].iloc[-1]) > zscore_entry_threshold),
            ("Correlation above minimum", df['rolling_corr'].iloc[-1] > min_correlation),
            ("Volatility not extreme", stoploss_levels['current_volatility'] < stats_dict['spread_std'] * 2),
        ]
        
        for item, condition in checklist_items:
            status = "✅"
