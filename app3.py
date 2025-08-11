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
st.title("Advanced Pairs Trading Monitor")

# Sidebar instellingen
with st.sidebar:
    st.header("Pair Selection")
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

# === IMPROVED PRACTICAL TRADING EXECUTION SECTION ===
with st.expander("🎯 Praktische Trade Uitvoering", expanded=True):
    st.header("🎯 Praktische Trade Uitvoering")
    
    # Consistent calculations with main analysis
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]  # hedge ratio
    
    # Use consistent spread calculation
    current_price1 = df['price1'].iloc[-1]
    current_price2 = df['price2'].iloc[-1]
    current_spread = df['spread'].iloc[-1]  # Already calculated consistently
    current_zscore = df['zscore'].iloc[-1]  # Already calculated consistently
    
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    
    # === CURRENT MARKET STATUS ===
    st.subheader("📊 Huidige Marktsituatie")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(f"{name1} Prijs", f"${current_price1:.4f}")
    with col2:
        st.metric(f"{name2} Prijs", f"${current_price2:.4f}")
    with col3:
        color = "normal"
        if abs(current_zscore) >= zscore_entry_threshold:
            color = "inverse"
        st.metric("Z-Score", f"{current_zscore:.2f}", delta=f"{'Entry!' if abs(current_zscore) >= zscore_entry_threshold else 'Monitor'}", delta_color=color)
    with col4:
        st.metric("Hedge Ratio (β)", f"{beta:.3f}")
    with col5:
        st.metric("Correlation", f"{df['price1'].corr(df['price2']):.3f}")
    
    # === TRADING DECISION LOGIC ===
    st.markdown("---")
    
    # Calculate trade sizes for practical examples
    example_capital = st.number_input("💰 Voorbeeld Trading Capital ($)", 
                                    min_value=1000, max_value=1000000, 
                                    value=50000, step=1000,
                                    help="Voer uw trading capital in voor realistische berekeningen")
    
    position_percentage = st.slider("📊 Positie Grootte (% van capital)", 
                                   min_value=5.0, max_value=50.0, 
                                   value=20.0, step=2.5,
                                   help="Hoeveel procent van uw capital per trade")
    
    trade_capital = example_capital * (position_percentage / 100)
    
    # Determine current signal and calculate exact execution
    if current_zscore <= -zscore_entry_threshold:
        # LONG SPREAD SIGNAL
        st.success(f"🟢 **LONG SPREAD SIGNAAL** (Z-Score: {current_zscore:.2f})")
        
        st.markdown(f"""
        ### 📈 **UITVOERING LONG SPREAD TRADE**
        
        **🎯 Trading Logic:**
        - Spread is {abs(current_zscore):.2f} standaarddeviaties ONDER het gemiddelde
        - Verwachting: spread zal terug stijgen naar het gemiddelde
        - **BUY** de spread = Long {name1} + Short {name2}
        """)
        
        # Calculate exact position sizes
        capital_per_leg = trade_capital / 2
        
        # Long position in asset 1
        shares_asset1 = capital_per_leg / current_price1
        notional_asset1 = shares_asset1 * current_price1
        
        # Short position in asset 2 (hedge ratio adjusted)
        hedge_notional = notional_asset1 * beta
        shares_asset2 = hedge_notional / current_price2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 LEG 1: LONG " + name1)
            st.markdown(f"""
            - **Actie**: BUY (Long)
            - **Aantal aandelen**: {shares_asset1:,.2f}
            - **Prijs per aandeel**: ${current_price1:.4f}
            - **Totaal investering**: ${notional_asset1:,.2f}
            """)
            
        with col2:
            st.markdown("#### 🔴 LEG 2: SHORT " + name2)
            st.markdown(f"""
            - **Actie**: SELL (Short)
            - **Aantal aandelen**: {shares_asset2:,.2f}
            - **Prijs per aandeel**: ${current_price2:.4f}
            - **Totaal notional**: ${hedge_notional:,.2f}
            """)
        
        # Exit strategy
        exit_zscore_target = -zscore_exit_threshold
        target_spread = spread_mean + exit_zscore_target * spread_std
        
        st.markdown(f"""
        ### 🎯 **EXIT STRATEGIE**
        - **Exit wanneer**: Z-score stijgt naar -{zscore_exit_threshold:.1f} of hoger
        - **Target spread**: ${target_spread:.4f} (vs huidige ${current_spread:.4f})
        - **Stop Loss**: Bij Z-score van +{zscore_entry_threshold:.1f} (spread keert volledig om)
        """)
        
    elif current_zscore >= zscore_entry_threshold:
        # SHORT SPREAD SIGNAL
        st.error(f"🔴 **SHORT SPREAD SIGNAAL** (Z-Score: {current_zscore:.2f})")
        
        st.markdown(f"""
        ### 📉 **UITVOERING SHORT SPREAD TRADE**
        
        **🎯 Trading Logic:**
        - Spread is {current_zscore:.2f} standaarddeviaties BOVEN het gemiddelde
        - Verwachting: spread zal terug dalen naar het gemiddelde
        - **SELL** de spread = Short {name1} + Long {name2}
        """)
        
        # Calculate exact position sizes
        capital_per_leg = trade_capital / 2
        
        # Short position in asset 1
        shares_asset1 = capital_per_leg / current_price1
        notional_asset1 = shares_asset1 * current_price1
        
        # Long position in asset 2 (hedge ratio adjusted)
        hedge_notional = notional_asset1 * beta
        shares_asset2 = hedge_notional / current_price2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 LEG 1: SHORT " + name1)
            st.markdown(f"""
            - **Actie**: SELL (Short)
            - **Aantal aandelen**: {shares_asset1:,.2f}
            - **Prijs per aandeel**: ${current_price1:.4f}
            - **Totaal notional**: ${notional_asset1:,.2f}
            """)
            
        with col2:
            st.markdown("#### 🟢 LEG 2: LONG " + name2)
            st.markdown(f"""
            - **Actie**: BUY (Long)
            - **Aantal aandelen**: {shares_asset2:,.2f}
            - **Prijs per aandeel**: ${current_price2:.4f}
            - **Totaal investering**: ${hedge_notional:,.2f}
            """)
        
        # Exit strategy
        exit_zscore_target = zscore_exit_threshold
        target_spread = spread_mean + exit_zscore_target * spread_std
        
        st.markdown(f"""
        ### 🎯 **EXIT STRATEGIE**
        - **Exit wanneer**: Z-score daalt naar +{zscore_exit_threshold:.1f} of lager
        - **Target spread**: ${target_spread:.4f} (vs huidige ${current_spread:.4f})
        - **Stop Loss**: Bij Z-score van -{zscore_entry_threshold:.1f} (spread keert volledig om)
        """)
        
    else:
        # NO SIGNAL
        st.info(f"⏳ **GEEN SIGNAAL** - WACHT (Z-Score: {current_zscore:.2f})")
        
        # Show distance to signals
        distance_to_long = abs(current_zscore - (-zscore_entry_threshold))
        distance_to_short = abs(current_zscore - zscore_entry_threshold)
        
        next_signal = "LONG" if distance_to_long < distance_to_short else "SHORT"
        next_distance = min(distance_to_long, distance_to_short)
        
        st.markdown(f"""
        ### ⌛ **WACHT OP SIGNAAL**
        
        **Huidige Status:**
        - Z-score: {current_zscore:.2f} (binnen neutrale zone ±{zscore_entry_threshold:.1f})
        - Spread is {abs(current_zscore):.2f} standaarddeviaties van gemiddelde
        - **Dichtstbijzijnde signaal**: {next_signal} (nog {next_distance:.2f} Z-score punten)
        
        **Entry Levels:**
        - 🟢 **LONG SPREAD** bij Z-score ≤ -{zscore_entry_threshold:.1f}
        - 🔴 **SHORT SPREAD** bij Z-score ≥ +{zscore_entry_threshold:.1f}
        """)
        
        # Calculate what prices would trigger signals
        long_entry_spread = spread_mean - zscore_entry_threshold * spread_std
        short_entry_spread = spread_mean + zscore_entry_threshold * spread_std
        
        st.markdown(f"""
        **In Spread termen:**
        - LONG entry bij spread ≤ ${long_entry_spread:.4f}
        - SHORT entry bij spread ≥ ${short_entry_spread:.4f}
        - Huidige spread: ${current_spread:.4f}
        """)
    
    # === RISK MANAGEMENT SECTION ===
    st.markdown("---")
    st.subheader("⚠️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🛡️ **Stop Loss Rules**
        - **Z-score omkering**: Exit als Z-score volledig omkeert
        - **Time decay**: Max 30 dagen per trade
        - **Correlatie breakdown**: Exit als correlatie < 0.6
        - **Max verlies**: Stop bij -5% van trade capital
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 **Profit Taking**
        - **Primaire exit**: Z-score normaliseert (mean reversion)
        - **Profit target**: +8% van trade capital  
        - **Gedeeltelijke exit**: 50% bij +4%, rest laten lopen
        - **Trailing stop**: Na +6% profit
        """)
    
    # === EXECUTION CHECKLIST ===
    st.markdown("---")
    st.subheader("✅ Pre-Trade Checklist")
    
    # Calculate real-time risk metrics
    recent_corr = df['price1'].tail(20).corr(df['price2'].tail(20))
    volatility_1 = df['price1'].pct_change().tail(20).std() * np.sqrt(252) * 100
    volatility_2 = df['price2'].pct_change().tail(20).std() * np.sqrt(252) * 100
    
    checklist_items = [
        {"item": "Correlatie Check", "status": recent_corr > 0.6, "value": f"{recent_corr:.3f}", "threshold": "> 0.6"},
        {"item": "Z-Score Significantie", "status": abs(current_zscore) >= zscore_entry_threshold, "value": f"{abs(current_zscore):.2f}", "threshold": f"≥ {zscore_entry_threshold}"},
        {"item": "Volatiliteit " + name1, "status": volatility_1 < 50, "value": f"{volatility_1:.1f}%", "threshold": "< 50%"},
        {"item": "Volatiliteit " + name2, "status": volatility_2 < 50, "value": f"{volatility_2:.1f}%", "threshold": "< 50%"},
        {"item": "Spread Stabiliteit", "status": spread_std > 0.001, "value": f"{spread_std:.4f}", "threshold": "> 0.001"},
    ]
    
    for item in checklist_items:
        status_icon = "✅" if item["status"] else "❌"
        status_color = "green" if item["status"] else "red"
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.markdown(f"**{item['item']}**")
        with col2:
            st.markdown(f"**{item['value']}**")
        with col3:
            st.markdown(f"Threshold: {item['threshold']}")
        with col4:
            st.markdown(f"<span style='color: {status_color}; font-size: 20px;'>{status_icon}</span>", unsafe_allow_html=True)
    
    # Overall trade readiness
    ready_count = sum(item["status"] for item in checklist_items)
    trade_ready = ready_count >= 4
    
    if trade_ready and abs(current_zscore) >= zscore_entry_threshold:
        st.success(f"🚀 **TRADE READY** ({ready_count}/5 checks passed)")
    elif abs(current_zscore) >= zscore_entry_threshold:
        st.warning(f"⚠️ **SIGNAAL AANWEZIG maar risico's** ({ready_count}/5 checks passed)")
    else:
        st.info(f"⏳ **WACHT OP SIGNAAL** ({ready_count}/5 checks passed)")
    
    # === PRACTICAL EXECUTION NOTES ===
    st.markdown("---")
    st.subheader("📝 Praktische Uitvoering Tips")
    
    with st.expander("🔧 Execution Details", expanded=False):
        st.markdown(f"""
        ### **Broker Requirements**
        - **Short selling**: Beide assets moeten shortbaar zijn
        - **Margin**: Minimaal 50% margin voor short posities
        - **Hedge ratio**: Altijd {beta:.3f} ratio aanhouden tussen assets
        
        ### **Order Types**  
        - **Market orders**: Voor snelle entry bij sterke signalen
        - **Limit orders**: Bij langzame mean reversion
        - **Stop orders**: Voor risk management exits
        
        ### **Timing**
        - **Best execution**: Tijdens market hours, hoge liquiditeit
        - **Avoid**: Earnings announcements, Fed meetings
        - **Monitor**: Na entry, dagelijks Z-score checken
        
        ### **Portfolio Impact**
        - **Correlatie**: Max 30% van portfolio in pairs trades
        - **Sector exposure**: Diversifieer over sectoren
        - **Rebalancing**: Weekly review van hedge ratios
        """)
    
    # Current hedge ratio validation
    if abs(beta - 1.0) > 0.5:
        st.warning(f"⚠️ **Hedge Ratio Warning**: β = {beta:.3f} betekent ongelijke position sizes. Voor elke ${notional_asset1:,.0f} in {name1}, heb je ${notional_asset1 * beta:,.0f} in {name2} nodig.")
# === REALISTIC PAIRS TRADING BACKTEST ===
def run_realistic_pairs_backtest(df, entry_threshold, exit_threshold, initial_capital=100000, 
                                transaction_cost=0.05, position_size_pct=20, 
                                stop_loss_pct=5.0, take_profit_pct=8.0, 
                                lookback_window=20, max_trade_days=30):
    """
    Realistische pairs trading backtest die beide legs van elke trade tracked
    """
    # Bereken regression parameters
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]  # hedge ratio
    
    # Bereken spread en z-score met rolling window
    df = df.copy()
    df['spread'] = df['price2'] - (alpha + beta * df['price1'])
    df['spread_mean'] = df['spread'].rolling(lookback_window).mean()
    df['spread_std'] = df['spread'].rolling(lookback_window).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # Trading state
    cash = initial_capital
    position_active = False
    position_type = None  # 'long_spread' or 'short_spread'
    
    # Position details - BEIDE LEGS APART TRACKED
    shares_asset1 = 0.0
    shares_asset2 = 0.0
    entry_price1 = 0.0
    entry_price2 = 0.0
    entry_zscore = 0.0
    entry_date = None
    position_value = 0.0
    
    # Results tracking
    trades = []
    daily_portfolio = []
    
    for i in range(lookback_window, len(df)):
        current_date = df.index[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_zscore = df['zscore'].iloc[i]
        
        # Skip if zscore is NaN
        if pd.isna(current_zscore):
            continue
            
        # Calculate current portfolio value
        if position_active:
            # Current value of both positions
            current_value1 = shares_asset1 * current_price1
            current_value2 = shares_asset2 * current_price2
            total_position_value = current_value1 + current_value2
            
            # Calculate unrealized P&L
            unrealized_pnl = total_position_value - position_value
            portfolio_value = cash + total_position_value
            
        else:
            total_position_value = 0
            unrealized_pnl = 0
            portfolio_value = cash
        
        # Store daily portfolio data
        daily_portfolio.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position_value': total_position_value,
            'unrealized_pnl': unrealized_pnl,
            'price1': current_price1,
            'price2': current_price2,
            'zscore': current_zscore,
            'position_active': position_active,
            'position_type': position_type
        })
        
        # ENTRY LOGIC
        if not position_active:
            position_size = (position_size_pct / 100) * portfolio_value
            
            # LONG SPREAD: Long Asset1, Short Asset2
            if current_zscore <= -entry_threshold:
                position_active = True
                position_type = 'long_spread'
                entry_date = current_date
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_zscore = current_zscore
                
                # Calculate position sizes with proper hedging
                # Split capital between both legs
                capital_per_leg = position_size / 2
                
                # Long position in asset 1
                shares_asset1 = capital_per_leg / current_price1
                gross_cost1 = shares_asset1 * current_price1
                transaction_cost1 = gross_cost1 * (transaction_cost / 100)
                
                # Short position in asset 2 (hedged amount)
                hedge_notional = shares_asset1 * current_price1 * beta
                shares_asset2 = -hedge_notional / current_price2  # Negative = short
                gross_cost2 = abs(shares_asset2 * current_price2)
                transaction_cost2 = gross_cost2 * (transaction_cost / 100)
                
                # Total initial position value (what we paid)
                position_value = gross_cost1 + abs(shares_asset2 * current_price2)
                total_transaction_costs = transaction_cost1 + transaction_cost2
                
                # Deduct transaction costs from cash
                cash -= total_transaction_costs
                
            # SHORT SPREAD: Short Asset1, Long Asset2  
            elif current_zscore >= entry_threshold:
                position_active = True
                position_type = 'short_spread'
                entry_date = current_date
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_zscore = current_zscore
                
                capital_per_leg = position_size / 2
                
                # Short position in asset 1
                shares_asset1 = -capital_per_leg / current_price1  # Negative = short
                gross_cost1 = abs(shares_asset1 * current_price1)
                transaction_cost1 = gross_cost1 * (transaction_cost / 100)
                
                # Long position in asset 2 (hedged amount)
                hedge_notional = abs(shares_asset1) * current_price1 * beta
                shares_asset2 = hedge_notional / current_price2
                gross_cost2 = shares_asset2 * current_price2
                transaction_cost2 = gross_cost2 * (transaction_cost / 100)
                
                position_value = gross_cost1 + gross_cost2
                total_transaction_costs = transaction_cost1 + transaction_cost2
                cash -= total_transaction_costs
        
        # EXIT LOGIC
        elif position_active:
            # Calculate current P&L percentage
            current_position_value = shares_asset1 * current_price1 + shares_asset2 * current_price2
            pnl_dollars = current_position_value - position_value
            pnl_percentage = (pnl_dollars / position_value) * 100 if position_value != 0 else 0
            
            # Days in trade
            days_in_trade = (current_date - entry_date).days
            
            exit_trade = False
            exit_reason = ""
            
            # Mean reversion exit
            if position_type == 'long_spread' and current_zscore >= -exit_threshold:
                exit_trade = True
                exit_reason = "Mean reversion (long spread)"
            elif position_type == 'short_spread' and current_zscore <= exit_threshold:
                exit_trade = True
                exit_reason = "Mean reversion (short spread)"
            
            # Stop loss
            elif pnl_percentage <= -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            
            # Take profit
            elif pnl_percentage >= take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            # Maximum time in trade
            elif days_in_trade >= max_trade_days:
                exit_trade = True
                exit_reason = "Time limit"
            
            # Extreme Z-score reversal (risk management)
            elif ((position_type == 'long_spread' and current_zscore >= entry_threshold) or 
                  (position_type == 'short_spread' and current_zscore <= -entry_threshold)):
                exit_trade = True
                exit_reason = "Z-score reversal"
            
            if exit_trade:
                # Calculate exit transaction costs
                exit_cost1 = abs(shares_asset1 * current_price1) * (transaction_cost / 100)
                exit_cost2 = abs(shares_asset2 * current_price2) * (transaction_cost / 100)
                total_exit_costs = exit_cost1 + exit_cost2
                
                # Final P&L after exit costs
                gross_pnl = current_position_value - position_value
                net_pnl = gross_pnl - total_exit_costs
                
                # Update cash with proceeds
                cash += current_position_value - total_exit_costs
                
                # Record the trade with BOTH LEGS
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'position_type': position_type,
                    'days_held': days_in_trade,
                    
                    # Entry details
                    'entry_zscore': entry_zscore,
                    'entry_price1': entry_price1,
                    'entry_price2': entry_price2,
                    
                    # Exit details  
                    'exit_zscore': current_zscore,
                    'exit_price1': current_price1,
                    'exit_price2': current_price2,
                    
                    # Position details
                    'shares_asset1': shares_asset1,
                    'shares_asset2': shares_asset2,
                    'hedge_ratio': beta,
                    
                    # Performance
                    'position_size': position_value,
                    'gross_pnl': gross_pnl,
                    'transaction_costs': total_transaction_costs + total_exit_costs,
                    'net_pnl': net_pnl,
                    'pnl_percentage': (net_pnl / position_value) * 100 if position_value != 0 else 0,
                    
                    # Individual leg performance
                    'asset1_return': ((current_price1 - entry_price1) / entry_price1) * 100,
                    'asset2_return': ((current_price2 - entry_price2) / entry_price2) * 100,
                    
                    'exit_reason': exit_reason
                })
                
                # Reset position
                position_active = False
                position_type = None
                shares_asset1 = 0.0
                shares_asset2 = 0.0
                position_value = 0.0
    
    # Convert to DataFrames
    portfolio_df = pd.DataFrame(daily_portfolio)
    if not portfolio_df.empty:
        portfolio_df.set_index('date', inplace=True)
    
    trades_df = pd.DataFrame(trades)
    
    return portfolio_df, trades_df

# === REALISTIC BACKTEST UI ===
with st.expander("🎯 Realistic Pairs Trading Backtest", expanded=True):
    st.header("🎯 Realistic Pairs Trading Backtest")
    
    st.info("""
    **Deze backtest toont beide legs van elke trade:**
    - Long Spread = Long Asset1 + Short Asset2 (hedged)
    - Short Spread = Short Asset1 + Long Asset2 (hedged)
    - Realistische profit targets en risk management
    """)
    
    # Realistic parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Capital Settings")
        initial_capital = st.number_input("Initial Capital ($)", 1000, 10000000, 100000, 1000)
        position_size_pct = st.slider("Position Size (% of portfolio)", 5.0, 50.0, 20.0, 2.5)
        transaction_cost = st.slider("Transaction Cost (%)", 0.01, 0.5, 0.05, 0.01)
    
    with col2:
        st.subheader("🎯 Risk Management")
        stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 15.0, 5.0, 0.5)
        take_profit_pct = st.slider("Take Profit (%)", 2.0, 20.0, 8.0, 0.5)
        max_trade_days = st.slider("Max Days per Trade", 5, 90, 30, 5)
    
    with col3:
        st.subheader("📊 Technical Settings") 
        lookback_window = st.slider("Z-score Lookback Window", 10, 100, 20, 5)
        
        # Show current Z-score thresholds from sidebar
        st.metric("Entry Z-score Threshold", f"±{zscore_entry_threshold:.1f}")
        st.metric("Exit Z-score Threshold", f"±{zscore_exit_threshold:.1f}")
    
    # Run backtest
    if st.button("🚀 Run Realistic Backtest", type="primary"):
        with st.spinner("Running realistic pairs trading backtest..."):
            portfolio_df, trades_df = run_realistic_pairs_backtest(
                df, zscore_entry_threshold, zscore_exit_threshold,
                initial_capital, transaction_cost, position_size_pct,
                stop_loss_pct, take_profit_pct, lookback_window, max_trade_days
            )
        
        if trades_df.empty:
            st.warning("⚠️ No trades generated. Try adjusting Z-score thresholds or time period.")
        else:
            # Performance Summary
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            num_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
            win_rate = (winning_trades / num_trades) * 100
            avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if winning_trades < num_trades else 0
            
            # Risk metrics
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            max_dd = ((portfolio_df['portfolio_value'].cummax() - portfolio_df['portfolio_value']) / portfolio_df['portfolio_value'].cummax()).max() * 100
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            st.success(f"✅ Backtest Complete: {num_trades} trades executed")
            
            # Key Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Return", f"{total_return:.2f}%", 
                         delta=f"${final_value - initial_capital:,.0f}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                         delta=f"{winning_trades}/{num_trades} trades")
            with col3:
                st.metric("Avg Win/Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")
            with col4:
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            with col5:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Portfolio Performance Chart
            st.subheader("📈 Portfolio Performance")
            
            fig_performance = go.Figure()
            
            # Portfolio value
            fig_performance.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Mark trade entries and exits
            for _, trade in trades_df.iterrows():
                color = 'green' if trade['position_type'] == 'long_spread' else 'red'
                
                # Entry marker
                fig_performance.add_trace(go.Scatter(
                    x=[trade['entry_date']],
                    y=[portfolio_df.loc[trade['entry_date'], 'portfolio_value']],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-up'),
                    name=f"Entry {trade['position_type']}",
                    showlegend=False,
                    hovertemplate=f"<b>Entry</b><br>Type: {trade['position_type']}<br>Z-score: {trade['entry_zscore']:.2f}<extra></extra>"
                ))
                
                # Exit marker
                fig_performance.add_trace(go.Scatter(
                    x=[trade['exit_date']],
                    y=[portfolio_df.loc[trade['exit_date'], 'portfolio_value']],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-down'),
                    name=f"Exit {trade['position_type']}",
                    showlegend=False,
                    hovertemplate=f"<b>Exit</b><br>P&L: ${trade['net_pnl']:.0f}<br>Reason: {trade['exit_reason']}<extra></extra>"
                ))
            
            fig_performance.update_layout(
                title=f"Portfolio Performance - {name1} vs {name2}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Detailed Trade Analysis
            st.subheader("📋 Trade Details (Both Legs Shown)")
            
            # Format trades for display
            display_trades = trades_df.copy()
            
            # Round and format numeric columns
            numeric_cols = ['entry_zscore', 'exit_zscore', 'shares_asset1', 'shares_asset2', 
                           'gross_pnl', 'net_pnl', 'pnl_percentage', 'asset1_return', 'asset2_return']
            for col in numeric_cols:
                if col in display_trades.columns:
                    display_trades[col] = display_trades[col].round(3)
            
            # Format currency columns
            currency_cols = ['position_size', 'gross_pnl', 'net_pnl', 'transaction_costs']
            for col in currency_cols:
                if col in display_trades.columns:
                    display_trades[col] = display_trades[col].apply(lambda x: f"${x:,.2f}")
            
            # Rename columns for clarity
            display_trades = display_trades.rename(columns={
                'entry_date': 'Entry Date',
                'exit_date': 'Exit Date', 
                'position_type': 'Position Type',
                'days_held': 'Days',
                'entry_zscore': 'Entry Z',
                'exit_zscore': 'Exit Z',
                'shares_asset1': f'Shares {name1}',
                'shares_asset2': f'Shares {name2}',
                'position_size': 'Position Size',
                'net_pnl': 'Net P&L',
                'pnl_percentage': 'P&L %',
                'asset1_return': f'{name1} Return %',
                'asset2_return': f'{name2} Return %',
                'exit_reason': 'Exit Reason'
            })
            
            # Display trades table
            st.dataframe(display_trades, use_container_width=True, hide_index=True)
            
            # Trade Analysis Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L Distribution
                fig_pnl = px.histogram(
                    trades_df, x='net_pnl', nbins=min(20, len(trades_df)),
                    title="P&L Distribution",
                    labels={'net_pnl': 'Net P&L ($)'}
                )
                fig_pnl.update_traces(marker_color='lightblue')
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Trade Duration
                fig_duration = px.histogram(
                    trades_df, x='days_held', nbins=min(15, len(trades_df)),
                    title="Trade Duration Distribution",
                    labels={'days_held': 'Days Held'}
                )
                fig_duration.update_traces(marker_color='lightgreen')
                st.plotly_chart(fig_duration, use_container_width=True)
            
            # Performance by Position Type
            if len(trades_df['position_type'].unique()) > 1:
                st.subheader("📊 Performance by Position Type")
                
                perf_by_type = trades_df.groupby('position_type').agg({
                    'net_pnl': ['count', 'mean', 'sum'],
                    'pnl_percentage': 'mean',
                    'days_held': 'mean'
                }).round(2)
                
                perf_by_type.columns = ['Count', 'Avg P&L ($)', 'Total P&L ($)', 'Avg P&L %', 'Avg Days']
                
                st.dataframe(perf_by_type, use_container_width=True)
                
                # P&L by position type
                fig_by_type = px.box(
                    trades_df, x='position_type', y='net_pnl',
                    title="P&L Distribution by Position Type"
                )
                st.plotly_chart(fig_by_type, use_container_width=True)
