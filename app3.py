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
with st.expander("🎯 Praktische Trade Uitvoering - USDT Paren", expanded=True):
    st.header("🎯 Praktische Trade Uitvoering - USDT Coin Paren")
    
    # Consistent calculations with main analysis
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]  # hedge ratio
    
    # Current market data
    current_price1 = df['price1'].iloc[-1]  # Current price Asset 1
    current_price2 = df['price2'].iloc[-1]  # Current price Asset 2
    current_spread = df['spread'].iloc[-1]  # Already calculated consistently
    current_zscore = df['zscore'].iloc[-1]  # Already calculated consistently
    
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    
    # === TRADING CAPITAL INPUT ===
    st.subheader("💰 Jouw Trading Capital")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        trading_capital = st.number_input("💵 Trading Budget (USDT)", 
                                        min_value=50, max_value=100000, 
                                        value=1000, step=25,
                                        help="Minimum 50 USDT voor pairs trading")
    with col2:
        risk_per_trade = st.slider("🎯 Risico per Trade (%)", 
                                 min_value=1.0, max_value=10.0, 
                                 value=2.0, step=0.5,
                                 help="Maximaal verlies dat je accepteert")
    with col3:
        max_risk_usdt = trading_capital * (risk_per_trade / 100)
        st.metric("Max Verlies per Trade", f"{max_risk_usdt:.2f} USDT")
    
    # === CURRENT MARKET STATUS ===
    st.subheader("📊 Huidige Marktsituatie")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{name1}", f"{current_price1:.8f} USDT")
    with col2:
        st.metric(f"{name2}", f"{current_price2:.8f} USDT")
    with col3:
        color = "normal"
        if abs(current_zscore) >= zscore_entry_threshold:
            color = "inverse"
        st.metric("Z-Score", f"{current_zscore:.2f}", 
                 delta=f"{'TRADE!' if abs(current_zscore) >= zscore_entry_threshold else 'Wacht'}", 
                 delta_color=color)
    with col4:
        st.metric("Hedge Ratio", f"{beta:.6f}")
    
    # === IMPROVED POSITION SIZING ===
    
    def calculate_balanced_positions(capital, price1, price2, hedge_ratio, max_position_ratio=3.0):
        """
        Calculate balanced positions that respect both capital limits and hedge ratio
        
        For pairs trading:
        - Long Asset1: quantity1 * price1 = dollar_amount1
        - Short Asset2: quantity2 * price2 = dollar_amount2
        - Hedge constraint: quantity1 * hedge_ratio ≈ quantity2 (for correlation)
        
        But we want balanced dollar amounts, so we use leverage on the smaller position
        """
        
        # Start with 50/50 capital split
        capital_per_leg = capital * 0.49  # 98% of capital, 49% per leg
        
        # Calculate base quantities with equal dollar amounts
        base_quantity1 = capital_per_leg / price1
        base_quantity2 = capital_per_leg / price2
        
        # Calculate what hedge ratio would give us with equal dollars
        actual_ratio = base_quantity2 / base_quantity1 if base_quantity1 != 0 else 0
        
        # Compare with theoretical hedge ratio
        ratio_difference = abs(actual_ratio - abs(hedge_ratio)) / abs(hedge_ratio) if hedge_ratio != 0 else float('inf')
        
        # If ratios are very different, use leverage/adjustment
        if ratio_difference > 0.5 or actual_ratio == 0:  # More than 50% difference
            
            # Method 1: Use theoretical hedge ratio with leverage on smaller position
            if abs(hedge_ratio) < 1:  # Asset2 needs more leverage
                quantity1 = capital_per_leg / price1
                quantity2 = quantity1 * abs(hedge_ratio)
                
                # If this makes quantity2 too expensive, add leverage factor
                cost2 = quantity2 * price2
                if cost2 > capital_per_leg * max_position_ratio:
                    # Scale down to manageable size
                    scale_factor = (capital_per_leg * max_position_ratio) / cost2
                    quantity2 *= scale_factor
                    quantity1 *= scale_factor
                    
            else:  # Asset1 needs more leverage  
                quantity2 = capital_per_leg / price2
                quantity1 = quantity2 / abs(hedge_ratio)
                
                # If this makes quantity1 too expensive, add leverage factor
                cost1 = quantity1 * price1
                if cost1 > capital_per_leg * max_position_ratio:
                    scale_factor = (capital_per_leg * max_position_ratio) / cost1
                    quantity1 *= scale_factor
                    quantity2 *= scale_factor
                    
        else:
            # Hedge ratio is reasonable, use equal dollar amounts
            quantity1 = base_quantity1
            quantity2 = base_quantity2
        
        # Calculate final costs
        final_cost1 = quantity1 * price1
        final_cost2 = quantity2 * price2
        
        return quantity1, quantity2, final_cost1, final_cost2, ratio_difference
    
    # === PRICE LEVEL CALCULATIONS ===
    
    # Calculate exact price levels for exits based on Z-score targets
    exit_zscore_long = -zscore_exit_threshold
    exit_zscore_short = zscore_exit_threshold
    stoploss_zscore_long = zscore_entry_threshold * 1.5  # More aggressive stop
    stoploss_zscore_short = -zscore_entry_threshold * 1.5
    
    # Convert Z-scores back to spread levels
    exit_spread_long = spread_mean + exit_zscore_long * spread_std
    exit_spread_short = spread_mean + exit_zscore_short * spread_std
    stoploss_spread_long = spread_mean + stoploss_zscore_long * spread_std
    stoploss_spread_short = spread_mean + stoploss_zscore_short * spread_std
    
    def calculate_price_levels_for_spread(target_spread, current_price1, current_price2, hedge_ratio):
        """
        Calculate what individual asset prices should be for a target spread
        More robust version that gives multiple scenarios
        """
        
        # Scenario 1: Price1 moves 10%, calculate required Price2
        price1_up = current_price1 * 1.1
        price1_down = current_price1 * 0.9
        
        price2_for_spread_up = target_spread + alpha + hedge_ratio * price1_up
        price2_for_spread_down = target_spread + alpha + hedge_ratio * price1_down
        
        # Scenario 2: Price2 moves 10%, calculate required Price1  
        price2_up = current_price2 * 1.1
        price2_down = current_price2 * 0.9
        
        price1_for_spread_up = (price2_up - alpha - target_spread) / hedge_ratio if hedge_ratio != 0 else current_price1
        price1_for_spread_down = (price2_down - alpha - target_spread) / hedge_ratio if hedge_ratio != 0 else current_price1
        
        return {
            'price1_scenarios': {
                'if_price1_up_10pct': price2_for_spread_up,
                'if_price1_down_10pct': price2_for_spread_down
            },
            'price2_scenarios': {
                'if_price2_up_10pct': price1_for_spread_up,
                'if_price2_down_10pct': price1_for_spread_down
            }
        }
    
    # === TRADING DECISION ===
    st.markdown("---")
    
    # Determine signal and show execution with PROPER POSITION SIZING
    if current_zscore <= -zscore_entry_threshold:
        # LONG SPREAD SIGNAL
        st.success(f"🟢 **KOOP SIGNAAL** - Z-Score: {current_zscore:.2f}")
        
        # Calculate balanced positions
        shares_asset1, shares_asset2, cost_asset1, cost_asset2, ratio_diff = calculate_balanced_positions(
            trading_capital, current_price1, current_price2, beta
        )
        
        total_cost = cost_asset1 + cost_asset2
        
        st.markdown("### 📈 EXACTE TRADE UITVOERING:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            #### 🟢 KOOP {name1}
            - **Aantal**: {shares_asset1:.4f} {name1}
            - **Huidige Prijs**: {current_price1:.8f} USDT
            - **Totaal**: {cost_asset1:.6f} USDT
            - **Order Type**: MARKET BUY
            """)
            
        with col2:
            st.markdown(f"""
            #### 🔴 VERKOOP {name2} (SHORT)
            - **Aantal**: {shares_asset2:.0f} {name2}
            - **Huidige Prijs**: {current_price2:.8f} USDT  
            - **Totaal**: {cost_asset2:.6f} USDT
            - **Order Type**: MARKET SELL (SHORT)
            """)
        
        # Position quality check
        position_ratio = max(cost_asset1, cost_asset2) / min(cost_asset1, cost_asset2) if min(cost_asset1, cost_asset2) > 0 else float('inf')
        
        if position_ratio > 5:
            st.warning(f"""
            ⚠️ **HEDGE RATIO WAARSCHUWING**
            
            **Positie verhouding**: {position_ratio:.1f}:1 (ideaal < 3:1)
            **Hedge ratio verschil**: {ratio_diff*100:.1f}% van theorie
            
            **Opties voor betere balans:**
            1. **Gebruik LEVERAGE op kleinere positie** (2x-5x)
            2. **Kies andere paren** met betere correlatie
            3. **Trade met hoger capital** (≥€{int(trading_capital * 3)})
            """)
        
        st.info(f"**Totaal Gebruikt**: {total_cost:.2f} USDT van {trading_capital:.2f} USDT | **Efficiency**: {(total_cost/trading_capital)*100:.1f}%")
        
        # === PRICE ALERTS VOOR LONG SPREAD ===
        st.markdown("### 🚨 STEL DEZE PRICE ALERTS IN:")
        
        # Calculate exit price scenarios
        profit_scenarios = calculate_price_levels_for_spread(exit_spread_long, current_price1, current_price2, beta)
        stoploss_scenarios = calculate_price_levels_for_spread(stoploss_spread_long, current_price1, current_price2, beta)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("#### 🎯 PROFIT TARGET ALERTS")
            st.markdown(f"""
            **Sluit LONG spread wanneer BEIDE voorwaarden:**
            
            **Scenario A: Als {name1} weinig beweegt**
            - {name1} rond {current_price1:.8f} USDT
            - **{name2} ≤ {profit_scenarios['price1_scenarios']['if_price1_up_10pct']:.8f} USDT**
            
            **Scenario B: Als {name2} weinig beweegt**  
            - {name2} rond {current_price2:.8f} USDT
            - **{name1} ≥ {profit_scenarios['price2_scenarios']['if_price2_down_10pct']:.8f} USDT**
            
            **🎯 Target Z-score: {exit_zscore_long:.1f}**
            """)
            
        with col2:
            st.error("#### 🛑 STOP LOSS ALERTS")  
            st.markdown(f"""
            **EMERGENCY EXIT wanneer:**
            
            **Scenario A: Als {name1} weinig beweegt**
            - {name1} rond {current_price1:.8f} USDT
            - **{name2} ≥ {stoploss_scenarios['price1_scenarios']['if_price1_down_10pct']:.8f} USDT**
            
            **Scenario B: Als {name2} weinig beweegt**
            - {name2} rond {current_price2:.8f} USDT  
            - **{name1} ≤ {stoploss_scenarios['price2_scenarios']['if_price2_up_10pct']:.8f} USDT**
            
            **🛑 Max Verlies: -{max_risk_usdt:.2f} USDT**
            """)
        
    elif current_zscore >= zscore_entry_threshold:
        # SHORT SPREAD SIGNAL  
        st.error(f"🔴 **VERKOOP SIGNAAL** - Z-Score: {current_zscore:.2f}")
        
        # Calculate balanced positions for short spread
        shares_asset1, shares_asset2, cost_asset1, cost_asset2, ratio_diff = calculate_balanced_positions(
            trading_capital, current_price1, current_price2, beta
        )
        
        total_cost = cost_asset1 + cost_asset2
        
        st.markdown("### 📉 EXACTE TRADE UITVOERING:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            #### 🔴 VERKOOP {name1} (SHORT)
            - **Aantal**: {shares_asset1:.4f} {name1}
            - **Huidige Prijs**: {current_price1:.8f} USDT
            - **Totaal**: {cost_asset1:.6f} USDT
            - **Order Type**: MARKET SELL (SHORT)
            """)
            
        with col2:
            st.markdown(f"""
            #### 🟢 KOOP {name2}
            - **Aantal**: {shares_asset2:.0f} {name2}
            - **Huidige Prijs**: {current_price2:.8f} USDT
            - **Totaal**: {cost_asset2:.6f} USDT
            - **Order Type**: MARKET BUY
            """)
        
        # Position quality check for short spread
        position_ratio = max(cost_asset1, cost_asset2) / min(cost_asset1, cost_asset2) if min(cost_asset1, cost_asset2) > 0 else float('inf')
        
        if position_ratio > 5:
            st.warning(f"""
            ⚠️ **HEDGE RATIO ISSUE** - Positie verhouding: {position_ratio:.1f}:1
            **Overweeg LEVERAGE** op kleinere positie voor betere hedge.
            """)
        
        st.info(f"**Totaal Gebruikt**: {total_cost:.2f} USDT van {trading_capital:.2f} USDT | **Efficiency**: {(total_cost/trading_capital)*100:.1f}%")
        
        # === PRICE ALERTS VOOR SHORT SPREAD ===
        st.markdown("### 🚨 STEL DEZE PRICE ALERTS IN:")
        
        # Calculate exit price scenarios for short spread
        profit_scenarios_short = calculate_price_levels_for_spread(exit_spread_short, current_price1, current_price2, beta)
        stoploss_scenarios_short = calculate_price_levels_for_spread(stoploss_spread_short, current_price1, current_price2, beta)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("#### 🎯 PROFIT TARGET ALERTS")
            st.markdown(f"""
            **Sluit SHORT spread wanneer:**
            
            **Scenario A: Als {name1} weinig beweegt**
            - {name1} rond {current_price1:.8f} USDT
            - **{name2} ≥ {profit_scenarios_short['price1_scenarios']['if_price1_down_10pct']:.8f} USDT**
            
            **Scenario B: Als {name2} weinig beweegt**
            - {name2} rond {current_price2:.8f} USDT
            - **{name1} ≤ {profit_scenarios_short['price2_scenarios']['if_price2_up_10pct']:.8f} USDT**
            
            **🎯 Target Z-score: {exit_zscore_short:.1f}**
            """)
            
        with col2:
            st.error("#### 🛑 STOP LOSS ALERTS")
            st.markdown(f"""
            **EMERGENCY EXIT wanneer:**
            
            **Scenario A: Als {name1} weinig beweegt**
            - {name1} rond {current_price1:.8f} USDT
            - **{name2} ≤ {stoploss_scenarios_short['price1_scenarios']['if_price1_up_10pct']:.8f} USDT**
            
            **Scenario B: Als {name2} weinig beweegt**
            - {name2} rond {current_price2:.8f} USDT
            - **{name1} ≥ {stoploss_scenarios_short['price2_scenarios']['if_price2_down_10pct']:.8f} USDT**
            
            **🛑 Max Verlies: -{max_risk_usdt:.2f} USDT**
            """)
        
    else:
        # NO SIGNAL
        st.info(f"⏳ **GEEN SIGNAAL** - Z-Score: {current_zscore:.2f}")
        
        distance_to_long = abs(current_zscore - (-zscore_entry_threshold))
        distance_to_short = abs(current_zscore - zscore_entry_threshold)
        
        st.markdown(f"""
        ### ⌛ WACHT OP SIGNAAL
        
        **Entry Levels:**
        - 🟢 LONG SPREAD bij Z-score ≤ -{zscore_entry_threshold:.1f} (nog {distance_to_long:.2f} punten)
        - 🔴 SHORT SPREAD bij Z-score ≥ +{zscore_entry_threshold:.1f} (nog {distance_to_short:.2f} punten)
        
        **Huidige Status**: Neutrale zone, monitor prijsbeweging
        """)

    # === LEVERAGE OPTIMIZATION SECTION ===
    st.markdown("---")
    st.subheader("⚡ Leverage Optimalisatie voor Betere Hedge")
    
    with st.expander("📊 Leverage Strategieën voor Ongebalanceerde Paren", expanded=False):
        
        # Calculate what leverage would balance the positions
        if trading_capital > 0:
            test_qty1, test_qty2, test_cost1, test_cost2, test_ratio_diff = calculate_balanced_positions(
                trading_capital, current_price1, current_price2, beta, max_position_ratio=10
            )
            
            if test_cost1 > 0 and test_cost2 > 0:
                current_ratio = max(test_cost1, test_cost2) / min(test_cost1, test_cost2)
                
                if current_ratio > 3:
                    # Calculate optimal leverage
                    smaller_cost = min(test_cost1, test_cost2)
                    larger_cost = max(test_cost1, test_cost2)
                    optimal_leverage = larger_cost / smaller_cost
                    
                    st.markdown(f"""
                    ### 🎯 Leverage Aanbevelingen voor {name1} vs {name2}
                    
                    **Huidige situatie:**
                    - Grotere positie: {larger_cost:.2f} USDT
                    - Kleinere positie: {smaller_cost:.2f} USDT  
                    - Verhouding: {current_ratio:.1f}:1
                    
                    **Oplossing met Leverage:**
                    - **{optimal_leverage:.1f}x leverage** op kleinere positie
                    - Dit geeft balans van ~1:1 in dollar termen
                    - **Futures/Margin trading** vereist
                    
                    **Alternatieve Strategieën:**
                    1. **ETFs gebruiken** (bijv. crypto ETFs met balans)
                    2. **Hoger capital** (minimaal €{int(trading_capital * 3)})
                    3. **Andere paren kiezen** met natuurlijke balans
                    4. **Opties strategies** voor capital efficiency
                    """)
                    
                    # Show leverage calculation
                    leveraged_cost = smaller_cost * optimal_leverage
                    st.success(f"""
                    **Met {optimal_leverage:.1f}x Leverage:**
                    - Kleinere positie wordt: {leveraged_cost:.2f} USDT
                    - Nieuwe verhouding: ~1:1 (perfect gebalanceerd)
                    - **Margin vereist**: {leveraged_cost - smaller_cost:.2f} USDT extra
                    """)
                    
                else:
                    st.success(f"""
                    ✅ **GOED GEBALANCEERD PAAR**
                    
                    Verhouding {current_ratio:.1f}:1 is acceptabel voor pairs trading.
                    Geen leverage optimalisatie nodig.
                    """)
    
    # === EXCHANGE SPECIFIC INSTRUCTIONS ===
    st.markdown("---")
    st.subheader("🏪 Exchange Uitvoering (Binance/Bybit/etc)")
    
    with st.expander("📱 Stap-voor-stap Exchange Orders", expanded=False):
        
        if abs(current_zscore) >= zscore_entry_threshold:
            
            signal_type = "LONG SPREAD" if current_zscore <= -zscore_entry_threshold else "SHORT SPREAD"
            
            st.markdown(f"### 🎯 {signal_type} - Exchange Orders:")
            
            if current_zscore <= -zscore_entry_threshold:
                # LONG SPREAD instructions
                st.markdown(f"""
                **STAP 1: KOOP {name1}**
                - Ga naar **{name1}/USDT** spot trading
                - Order Type: **MARKET BUY**  
                - Quantity: **{shares_asset1:.4f} {name1}**
                - Est. Cost: **~{cost_asset1:.2f} USDT**
                
                **STAP 2: SHORT {name2}**
                - Ga naar **{name2}/USDT** futures/margin
                - Order Type: **MARKET SELL** (Open Short)
                - Quantity: **{shares_asset2:.0f} {name2}**
                - Leverage: **1x-5x** (afhankelijk van balans)
                - Margin needed: **~{cost_asset2:.2f} USDT**
                
                **STAP 3: SET ALERTS**
                - **TradingView**: {name2} price alerts
                - **Exchange**: Conditional orders
                - **Mobile**: Push notifications
                """)
            else:
                # SHORT SPREAD instructions
                st.markdown(f"""
                **STAP 1: SHORT {name1}**
                - Ga naar **{name1}/USDT** futures/margin
                - Order Type: **MARKET SELL** (Open Short)
                - Quantity: **{shares_asset1:.4f} {name1}**
                - Leverage: **1x-5x** (afhankelijk van balans)
                - Margin needed: **~{cost_asset1:.2f} USDT**
                
                **STAP 2: KOOP {name2}**  
                - Ga naar **{name2}/USDT** spot trading
                - Order Type: **MARKET BUY**
                - Quantity: **{shares_asset2:.0f} {name2}**
                - Est. Cost: **~{cost_asset2:.2f} USDT**
                
                **STAP 3: SET ALERTS**
                - **Price alerts** voor beide assets
                - **Conditional close orders** 
                - **Risk management** alerts
                """)
                
        else:
            st.info("⏳ Nog geen signaal - stel price alerts in om automatisch genotificeerd te worden")
    
    # === RISK WARNING ===
    st.markdown("---")
    st.warning(f"""
    ⚠️ **TRADING RISICO'S CRYPTO PAIRS:**
    - **Max verlies per trade**: {max_risk_usdt:.2f} USDT ({risk_per_trade}% van capital)
    - **Leverage amplifies risk**: 5x leverage = 5x profit/loss
    - **Funding fees**: Futures posities hebben dagelijkse kosten
    - **Liquidation risk**: Bij te hoge leverage of margin calls
    - **Correlation breakdown**: Crypto pairs kunnen snel decorreleren
    - **Extreme volatility**: 50%+ bewegingen mogelijk op crypto
    
    **Crypto-specifieke tips:**
    - **Start klein**: Test eerst met €50-100 
    - **Funding arbitrage**: Let op funding rates difference
    - **News events**: DeFi/regulation news kan alles verstoren
    - **Weekend gaps**: Crypto tradet 24/7, traditionele pairs niet
    - **Slippage**: Bij grote orders, gebruik limit orders
    """)
    
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
