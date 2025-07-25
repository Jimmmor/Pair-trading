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

# === TRADING SIGNALS SECTION - FIXED ===
with st.expander("Trading Signals - Praktische Uitvoering", expanded=True):
    st.header("Praktische Trade Uitvoering")
    
    # Gebruik de al berekende Z-score waardes (consistent met statistiek sectie)
    current_price1 = df['price1'].iloc[-1]
    current_price2 = df['price2'].iloc[-1]
    current_zscore = df['zscore'].iloc[-1]  # Deze is al berekend met consistente parameters
    
    # Gebruik de al berekende spread statistieken
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]  # hedge ratio
    
    # Gebruik de al berekende spread mean en std (consistent)
    spread_mean = df['spread'].mean()  # Zelfde als in statistiek sectie
    spread_std = df['spread'].std()    # Zelfde als in statistiek sectie
    
    # Bereken fair value op basis van consistente parameters
    current_spread = current_price2 - (alpha + beta * current_price1)
    fair_value_spread = spread_mean  # Target spread
    fair_value2 = current_price1 * beta + alpha + fair_value_spread
    
    # Toon huidige marktsituatie
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{name1} Prijs", f"${current_price1:.4f}")
    with col2:
        st.metric(f"{name2} Prijs", f"${current_price2:.4f}")
    with col3:
        st.metric("Current Z-score", f"{current_zscore:.2f}")
    with col4:
        st.metric("Current Spread", f"${current_spread:.4f}")
    
    # Trade execution parameters - consistent Z-score logic
    if current_zscore < -zscore_entry_threshold:
        st.success(f"**LONG SPREAD SIGNAL (Z = {current_zscore:.2f})**")
        st.markdown(f"""
        ### 📈 Uitvoering LONG SPREAD:
        1. **Koop {name1}** tegen huidige prijs: ${current_price1:.4f}
        2. **Verkoop {beta:.4f} eenheden {name2}** per {name1} tegen: ${current_price2:.4f}
        3. **Hedge Ratio**: 1 {name1} = {beta:.4f} {name2}
        
        ### 🎯 Trading Logic:
        - **Huidige spread**: ${current_spread:.4f} (te laag)
        - **Gemiddelde spread**: ${spread_mean:.4f} (target)
        - **Verwacht herstel**: Spread stijgt naar gemiddelde
        - **Exit bij Z-score**: -{zscore_exit_threshold:.1f} (spread normaliseert)
        
        ### 💡 Waarom deze trade?
        Z-score van {current_zscore:.2f} betekent dat de spread {abs(current_zscore):.1f} standaarddeviaties onder het gemiddelde ligt.
        Historisch keert dit terug naar het gemiddelde.
        """)
        
    elif current_zscore > zscore_entry_threshold:
        st.error(f"**SHORT SPREAD SIGNAL (Z = {current_zscore:.2f})**")
        st.markdown(f"""
        ### 📉 Uitvoering SHORT SPREAD:
        1. **Verkoop {name1}** tegen huidige prijs: ${current_price1:.4f}
        2. **Koop {beta:.4f} eenheden {name2}** per {name1} tegen: ${current_price2:.4f}
        3. **Hedge Ratio**: 1 {name1} = {beta:.4f} {name2}
        
        ### 🎯 Trading Logic:
        - **Huidige spread**: ${current_spread:.4f} (te hoog)
        - **Gemiddelde spread**: ${spread_mean:.4f} (target)
        - **Verwacht herstel**: Spread daalt naar gemiddelde
        - **Exit bij Z-score**: {zscore_exit_threshold:.1f} (spread normaliseert)
        
        ### 💡 Waarom deze trade?
        Z-score van {current_zscore:.2f} betekent dat de spread {current_zscore:.1f} standaarddeviaties boven het gemiddelde ligt.
        Historisch keert dit terug naar het gemiddelde.
        """)
    else:
        st.info(f"**GEEN SIGNAL (Z = {current_zscore:.2f})**")
        
        # Toon afstand tot entry levels
        distance_to_long = abs(current_zscore - (-zscore_entry_threshold))
        distance_to_short = abs(current_zscore - zscore_entry_threshold)
        
        st.markdown(f"""
        ### ⏳ Wacht op entry signaal:
        - **LONG entry** bij Z < -{zscore_entry_threshold:.1f} (nog {distance_to_long:.2f} punten)
        - **SHORT entry** bij Z > {zscore_entry_threshold:.1f} (nog {distance_to_short:.2f} punten)
        
        ### 📊 Huidige status:
        - Spread is {abs(current_zscore):.1f} standaarddeviaties van gemiddelde
        - {"Boven" if current_zscore > 0 else "Onder"} gemiddelde spread
        """)
    
    # Toon praktische trading levels - consistent met hoofdberekening
    st.markdown("---")
    st.subheader("Trading Parameters (Consistent)")
    
    # Bereken entry/exit levels in werkelijke prijzen
    long_entry_spread = spread_mean - zscore_entry_threshold * spread_std
    long_exit_spread = spread_mean - zscore_exit_threshold * spread_std
    short_entry_spread = spread_mean + zscore_entry_threshold * spread_std
    short_exit_spread = spread_mean + zscore_exit_threshold * spread_std
    
    col1, col2 = st.columns(2)
    
    with col1:
        entry_levels = pd.DataFrame({
            'Parameter': [
                'Alpha (α)',
                'Beta/Hedge Ratio (β)', 
                'Spread Mean',
                'Spread Std Dev',
                'Current Spread',
                'Current Z-score'
            ],
            'Waarde': [
                f"{alpha:.6f}",
                f"{beta:.4f}",
                f"${spread_mean:.4f}",
                f"${spread_std:.4f}",
                f"${current_spread:.4f}",
                f"{current_zscore:.2f}"
            ]
        })
        st.table(entry_levels)
    
    with col2:
        trading_levels = pd.DataFrame({
            'Level': [
                'Long Entry Spread',
                'Long Exit Spread', 
                'Short Entry Spread',
                'Short Exit Spread'
            ],
            'Z-score': [
                f"-{zscore_entry_threshold:.1f}",
                f"-{zscore_exit_threshold:.1f}",
                f"+{zscore_entry_threshold:.1f}",
                f"+{zscore_exit_threshold:.1f}"
            ],
            'Spread Value': [
                f"${long_entry_spread:.4f}",
                f"${long_exit_spread:.4f}",
                f"${short_entry_spread:.4f}",
                f"${short_exit_spread:.4f}"
            ]
        })
        st.table(trading_levels)
    
    # FIXED: Gebruik de al berekende Z-score (consistent met statistiek sectie)
    st.subheader("Z-score Trading Chart (Consistent)")
    
    fig = go.Figure()
    
    # Z-score lijn (gebruik al berekende waardes)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['zscore'],  # Deze is al correct berekend in het begin
        name='Z-score',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Z-score</b>: %{y:.2f}<extra></extra>'
    ))
    
    # Mean lijn (bij z-score = 0)
    fig.add_hline(
        y=0,
        line=dict(color='black', width=1),
        annotation_text="Mean (Z=0)",
        annotation_position="top right"
    )
    
    # Entry levels
    fig.add_hline(
        y=zscore_entry_threshold,
        line=dict(color='red', dash='dash', width=2),
        annotation_text=f"SHORT Entry (Z=+{zscore_entry_threshold})",
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=-zscore_entry_threshold,
        line=dict(color='green', dash='dash', width=2),
        annotation_text=f"LONG Entry (Z=-{zscore_entry_threshold})",
        annotation_position="bottom right"
    )
    
    # Exit levels
    fig.add_hline(
        y=zscore_exit_threshold,
        line=dict(color='pink', dash='dot', width=1),
        annotation_text=f"SHORT Exit (Z=+{zscore_exit_threshold})",
        annotation_position="top left"
    )
    
    fig.add_hline(
        y=-zscore_exit_threshold,
        line=dict(color='lightgreen', dash='dot', width=1),
        annotation_text=f"LONG Exit (Z=-{zscore_exit_threshold})",
        annotation_position="bottom left"
    )
    
    # Markeer huidige positie
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[current_zscore],
        mode='markers',
        marker=dict(
            size=15,
            color='yellow',
            symbol='star',
            line=dict(color='black', width=2)
        ),
        name=f'Current (Z={current_zscore:.2f})',
        hovertemplate=f'<b>Current Position</b><br>Z-score: {current_zscore:.2f}<extra></extra>'
    ))
    
    # Kleur de trading zones
    fig.add_hrect(
        y0=zscore_entry_threshold, y1=5,
        fillcolor="rgba(255,0,0,0.1)",
        layer="below", line_width=0,
        annotation_text="SHORT Zone", 
        annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=-5, y1=-zscore_entry_threshold,
        fillcolor="rgba(0,255,0,0.1)",
        layer="below", line_width=0,
        annotation_text="LONG Zone", 
        annotation_position="bottom left"
    )
    
    # Neutrale zone
    fig.add_hrect(
        y0=-zscore_entry_threshold, y1=zscore_entry_threshold,
        fillcolor="rgba(128,128,128,0.05)",
        layer="below", line_width=0
    )
    
    fig.update_layout(
        title=f"Z-score Trading Levels - {name1} vs {name2} (Consistent Calculation)",
        xaxis_title="Date",
        yaxis_title="Z-score",
        height=500,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Voeg een validatie toe
    st.markdown("---")
    with st.expander("🔍 Calculation Validation", expanded=False):
        st.markdown("""
        **Validation Check**: Deze Z-scores zijn nu consistent met de Statistical Analysis sectie.
        
        **Berekening**:
        1. Spread = Price2 - (α + β × Price1)  
        2. Z-score = (Spread - Mean(Spread)) / Std(Spread)
        3. Gebruik dezelfde parameters door hele app
        
        **Key Parameters**:
        """)
        
        validation_df = pd.DataFrame({
            'Parameter': ['Alpha (α)', 'Beta (β)', 'Spread Mean', 'Spread Std', 'Current Spread', 'Current Z-score'],
            'Value': [f"{alpha:.6f}", f"{beta:.4f}", f"{spread_mean:.6f}", f"{spread_std:.6f}", f"{current_spread:.6f}", f"{current_zscore:.4f}"],
            'Used In': ['Spread calc', 'Spread calc', 'Z-score calc', 'Z-score calc', 'Display', 'Trading logic']
        })
        
        st.dataframe(validation_df, use_container_width=True)
# === ENHANCED BACKTESTING SECTION ===
def run_advanced_backtest(df, entry_threshold, exit_threshold, initial_capital=100000, 
                         transaction_cost=0.1, max_position_size=50, stop_loss_pct=10, 
                         take_profit_pct=20, lookback_period=30):
    """
    Verbeterde backtest met realistische prijs-gebaseerde berekeningen
    """
    # Consistent met de rest van de app - gebruik dezelfde spread berekening
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]  # hedge ratio
    
    df = df.copy()
    df['spread'] = df['price2'] - (alpha + beta * df['price1'])
    
    # Rolling statistics voor adaptieve thresholds
    df['spread_mean'] = df['spread'].rolling(lookback_period).mean()
    df['spread_std'] = df['spread'].rolling(lookback_period).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # Trading state variables
    cash = initial_capital
    position = 0  # 0=geen positie, 1=long spread, -1=short spread
    shares1 = 0   # aantal aandelen asset 1
    shares2 = 0   # aantal aandelen asset 2  
    entry_price1 = 0
    entry_price2 = 0
    entry_date = None
    entry_zscore = 0
    max_position_value = (max_position_size / 100) * initial_capital
    
    # Tracking arrays
    trades = []
    portfolio_values = []
    positions = []
    drawdowns = []
    peak_value = initial_capital

    for i in range(lookback_period, len(df)):
        current_zscore = df['zscore'].iloc[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_date = df.index[i]
        
        # Bereken huidige portfolio waarde op basis van werkelijke prijzen
        if position != 0:
            position_value1 = shares1 * current_price1  # waarde positie asset 1
            position_value2 = shares2 * current_price2  # waarde positie asset 2
            total_position_value = position_value1 + position_value2
            portfolio_value = cash + total_position_value
        else:
            portfolio_value = cash
            total_position_value = 0
        
        portfolio_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'position_value': total_position_value,
            'price1': current_price1,
            'price2': current_price2,
            'zscore': current_zscore
        })
        
        positions.append(position)
        
        # Track drawdown
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        drawdown = (peak_value - portfolio_value) / peak_value * 100
        drawdowns.append(drawdown)
        
        # ENTRY LOGIC - alleen als geen positie open staat
        if position == 0 and not pd.isna(current_zscore):
            
            if current_zscore < -entry_threshold:  # LONG SPREAD ENTRY
                position = 1
                entry_date = current_date
                entry_zscore = current_zscore
                entry_price1 = current_price1
                entry_price2 = current_price2
                
                # Bereken positie grootte gebaseerd op beschikbaar kapitaal
                available_capital = min(max_position_value, cash * 0.95)  # 5% cash buffer
                
                # Long spread = Long asset1, Short asset2
                # Verdeel kapitaal over beide posities volgens hedge ratio
                capital_asset1 = available_capital / (1 + abs(beta))
                capital_asset2 = available_capital - capital_asset1
                
                shares1 = capital_asset1 / current_price1  # LONG asset 1
                shares2 = -(capital_asset2 / current_price2) * (beta / abs(beta))  # SHORT asset 2, correct hedge
                
                # Transaction costs op basis van totale notional value
                notional_value = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                transaction_fee = notional_value * (transaction_cost / 100)
                cash -= transaction_fee
                
            elif current_zscore > entry_threshold:  # SHORT SPREAD ENTRY
                position = -1
                entry_date = current_date
                entry_zscore = current_zscore
                entry_price1 = current_price1
                entry_price2 = current_price2
                
                available_capital = min(max_position_value, cash * 0.95)
                
                # Short spread = Short asset1, Long asset2
                capital_asset1 = available_capital / (1 + abs(beta))
                capital_asset2 = available_capital - capital_asset1
                
                shares1 = -(capital_asset1 / current_price1)  # SHORT asset 1
                shares2 = (capital_asset2 / current_price2) * (beta / abs(beta))   # LONG asset 2, correct hedge
                
                notional_value = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                transaction_fee = notional_value * (transaction_cost / 100)
                cash -= transaction_fee
        
        # EXIT LOGIC
        elif position != 0 and not pd.isna(current_zscore):
            exit_trade = False
            exit_reason = ""
            
            # Bereken huidige P&L op basis van werkelijke prijzen
            current_position_value = shares1 * current_price1 + shares2 * current_price2
            entry_position_value = shares1 * entry_price1 + shares2 * entry_price2
            unrealized_pnl = current_position_value - entry_position_value
            pnl_pct = (unrealized_pnl / abs(entry_position_value)) * 100 if entry_position_value != 0 else 0
            
            # Z-score mean reversion exit
            if (position == 1 and current_zscore > -exit_threshold) or \
               (position == -1 and current_zscore < exit_threshold):
                exit_trade = True
                exit_reason = "Mean reversion"
            
            # Stop loss op basis van percentage verlies
            elif pnl_pct < -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            
            # Take profit
            elif pnl_pct > take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            # Time-based exit (maximum 60 dagen)
            elif (current_date - entry_date).days > 60:
                exit_trade = True
                exit_reason = "Time exit"
            
            # Z-score extreme omkering (risk management)
            elif (position == 1 and current_zscore > entry_threshold) or \
                 (position == -1 and current_zscore < -entry_threshold):
                exit_trade = True
                exit_reason = "Z-score reversal"
            
            if exit_trade:
                # Sluit positie - bereken finale P&L
                exit_value = shares1 * current_price1 + shares2 * current_price2
                
                # Transaction costs bij exit
                notional_value = abs(shares1 * current_price1) + abs(shares2 * current_price2)
                exit_transaction_fee = notional_value * (transaction_cost / 100)
                
                # Netto P&L na transaction costs
                net_pnl = exit_value - exit_transaction_fee
                final_pnl = net_pnl  # Dit is de werkelijke P&L van de trade
                
                # Update cash
                cash += net_pnl
                
                # Record trade
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Position': 'Long Spread' if position == 1 else 'Short Spread',
                    'Entry Z-score': entry_zscore,
                    'Exit Z-score': current_zscore,
                    'Entry Price 1': entry_price1,
                    'Entry Price 2': entry_price2,
                    'Exit Price 1': current_price1,
                    'Exit Price 2': current_price2,
                    'Shares 1': shares1,
                    'Shares 2': shares2,
                    'Entry Position Value': abs(entry_position_value),
                    'Exit Position Value': abs(exit_value),
                    'Gross P&L': unrealized_pnl,
                    'Transaction Costs': exit_transaction_fee,
                    'Net P&L': final_pnl,
                    'P&L %': (final_pnl / abs(entry_position_value)) * 100 if entry_position_value != 0 else 0,
                    'Exit Reason': exit_reason,
                    'Days Held': (current_date - entry_date).days,
                    'Price Change 1': ((current_price1 - entry_price1) / entry_price1) * 100,
                    'Price Change 2': ((current_price2 - entry_price2) / entry_price2) * 100
                })
                
                # Reset position
                position = 0
                shares1 = 0
                shares2 = 0
                entry_price1 = 0
                entry_price2 = 0
                entry_date = None
                entry_zscore = 0
    
    # Converteer naar DataFrames
    portfolio_df = pd.DataFrame(portfolio_values)
    portfolio_df.set_index('date', inplace=True)
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return portfolio_df, trades_df, drawdowns

# === BACKTEST EXECUTION ===
with st.expander("🔍 Advanced Backtest Analysis", expanded=True):
    st.header("🔍 Advanced Backtest Analysis")
    
    # Backtest parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
        transaction_cost = st.slider("Transaction Cost (%)", 0.01, 1.0, 0.1, 0.01)
    with col2:
        max_position_size = st.slider("Max Position Size (%)", 10, 100, 50, 5)
        stop_loss_pct = st.slider("Stop Loss (%)", 5, 30, 15, 1)
    with col3:
        take_profit_pct = st.slider("Take Profit (%)", 10, 50, 25, 1)
        lookback_period = st.slider("Lookback Period (days)", 10, 60, 30, 5)
    
    # Run backtest
    with st.spinner("Running backtest..."):
        portfolio_df, trades_df, drawdowns = run_advanced_backtest(
            df, zscore_entry_threshold, zscore_exit_threshold, 
            initial_capital, transaction_cost, max_position_size, 
            stop_loss_pct, take_profit_pct, lookback_period
        )
    
    if not trades_df.empty:
        # Performance metrics
        total_return = ((portfolio_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital) * 100
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['Net P&L'] > 0])
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        avg_win = trades_df[trades_df['Net P&L'] > 0]['Net P&L'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['Net P&L'] < 0]['Net P&L'].mean() if (num_trades - winning_trades) > 0 else 0
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate Sharpe ratio
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        if len(portfolio_returns) > 1 and portfolio_returns.std() > 0:
            sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Display key metrics
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col2:
            st.metric("Total Trades", num_trades)
            st.metric("Winning Trades", winning_trades)
        with col3:
            st.metric("Avg Win", f"${avg_win:.2f}")
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Portfolio value chart
        st.subheader("📈 Portfolio Value Over Time")
        fig_portfolio = go.Figure()
        
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            fill='tonexty',
            fillcolor='rgba(0,100,255,0.1)'
        ))
        
        # Add benchmark (buy and hold asset 1)
        benchmark_value = initial_capital * (df['price1'] / df['price1'].iloc[lookback_period])
        benchmark_aligned = benchmark_value.reindex(portfolio_df.index, method='nearest')
        
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=benchmark_aligned,
            name=f'{name1} Buy & Hold',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig_portfolio.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Trade analysis
        st.subheader("📋 Trade Analysis")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Trades", "Performance Analysis", "Trade Distribution"])
        
        with tab1:
            # Format trades dataframe for display
            display_trades = trades_df.copy()
            for col in ['Entry Price 1', 'Entry Price 2', 'Exit Price 1', 'Exit Price 2']:
                display_trades[col] = display_trades[col].apply(lambda x: f"${x:.4f}")
            for col in ['Gross P&L', 'Net P&L', 'Transaction Costs']:
                display_trades[col] = display_trades[col].apply(lambda x: f"${x:.2f}")
            for col in ['P&L %', 'Price Change 1', 'Price Change 2']:
                display_trades[col] = display_trades[col].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_trades, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L distribution
                fig_pnl = px.histogram(
                    trades_df, x='Net P&L', nbins=20,
                    title="P&L Distribution",
                    labels={'Net P&L': 'Net P&L ($)', 'count': 'Number of Trades'}
                )
                fig_pnl.update_traces(marker_color='lightblue', marker_line_color='blue', marker_line_width=1)
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Win/Loss by position type
                win_loss_data = trades_df.groupby(['Position', trades_df['Net P&L'] > 0]).size().reset_index()
                win_loss_data['Result'] = win_loss_data['Net P&L'].map({True: 'Win', False: 'Loss'})
                
                fig_winloss = px.bar(
                    win_loss_data, x='Position', y=0, color='Result',
                    title="Wins vs Losses by Position Type",
                    labels={0: 'Number of Trades'}
                )
                st.plotly_chart(fig_winloss, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Days held distribution
                fig_days = px.histogram(
                    trades_df, x='Days Held', nbins=15,
                    title="Trade Duration Distribution",
                    labels={'Days Held': 'Days Held', 'count': 'Number of Trades'}
                )
                st.plotly_chart(fig_days, use_container_width=True)
            
            with col2:
                # Exit reason pie chart
                exit_reasons = trades_df['Exit Reason'].value_counts()
                fig_exit = px.pie(
                    values=exit_reasons.values, names=exit_reasons.index,
                    title="Exit Reasons Distribution"
                )
                st.plotly_chart(fig_exit, use_container_width=True)
        
        # Risk metrics
        st.subheader("⚠️ Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            risk_metrics = pd.DataFrame({
                'Metric': [
                    'Maximum Drawdown',
                    'Average Drawdown',
                    'Volatility (Daily)',
                    'Volatility (Annualized)',
                    'Value at Risk (95%)',
                    'Max Single Loss'
                ],
                'Value': [
                    f"{max_drawdown:.2f}%",
                    f"{np.mean(drawdowns):.2f}%",
                    f"{portfolio_returns.std()*100:.2f}%",
                    f"{portfolio_returns.std()*np.sqrt(252)*100:.2f}%",
                    f"${np.percentile(trades_df['Net P&L'], 5):.2f}" if len(trades_df) > 0 else "N/A",
                    f"${trades_df['Net P&L'].min():.2f}" if len(trades_df) > 0 else "N/A"
                ]
            })
            st.table(risk_metrics)
        
        with col2:
            # Drawdown chart
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=[-d for d in drawdowns],
                name='Drawdown',
                line=dict(color='red'),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)'
            ))
            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            st.plotly_chart(fig_dd, use_container_width=True)
    
    else:
        st.warning("No trades were generated during the backtest period. Try adjusting the Z-score thresholds or time period.")
