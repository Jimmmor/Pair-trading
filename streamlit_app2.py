import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.express as px
from datetime import datetime, timedelta
import tickers from tickers

# Pagina-instellingen
st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Monitor met Backtesting")
# Sidebar
with st.sidebar:
    st.header("🔍 Kies een Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)
    
    st.markdown("---")
    st.header("📊 Data Instellingen")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y", "2y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", min_value=5, max_value=60, value=20, step=1)
    
    st.markdown("---")
    st.header("⚙️ Trading Parameters")
    zscore_entry_threshold = st.slider("Z-score entry threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    
    st.markdown("---")
    st.header("🎯 Backtesting Instellingen")
    initial_capital = st.number_input("Startkapitaal (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
    transaction_cost = st.slider("Transactiekosten (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    max_position_size = st.slider("Max positie grootte (% van kapitaal)", min_value=10, max_value=100, value=50, step=10)
    
    # Backtesting periode
    st.subheader("📅 Backtesting Periode")
    backtest_periode = st.selectbox("Backtest periode", ["3mo", "6mo", "1y", "2y"], index=1)
    
    # Risk management
    st.subheader("🛡️ Risk Management")
    stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    take_profit_pct = st.slider("Take Profit (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

# Converteer namen naar ticker symbolen
coin1 = tickers[name1]
coin2 = tickers[name2]

# Data ophalen met caching
@st.cache_data
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker}: {e}")
        return pd.DataFrame()

# Functie voor data preprocessing
def preprocess_data(data1, data2):
    # Data verwerken - sluitprijzen gebruiken
    if isinstance(data1.columns, pd.MultiIndex):
        df1 = data1['Close'].iloc[:, 0].dropna()
        df2 = data2['Close'].iloc[:, 0].dropna()
    else:
        df1 = data1['Close'].dropna()
        df2 = data2['Close'].dropna()
    
    # Zorg ervoor dat we Series hebben
    if not isinstance(df1, pd.Series):
        df1 = pd.Series(df1)
    if not isinstance(df2, pd.Series):
        df2 = pd.Series(df2)
    
    # Combineer data op basis van datum
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    
    # Maak DataFrame
    df = pd.DataFrame({
        'price1': df1_aligned,
        'price2': df2_aligned
    }).dropna()
    
    return df

# Backtesting functie
def run_backtest(df, entry_threshold, exit_threshold, initial_capital, transaction_cost, max_position_size, stop_loss_pct, take_profit_pct):
    # Bereken spread en z-score
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.intercept_
    beta = model.coef_[0]
    
    df['spread'] = df['price2'] - (alpha + beta * df['price1'])
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    # Initialiseer backtesting variabelen
    portfolio_value = initial_capital
    position = 0  # 0 = geen positie, 1 = long spread, -1 = short spread
    position_size = 0
    entry_price_spread = 0
    entry_date = None
    
    # Tracking variabelen
    trades = []
    portfolio_values = [initial_capital]
    positions = [0]
    
    # Bereken maximum positie grootte
    max_position_value = (max_position_size / 100) * initial_capital
    
    for i in range(1, len(df)):
        current_zscore = df['zscore'].iloc[i]
        current_spread = df['spread'].iloc[i]
        current_date = df.index[i]
        
        # Check voor nieuwe posities
        if position == 0:  # Geen huidige positie
            if current_zscore < -entry_threshold:  # Long spread signaal
                position = 1
                entry_price_spread = current_spread
                entry_date = current_date
                position_size = min(max_position_value, portfolio_value * 0.95)  # 95% om kosten te dekken
                
            elif current_zscore > entry_threshold:  # Short spread signaal
                position = -1
                entry_price_spread = current_spread
                entry_date = current_date
                position_size = min(max_position_value, portfolio_value * 0.95)
        
        # Check voor exit condities
        elif position != 0:
            exit_trade = False
            exit_reason = ""
            
            # Normal exit op z-score
            if abs(current_zscore) < exit_threshold:
                exit_trade = True
                exit_reason = "Z-score exit"
            
            # Stop loss check
            if position == 1:  # Long spread
                pnl_pct = ((current_spread - entry_price_spread) / abs(entry_price_spread)) * 100
                if pnl_pct < -stop_loss_pct:
                    exit_trade = True
                    exit_reason = "Stop loss"
                elif pnl_pct > take_profit_pct:
                    exit_trade = True
                    exit_reason = "Take profit"
            
            elif position == -1:  # Short spread
                pnl_pct = ((entry_price_spread - current_spread) / abs(entry_price_spread)) * 100
                if pnl_pct < -stop_loss_pct:
                    exit_trade = True
                    exit_reason = "Stop loss"
                elif pnl_pct > take_profit_pct:
                    exit_trade = True
                    exit_reason = "Take profit"
            
            # Execute exit
            if exit_trade:
                # Bereken P&L
                if position == 1:  # Long spread
                    pnl = ((current_spread - entry_price_spread) / abs(entry_price_spread)) * position_size
                else:  # Short spread
                    pnl = ((entry_price_spread - current_spread) / abs(entry_price_spread)) * position_size
                
                # Transactiekosten aftrekken
                transaction_costs = position_size * (transaction_cost / 100) * 2  # Entry en exit
                pnl -= transaction_costs
                
                # Update portfolio
                portfolio_value += pnl
                
                # Log trade
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Position': 'Long Spread' if position == 1 else 'Short Spread',
                    'Entry Z-score': df['zscore'].loc[entry_date],
                    'Exit Z-score': current_zscore,
                    'Entry Spread': entry_price_spread,
                    'Exit Spread': current_spread,
                    'Position Size': position_size,
                    'P&L': pnl,
                    'P&L %': (pnl / position_size) * 100,
                    'Exit Reason': exit_reason,
                    'Days Held': (current_date - entry_date).days
                })
                
                # Reset position
                position = 0
                position_size = 0
                entry_price_spread = 0
                entry_date = None
        
        # Track portfolio value en posities
        portfolio_values.append(portfolio_value)
        positions.append(position)
    
    # Creëer results DataFrame
    df['portfolio_value'] = portfolio_values
    df['position'] = positions
    
    return df, trades

# Load data
data1 = load_data(coin1, periode, interval)
data2 = load_data(coin2, periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins. Probeer een andere combinatie of periode.")
    st.stop()

# Preprocess data
df = preprocess_data(data1, data2)

if df.empty:
    st.error("Geen overlappende data beschikbaar voor beide coins.")
    st.stop()

# Voer backtesting uit
df_backtest, trades = run_backtest(
    df, 
    zscore_entry_threshold, 
    zscore_exit_threshold, 
    initial_capital, 
    transaction_cost, 
    max_position_size, 
    stop_loss_pct, 
    take_profit_pct
)

# === BACKTESTING RESULTATEN SECTIE ===
st.header("🔙 Backtesting Resultaten")

# Bereken key metrics
if len(trades) > 0:
    trades_df = pd.DataFrame(trades)
    
    # Portfolio metrics
    final_value = df_backtest['portfolio_value'].iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Trade metrics
    winning_trades = trades_df[trades_df['P&L'] > 0]
    losing_trades = trades_df[trades_df['P&L'] < 0]
    
    win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    avg_win = winning_trades['P&L'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['P&L'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['P&L'].sum() / losing_trades['P&L'].sum()) if len(losing_trades) > 0 and losing_trades['P&L'].sum() != 0 else float('inf')
    
    # Risk metrics
    returns = df_backtest['portfolio_value'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
    
    max_drawdown = ((df_backtest['portfolio_value'].cummax() - df_backtest['portfolio_value']) / df_backtest['portfolio_value'].cummax()).max() * 100
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totaal Rendement", f"{total_return:.2f}%")
        st.metric("Aantal Trades", len(trades_df))
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col2:
        st.metric("Eindwaarde", f"${final_value:,.0f}")
        st.metric("Gemiddelde Win", f"${avg_win:.0f}")
        st.metric("Gemiddelde Loss", f"${avg_loss:.0f}")
    
    with col3:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    with col4:
        st.metric("Volatiliteit", f"{volatility:.2f}")
        avg_holding_period = trades_df['Days Held'].mean()
        st.metric("Gem. Holding Period", f"{avg_holding_period:.1f} dagen")
        total_transaction_costs = len(trades_df) * initial_capital * (transaction_cost / 100) * 2
        st.metric("Transactiekosten", f"${total_transaction_costs:.0f}")
    
    # Portfolio value grafiek
    fig_portfolio = go.Figure()
    
    fig_portfolio.add_trace(go.Scatter(
        x=df_backtest.index,
        y=df_backtest['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='green', width=2)
    ))
    
    # Buy and hold benchmark
    buy_hold_value = initial_capital * (df_backtest['price1'].iloc[-1] / df_backtest['price1'].iloc[0])
    fig_portfolio.add_hline(y=buy_hold_value, line_dash="dash", line_color="blue", 
                           annotation_text=f"Buy & Hold {name1}: ${buy_hold_value:,.0f}")
    
    # Voeg trade markers toe
    for trade in trades:
        # Entry marker
        fig_portfolio.add_trace(go.Scatter(
            x=[trade['Entry Date']],
            y=[df_backtest.loc[trade['Entry Date'], 'portfolio_value']],
            mode='markers',
            marker=dict(
                color='green' if trade['Position'] == 'Long Spread' else 'red',
                size=10,
                symbol='triangle-up' if trade['Position'] == 'Long Spread' else 'triangle-down'
            ),
            name=f"Entry {trade['Position']}",
            showlegend=False
        ))
        
        # Exit marker
        fig_portfolio.add_trace(go.Scatter(
            x=[trade['Exit Date']],
            y=[df_backtest.loc[trade['Exit Date'], 'portfolio_value']],
            mode='markers',
            marker=dict(
                color='blue',
                size=8,
                symbol='x'
            ),
            name="Exit",
            showlegend=False
        ))
    
    fig_portfolio.update_layout(
        title="Portfolio Value Over Time met Trade Markers",
        xaxis_title="Datum",
        yaxis_title="Portfolio Value (USD)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Trades tabel
    st.subheader("📋 Trade History")
    
    # Maak trades tabel meer leesbaar
    trades_display = trades_df.copy()
    trades_display['Entry Date'] = trades_display['Entry Date'].dt.strftime('%Y-%m-%d')
    trades_display['Exit Date'] = trades_display['Exit Date'].dt.strftime('%Y-%m-%d')
    trades_display['P&L'] = trades_display['P&L'].apply(lambda x: f"${x:,.0f}")
    trades_display['Position Size'] = trades_display['Position Size'].apply(lambda x: f"${x:,.0f}")
    trades_display['P&L %'] = trades_display['P&L %'].apply(lambda x: f"{x:.2f}%")
    trades_display['Entry Z-score'] = trades_display['Entry Z-score'].apply(lambda x: f"{x:.2f}")
    trades_display['Exit Z-score'] = trades_display['Exit Z-score'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(trades_display, use_container_width=True)
    
    # P&L distributie
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pnl = px.histogram(
            trades_df, 
            x='P&L %', 
            nbins=20,
            title="P&L Distributie (%)",
            labels={'P&L %': 'P&L Percentage', 'count': 'Aantal Trades'}
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    
    with col2:
        fig_holding = px.histogram(
            trades_df, 
            x='Days Held', 
            nbins=15,
            title="Holding Period Distributie",
            labels={'Days Held': 'Dagen Gehouden', 'count': 'Aantal Trades'}
        )
        st.plotly_chart(fig_holding, use_container_width=True)

else:
    st.warning("Geen trades uitgevoerd in de backtesting periode. Probeer andere parameters.")

# === ORIGINELE ANALYSE SECTIE ===
st.header("📊 Huidige Analyse")

# Bereken spread en z-score voor huidige analyse
X = df['price1'].values.reshape(-1, 1)
y = df['price2'].values

model = LinearRegression()
model.fit(X, y)

alpha = model.intercept_
beta = model.coef_[0]
r_squared = model.score(X, y)

# Spread berekenen
df['spread'] = df['price2'] - (alpha + beta * df['price1'])
spread_mean = df['spread'].mean()
spread_std = df['spread'].std()
df['zscore'] = (df['spread'] - spread_mean) / spread_std

# Rolling correlatie
df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
pearson_corr = df['price1'].corr(df['price2'])

# Trade signalen
df['long_entry'] = df['zscore'] < -zscore_entry_threshold
df['short_entry'] = df['zscore'] > zscore_entry_threshold
df['exit'] = df['zscore'].abs() < zscore_exit_threshold

# Huidige positie
if df['long_entry'].iloc[-1]:
    current_position = f"Long Spread (koop {name2}, verkoop {name1})"
elif df['short_entry'].iloc[-1]:
    current_position = f"Short Spread (verkoop {name2}, koop {name1})"
elif df['exit'].iloc[-1]:
    current_position = "Exit positie (geen trade)"
else:
    current_position = "Geen duidelijk signaal"

# Huidige signaal
st.subheader("🚦 Huidige Trade Signaal")
st.write(f"**Z-score laatste waarde:** {df['zscore'].iloc[-1]:.2f}")
st.write(f"**Signaal:** {current_position}")

# Spread grafiek met entry/exit levels
entry_long_level = -zscore_entry_threshold * spread_std + spread_mean
entry_short_level = zscore_entry_threshold * spread_std + spread_mean
exit_level_pos = zscore_exit_threshold * spread_std + spread_mean
exit_level_neg = -zscore_exit_threshold * spread_std + spread_mean

fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=df.index, y=df['spread'], mode='lines', name='Spread'))

fig_signal.add_hline(y=entry_long_level, line=dict(color='green', dash='dash'), 
                    annotation_text='Long Entry', annotation_position='bottom left')
fig_signal.add_hline(y=entry_short_level, line=dict(color='red', dash='dash'), 
                    annotation_text='Short Entry', annotation_position='top left')
fig_signal.add_hline(y=exit_level_pos, line=dict(color='blue', dash='dot'), 
                    annotation_text='Exit', annotation_position='top right')
fig_signal.add_hline(y=exit_level_neg, line=dict(color='blue', dash='dot'), 
                    annotation_text='Exit', annotation_position='bottom right')

fig_signal.update_layout(title="Spread met Entry en Exit Niveaus", yaxis_title="Spread", xaxis_title="Datum")
st.plotly_chart(fig_signal, use_container_width=True)

# Prijs en Z-score grafieken
col1, col2 = st.columns(2)

with col1:
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(x=df.index, y=df['price1'], name=name1, line=dict(color='blue')))
    fig_prices.add_trace(go.Scatter(x=df.index, y=df['price2'], name=name2, line=dict(color='red'), yaxis='y2'))
    
    fig_prices.update_layout(
        title="Prijsverloop",
        xaxis_title="Datum",
        yaxis_title=f"{name1} Prijs (USD)",
        yaxis2=dict(title=f"{name2} Prijs (USD)", overlaying='y', side='right')
    )
    st.plotly_chart(fig_prices, use_container_width=True)

with col2:
    fig_zscore = go.Figure()
    fig_zscore.add_trace(go.Scatter(x=df.index, y=df['zscore'], name='Z-score', line=dict(color='purple')))
    fig_zscore.add_hline(y=zscore_entry_threshold, line=dict(color='red', dash='dash'), annotation_text='Entry Threshold')
    fig_zscore.add_hline(y=-zscore_entry_threshold, line=dict(color='green', dash='dash'), annotation_text='Entry Threshold')
    fig_zscore.add_hline(y=zscore_exit_threshold, line=dict(color='blue', dash='dot'), annotation_text='Exit Threshold')
    fig_zscore.add_hline(y=-zscore_exit_threshold, line=dict(color='blue', dash='dot'), annotation_text='Exit Threshold')
    fig_zscore.update_layout(title="Z-score", yaxis_title="Z-score", xaxis_title="Datum")
    st.plotly_chart(fig_zscore, use_container_width=True)

# Correlatie statistieken
st.subheader("📈 Correlatie Statistieken")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pearson Correlatie", f"{pearson_corr:.4f}")
    st.metric("Beta (β)", f"{beta:.4f}")
    st.metric("R-squared", f"{r_squared:.4f}")

with col2:
    current_rolling_corr = df['rolling_corr'].iloc[-1]
    avg_rolling_corr = df['rolling_corr'].mean()
    st.metric("Rolling Correlatie", f"{current_rolling_corr:.4f}")
    st.metric("Gem. Rolling Correlatie", f"{avg_rolling_corr:.4f}")
    st.metric("Alpha (α)", f"{alpha:.6f}")

with col3:
    df['returns1'] = df['price1'].pct_change()
    df['returns2'] = df['price2'].pct_change()
    returns_clean = df[['returns1', 'returns2']].dropna()
    returns_corr = returns_clean['returns1'].corr(returns_clean['returns2'])
    volatility_ratio = returns_clean['returns2'].std() / returns_clean['returns1'].std()
    st.metric("Returns Correlatie", f"{returns_corr:.4f}")
    st.metric("Volatiliteit Ratio", f"{volatility_ratio:.4f}")
    st.metric("Spread Volatiliteit", f"{spread_std:.4f}")

# Export functionaliteit
if st.button("Exporteer analyse naar CSV"):
    # Combineer alle relevante data
    export_df = df.copy()
    if len(trades) > 0:
        export_df['backtest_portfolio_value'] = df_backtest['portfolio_value']
    
    csv = export_df.to_csv(index=True)
    st.download_button(
        label="Download CSV", 
        data=csv, 
        file_name=f"pairs_trading_analysis_{name1}_{name2}_{datetime.now().strftime('%Y%m%d')}.csv", 
        mime='text/csv'
    )

# Trade history export
if len(trades) > 0:
    if st.button("Exporteer trade history naar CSV"):
        trades_csv = trades_df.to_csv(index=False)
        st.download_button(
            label="Download Trade History CSV",
            data=trades_csv,
            file_name=f"trade_history_{name1}_{name2}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
