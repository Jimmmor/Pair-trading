import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.express as px
from datetime import datetime, timedelta
from constants.tickers import tickers

# Pagina-instellingen
st.set_page_config(layout="wide")
st.title("Pairs Trading Monitor met Backtesting")

# Sidebar
with st.sidebar:
    st.header("Coin Pair")
    name1 = st.selectbox("Coin 1", list(tickers.keys()), index=0)
    remaining = [k for k in tickers.keys() if k != name1]
    name2 = st.selectbox("Coin 2", remaining, index=0)
    
    st.markdown("---")
    st.header("Data Instellingen")
    periode = st.selectbox("Periode", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d"] if periode in ["6mo", "1y", "2y"] else ["1d", "1h", "30m"], index=0)
    corr_window = st.slider("Rolling correlatie window (dagen)", min_value=5, max_value=60, value=20, step=1)
    
    st.markdown("---")
    st.header("Trading Parameters")
    zscore_entry_threshold = st.slider("Z-score entry threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    zscore_exit_threshold = st.slider("Z-score exit threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    
    st.markdown("---")
    st.header("Backtesting Instellingen")
    initial_capital = st.number_input("Startkapitaal (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
    transaction_cost = st.slider("Transactiekosten (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    max_position_size = st.slider("Max positie grootte (% van kapitaal)", min_value=10, max_value=100, value=50, step=10)
    
    # Backtesting periode
    st.subheader("Backtesting Periode")
    backtest_periode = st.selectbox("Backtest periode", ["3mo", "6mo", "1y", "2y"], index=1)
    
    # Risk management
    st.subheader("Risk Management")
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

def validate_data_for_regression(X, y):
    """
    Valideer data voor linear regression
    """
    # Check for empty data
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty data arrays")
    
    # Check for minimum data points
    if len(X) < 10:
        raise ValueError("Insufficient data points for regression (minimum 10 required)")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Data contains NaN values")
    
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Data contains infinite values")
    
    # Check for zero variance
    if np.std(X.flatten()) == 0:
        raise ValueError("X data has zero variance")
    
    if np.std(y) == 0:
        raise ValueError("Y data has zero variance")
    
    # Check for identical values
    if len(np.unique(X.flatten())) < 2:
        raise ValueError("X data contains insufficient unique values")
    
    return True

def run_backtest(df, entry_threshold, exit_threshold, initial_capital, transaction_cost, max_position_size, stop_loss_pct, take_profit_pct):
    """
    Verbeterde backtest functie met correcte P&L berekening en data validatie
    """
    try:
        # Clean data en validatie
        df = df.dropna(subset=['price1', 'price2']).reset_index(drop=True)
        
        if len(df) < 10:
            raise ValueError(f"Insufficient data points: {len(df)} (minimum 10 required)")
        
        # Bereken spread en z-score
        X = df['price1'].values.reshape(-1, 1)
        y = df['price2'].values
        
        # Valideer data voor regression
        validate_data_for_regression(X, y)
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        alpha = model.intercept_
        beta = model.coef_[0]
        
        # Bereken spread
        df['spread'] = df['price2'] - (alpha + beta * df['price1'])
        
        # Check of spread valide is
        if df['spread'].std() == 0:
            raise ValueError("Spread has zero variance - cannot calculate z-score")
        
        spread_mean = df['spread'].mean()
        spread_std = df['spread'].std()
        df['zscore'] = (df['spread'] - spread_mean) / spread_std
        
        # Check voor NaN values in z-score
        if df['zscore'].isna().any():
            raise ValueError("Z-score calculation resulted in NaN values")
        
    except Exception as e:
        st.error(f"Error in data preparation for backtesting: {str(e)}")
        # Return empty results
        empty_df = df.copy() if 'df' in locals() else pd.DataFrame()
        if not empty_df.empty:
            empty_df['portfolio_value'] = [initial_capital] * len(empty_df)
            empty_df['position'] = [0] * len(empty_df)
        return empty_df, []
    
    # Backtest variabelen
    cash = initial_capital
    coin1_position = 0  # aantal coins
    coin2_position = 0  # aantal coins
    position_value = 0  # waarde van huidige positie
    trade_active = False
    entry_info = {}
    
    # Tracking
    trades = []
    portfolio_values = []
    positions = []
    
    # Bereken max position size in dollar
    max_position_value = (max_position_size / 100) * initial_capital
    
    for i in range(len(df)):
        current_zscore = df['zscore'].iloc[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_date = df.index[i]
        
        # Bereken huidige portfolio waarde
        position_market_value = (coin1_position * current_price1) + (coin2_position * current_price2)
        total_portfolio_value = cash + position_market_value
        
        # Check voor nieuwe posities (alleen als geen actieve trade)
        if not trade_active:
            position_size_dollar = min(max_position_value, cash * 0.95)  # 5% buffer voor kosten
            
            if current_zscore < -entry_threshold and position_size_dollar > 100:  # Long spread
                # Long spread = verwacht dat spread omhoog gaat
                # Short coin1, Long coin2
                
                # Bereken aantal coins per positie (50/50 split van position size)
                coin1_investment = position_size_dollar / 2
                coin2_investment = position_size_dollar / 2
                
                coin1_units = coin1_investment / current_price1
                coin2_units = coin2_investment / current_price2
                
                # Execute trade (short coin1, long coin2)
                coin1_position = -coin1_units  # Short position
                coin2_position = coin2_units   # Long position
                
                # Update cash (ontvang geld van short, betaal voor long, min transactiekosten)
                transaction_costs = position_size_dollar * (transaction_cost / 100)
                cash += coin1_investment - coin2_investment - transaction_costs
                
                trade_active = True
                entry_info = {
                    'date': current_date,
                    'type': 'Long Spread',
                    'zscore': current_zscore,
                    'spread': df['spread'].iloc[i],
                    'coin1_price': current_price1,
                    'coin2_price': current_price2,
                    'coin1_units': coin1_units,
                    'coin2_units': coin2_units,
                    'position_size': position_size_dollar,
                    'cash_after_entry': cash
                }
                
            elif current_zscore > entry_threshold and position_size_dollar > 100:  # Short spread
                # Short spread = verwacht dat spread omlaag gaat  
                # Long coin1, Short coin2
                
                coin1_investment = position_size_dollar / 2
                coin2_investment = position_size_dollar / 2
                
                coin1_units = coin1_investment / current_price1
                coin2_units = coin2_investment / current_price2
                
                # Execute trade (long coin1, short coin2)
                coin1_position = coin1_units   # Long position
                coin2_position = -coin2_units  # Short position
                
                # Update cash
                transaction_costs = position_size_dollar * (transaction_cost / 100)
                cash += coin2_investment - coin1_investment - transaction_costs
                
                trade_active = True
                entry_info = {
                    'date': current_date,
                    'type': 'Short Spread',
                    'zscore': current_zscore,
                    'spread': df['spread'].iloc[i],
                    'coin1_price': current_price1,
                    'coin2_price': current_price2,
                    'coin1_units': coin1_units,
                    'coin2_units': coin2_units,
                    'position_size': position_size_dollar,
                    'cash_after_entry': cash
                }
        
        # Check voor exit condities
        elif trade_active:
            exit_trade = False
            exit_reason = ""
            
            # Z-score exit
            if abs(current_zscore) < exit_threshold:
                exit_trade = True
                exit_reason = "Z-score exit"
            
            # Stop loss en take profit (gebaseerd op werkelijke P&L)
            current_position_value = (coin1_position * current_price1) + (coin2_position * current_price2)
            unrealized_pnl = current_position_value - (entry_info['cash_after_entry'] - cash)
            unrealized_pnl_pct = (unrealized_pnl / entry_info['position_size']) * 100
            
            if unrealized_pnl_pct < -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            elif unrealized_pnl_pct > take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            # Execute exit
            if exit_trade:
                # Close positions
                exit_value_coin1 = coin1_position * current_price1
                exit_value_coin2 = coin2_position * current_price2
                
                # Update cash (liquideer alle posities)
                cash += exit_value_coin1 + exit_value_coin2
                
                # Transactiekosten voor exit
                transaction_costs = entry_info['position_size'] * (transaction_cost / 100)
                cash -= transaction_costs
                
                # Bereken werkelijke P&L
                total_pnl = cash + (coin1_position * current_price1) + (coin2_position * current_price2) - entry_info['cash_after_entry']
                pnl_percentage = (total_pnl / entry_info['position_size']) * 100
                
                # Log trade
                trades.append({
                    'Entry Date': entry_info['date'],
                    'Exit Date': current_date,
                    'Position': entry_info['type'],
                    'Entry Z-score': entry_info['zscore'],
                    'Exit Z-score': current_zscore,
                    'Entry Spread': entry_info['spread'],
                    'Exit Spread': df['spread'].iloc[i],
                    'Position Size': entry_info['position_size'],
                    'P&L': total_pnl,
                    'P&L %': pnl_percentage,
                    'Exit Reason': exit_reason,
                    'Days Held': (current_date - entry_info['date']).days,
                    'Entry Coin1 Price': entry_info['coin1_price'],
                    'Exit Coin1 Price': current_price1,
                    'Entry Coin2 Price': entry_info['coin2_price'],
                    'Exit Coin2 Price': current_price2,
                    'Coin1 Units': abs(entry_info['coin1_units']),
                    'Coin2 Units': abs(entry_info['coin2_units'])
                })
                
                # Reset posities
                coin1_position = 0
                coin2_position = 0
                trade_active = False
                entry_info = {}
        
        # Track portfolio waarde en posities  
        final_portfolio_value = cash + (coin1_position * current_price1) + (coin2_position * current_price2)
        portfolio_values.append(final_portfolio_value)
        positions.append(1 if trade_active and entry_info.get('type') == 'Long Spread' 
                        else -1 if trade_active and entry_info.get('type') == 'Short Spread' 
                        else 0)
    
    # Update DataFrame
    df['portfolio_value'] = portfolio_values
    df['position'] = positions
    
    return df, trades

def calculate_backtest_metrics(trades_df, df_backtest, initial_capital):
    """
    Bereken alle backtest metrics correct
    """
    if len(trades_df) == 0:
        return None
        
    # Portfolio metrics
    final_value = df_backtest['portfolio_value'].iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Trade metrics
    winning_trades = trades_df[trades_df['P&L'] > 0]
    losing_trades = trades_df[trades_df['P&L'] <= 0]
    
    total_trades = len(trades_df)
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = winning_trades['P&L'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['P&L'].sum()) if len(losing_trades) > 0 else 0
    
    avg_win = winning_trades['P&L'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['P&L'].mean() if len(losing_trades) > 0 else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    # Risk metrics - gebruik dagelijkse returns van portfolio waarde
    portfolio_returns = df_backtest['portfolio_value'].pct_change().dropna()
    
    if len(portfolio_returns) > 1:
        daily_volatility = portfolio_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe ratio (assumeer 0% risk-free rate)
        annualized_return = total_return / 100
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_cummax = df_backtest['portfolio_value'].cummax()
        drawdowns = (portfolio_cummax - df_backtest['portfolio_value']) / portfolio_cummax
        max_drawdown = drawdowns.max() * 100
        
    else:
        annualized_volatility = 0
        sharpe_ratio = 0
        max_drawdown = 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'annualized_volatility': annualized_volatility,
        'total_profit': total_profit,
        'total_loss': total_loss
    }

# Data ophalen
data1 = load_data(coin1, periode, interval)
data2 = load_data(coin2, periode, interval)

# Check if data was successfully loaded
if data1.empty or data2.empty:
    st.error("Kon data niet ophalen. Controleer de ticker symbolen en internetverbinding.")
    st.stop()

# Data preprocessen
df = preprocess_data(data1, data2)

# Check if preprocessed data is valid
if df.empty:
    st.error("Geen geldige data na preprocessing. Controleer of de gekozen coins data hebben in de geselecteerde periode.")
    st.stop()

# Backtest uitvoeren
try:
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
except Exception as e:
    st.error(f"Fout tijdens backtest: {str(e)}")
    st.stop()

# === BACKTESTING RESULTATEN SECTIE ===
st.header("🔙 Backtesting Resultaten")

if len(trades) > 0:
    trades_df = pd.DataFrame(trades)
    
    # Bereken metrics met nieuwe functie
    metrics = calculate_backtest_metrics(trades_df, df_backtest, initial_capital)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totaal Rendement", f"{metrics['total_return']:.2f}%")
        st.metric("Aantal Trades", metrics['total_trades'])
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    with col2:
        st.metric("Eindwaarde", f"${metrics['final_value']:,.0f}")
        st.metric("Gemiddelde Win", f"${metrics['avg_win']:.0f}")
        st.metric("Gemiddelde Loss", f"${metrics['avg_loss']:.0f}")
    
    with col3:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    
    with col4:
        st.metric("Volatiliteit", f"{metrics['annualized_volatility']:.2f}")
        avg_holding_period = trades_df['Days Held'].mean()
        st.metric("Gem. Holding Period", f"{avg_holding_period:.1f} dagen")
        total_transaction_costs = metrics['total_trades'] * initial_capital * (transaction_cost / 100) * 2
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
        if trade['Entry Date'] in df_backtest.index:
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
        if trade['Exit Date'] in df_backtest.index:
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
    st.subheader("Trade History")
    
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

try:
    # Bereken spread en z-score voor huidige analyse
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    
    # Valideer data
    validate_data_for_regression(X, y)
    
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
        current_position = f"Long Spread (long {name2}, short {name1})"
    elif df['short_entry'].iloc[-1]:
        current_position = f"Short Spread (short {name2}, long {name1})"
    elif df['exit'].iloc[-1]:
        current_position = "Exit positie"
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
