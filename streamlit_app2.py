import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from constants.tickers import tickers

# Pagina-instellingen
st.set_page_config(layout="wide")
st.title("📈 Pairs Trading Monitor")

# Sidebar instellingen
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

# Data ophalen en verwerken
@st.cache_data
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        return df
    except Exception as e:
        st.error(f"Fout bij ophalen data voor {ticker}: {e}")
        return pd.DataFrame()

def preprocess_data(data1, data2):
    if isinstance(data1.columns, pd.MultiIndex):
        df1 = data1['Close'].iloc[:, 0].dropna()
        df2 = data2['Close'].iloc[:, 0].dropna()
    else:
        df1 = data1['Close'].dropna()
        df2 = data2['Close'].dropna()
    
    if not isinstance(df1, pd.Series):
        df1 = pd.Series(df1)
    if not isinstance(df2, pd.Series):
        df2 = pd.Series(df2)
    
    df1_aligned, df2_aligned = df1.align(df2, join='inner')
    
    df = pd.DataFrame({
        'price1': df1_aligned,
        'price2': df2_aligned
    }).dropna()
    
    return df

# Laad data
data1 = load_data(tickers[name1], periode, interval)
data2 = load_data(tickers[name2], periode, interval)

if data1.empty or data2.empty:
    st.error("Geen data beschikbaar voor één of beide coins. Probeer een andere combinatie of periode.")
    st.stop()

df = preprocess_data(data1, data2)

if df.empty:
    st.error("Geen overlappende data beschikbaar voor beide coins.")
    st.stop()

# === ANALYSE SECTIE ===
with st.expander("📊 Statistische Analyse", expanded=True):
    st.header("📊 Statistische Analyse")
    
    # Bereken statistieken
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
    
    # Rolling correlatie
    df['rolling_corr'] = df['price1'].rolling(window=corr_window).corr(df['price2'])
    pearson_corr = df['price1'].corr(df['price2'])
    
    # Ratiografiek
    df['ratio'] = df['price1'] / df['price2']
    
    # Returns voor scatterplot
    df['returns1'] = df['price1'].pct_change()
    df['returns2'] = df['price2'].pct_change()
    
    # Layout met tabs
    tab1, tab2 = st.tabs(["Prijsanalyse", "Correlatieanalyse"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prijsgrafiek met 2 y-assen
            fig_prices = go.Figure()
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price1'], name=name1, line=dict(color='blue')))
            fig_prices.add_trace(go.Scatter(
                x=df.index, y=df['price2'], name=name2, line=dict(color='red'), yaxis='y2'))
            
            fig_prices.update_layout(
                title="Prijsverloop",
                xaxis_title="Datum",
                yaxis_title=f"{name1} Prijs (USD)",
                yaxis2=dict(
                    title=f"{name2} Prijs (USD)",
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
            
            # Spread grafiek
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=df.index, y=df['spread'], name='Spread',
                line=dict(color='green'), fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            fig_spread.update_layout(
                title="Spread tussen coins",
                xaxis_title="Datum",
                yaxis_title="Spread",
                height=400
            )
            st.plotly_chart(fig_spread, use_container_width=True)
        
        with col2:
            # Ratio grafiek
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=df.index, y=df['ratio'], name=f"{name1}/{name2} Ratio",
                line=dict(color='purple'), fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            fig_ratio.update_layout(
                title=f"{name1}/{name2} Prijs Ratio",
                xaxis_title="Datum",
                yaxis_title="Ratio",
                height=400
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Z-score grafiek
            # Z-score grafiek met verduidelijkte trading signals
            fig_zscore = go.Figure()
            
            # Z-score lijn zelf (dikke grijze lijn)
            fig_zscore.add_trace(go.Scatter(
                x=df.index,
                y=df['zscore'],
                name='Z-score',
                line=dict(color='#888888', width=3)  # Grijze hoofdline
            ))
            
            # LONG ENTRY (volle groene lijn)
            fig_zscore.add_hline(
                y=-zscore_entry_threshold,
                line=dict(color='#2ECC71', width=2),
                annotation_text="LONG ENTRY (koop spread)",
                annotation_font=dict(color="white", size=12),
                annotation_position="bottom right"
            )
            
            # LONG EXIT (gestippelde groene lijn)
            fig_zscore.add_hline(
                y=-zscore_exit_threshold,
                line=dict(color='#2ECC71', width=2, dash='dot'),
                annotation_text="LONG EXIT",
                annotation_font=dict(color="white", size=12),
                annotation_position="bottom right"
            )
            
            # SHORT ENTRY (volle rode lijn)
            fig_zscore.add_hline(
                y=zscore_entry_threshold,
                line=dict(color='#E74C3C', width=2),
                annotation_text="SHORT ENTRY (verkoop spread)",
                annotation_font=dict(color="white", size=12),
                annotation_position="top right"
            )
            
            # SHORT EXIT (gestippelde rode lijn)
            fig_zscore.add_hline(
                y=zscore_exit_threshold,
                line=dict(color='#E74C3C', width=2, dash='dot'),
                annotation_text="SHORT EXIT",
                annotation_font=dict(color="white", size=12),
                annotation_position="top right"
            )
            
            # Nul lijn (dunne witte lijn)
            fig_zscore.add_hline(
                y=0,
                line=dict(color='white', width=1)
            )
            
            # Legenda toevoegen via dummy traces
            fig_zscore.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#2ECC71', width=2),
                name='Long signals'
            ))
            fig_zscore.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#2ECC71', width=2, dash='dot'),
                name='Long exit'
            ))
            fig_zscore.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#E74C3C', width=2),
                name='Short signals'
            ))
            fig_zscore.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#E74C3C', width=2, dash='dot'),
                name='Short exit'
            ))
            
            fig_zscore.update_layout(
                title="<b>Z-score Trading Signals</b>",
                xaxis_title="Datum",
                yaxis_title="Z-score waarde",
                height=500,
                plot_bgcolor='#2C3E50',  # Donkere achtergrond
                paper_bgcolor='#2C3E50',
                font=dict(color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        x=0.02, y=-zscore_entry_threshold,
                        xref="paper", yref="y",
                        text="<b>LONG ZONE</b>",
                        showarrow=False,
                        font=dict(color="#2ECC71", size=14)
                    ),
                    dict(
                        x=0.02, y=zscore_entry_threshold,
                        xref="paper", yref="y",
                        text="<b>SHORT ZONE</b>",
                        showarrow=False,
                        font=dict(color="#E74C3C", size=14)
                    )
                ]
            )
            
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlatiegrafiek
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df.index, y=df['rolling_corr'], name='Rolling Correlatie',
                line=dict(color='blue')
            ))
            fig_corr.update_layout(
                title=f"Rolling Correlatie ({corr_window}d)",
                xaxis_title="Datum",
                yaxis_title="Correlatie",
                yaxis_range=[-1, 1],
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Statistieken
            st.subheader("Statistieken")
            st.metric("Pearson Correlatie", f"{pearson_corr:.4f}")
            st.metric("Rolling Correlatie", f"{df['rolling_corr'].iloc[-1]:.4f}")
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

# === BACKTESTING SECTIE ===
def run_backtest(df, entry_threshold, exit_threshold, initial_capital, transaction_cost, max_position_size, stop_loss_pct, take_profit_pct):
    """UW ORIGINELE BACKTEST FUNCTIE"""
    X = df['price1'].values.reshape(-1, 1)
    y = df['price2'].values
    model = LinearRegression()
    model.fit(X, y)
    
    df['spread'] = df['price2'] - (model.intercept_ + model.coef_[0] * df['price1'])
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    cash = initial_capital
    position = 0
    coin1_shares = 0
    coin2_shares = 0
    entry_price1 = 0
    entry_price2 = 0
    entry_date = None
    position_value = 0
    max_position_value = (max_position_size / 100) * initial_capital
    
    trades = []
    portfolio_values = []
    positions = []

    for i in range(len(df)):
        current_zscore = df['zscore'].iloc[i]
        current_price1 = df['price1'].iloc[i]
        current_price2 = df['price2'].iloc[i]
        current_date = df.index[i]
        
        position_market_value = coin1_shares * current_price1 + coin2_shares * current_price2
        portfolio_value = cash + position_market_value
        
        # Entry logic
        if position == 0 and i > 0:
            if current_zscore < -entry_threshold:  # Long spread
                position = 1
                position_value = min(max_position_value, portfolio_value * 0.95)
                coin2_shares = (position_value / 2) / current_price2
                coin1_shares = -(position_value / 2) / current_price1
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                cash -= position_value * (transaction_cost / 100)
                
            elif current_zscore > entry_threshold:  # Short spread
                position = -1
                position_value = min(max_position_value, portfolio_value * 0.95)
                coin1_shares = (position_value / 2) / current_price1
                coin2_shares = -(position_value / 2) / current_price2
                entry_price1 = current_price1
                entry_price2 = current_price2
                entry_date = current_date
                cash -= position_value * (transaction_cost / 100)
        
        # Exit logic
        elif position != 0:
            exit_trade = False
            exit_reason = ""
            
            if abs(current_zscore) < exit_threshold:
                exit_trade = True
                exit_reason = "Z-score exit"
            
            pnl_dollar = (coin1_shares * (current_price1 - entry_price1) + 
                         coin2_shares * (current_price2 - entry_price2))
            pnl_pct = (pnl_dollar / position_value) * 100
            
            if pnl_pct < -stop_loss_pct:
                exit_trade = True
                exit_reason = "Stop loss"
            elif pnl_pct > take_profit_pct:
                exit_trade = True
                exit_reason = "Take profit"
            
            if exit_trade:
                final_pnl = pnl_dollar - (abs(coin1_shares * current_price1) + abs(coin2_shares * current_price2)) * (transaction_cost / 100)
                cash += (coin1_shares * current_price1 + coin2_shares * current_price2 + final_pnl)
                
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Position': 'Long Spread' if position == 1 else 'Short Spread',
                    'Entry Z-score': df['zscore'].loc[entry_date],
                    'Exit Z-score': current_zscore,
                    'Entry Price 1': entry_price1,
                    'Entry Price 2': entry_price2,
                    'Exit Price 1': current_price1,
                    'Exit Price 2': current_price2,
                    'Coin1 Shares': coin1_shares,
                    'Coin2 Shares': coin2_shares,
                    'Position Size': position_value,
                    'P&L': final_pnl,
                    'P&L %': (final_pnl / position_value) * 100,
                    'Exit Reason': exit_reason,
                    'Days Held': (current_date - entry_date).days
                })
                
                position = 0
                coin1_shares = 0
                coin2_shares = 0
                entry_price1 = 0
                entry_price2 = 0
                entry_date = None
                position_value = 0
        
        portfolio_values.append(portfolio_value)
        positions.append(position)
    
    df_result = df.copy()
    df_result['portfolio_value'] = portfolio_values
    df_result['position'] = positions
    return df_result, trades

with st.expander("🔙 Backtesting", expanded=False):
    st.header("🔙 Backtesting")
    
    with st.sidebar:
        st.markdown("---")
        st.header("🎯 Backtesting Instellingen")
        initial_capital = st.number_input("Startkapitaal (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
        transaction_cost = st.slider("Transactiekosten (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        max_position_size = st.slider("Max positie grootte (% van kapitaal)", min_value=10, max_value=100, value=50, step=10)
        
        st.subheader("🛡️ Risk Management")
        stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        take_profit_pct = st.slider("Take Profit (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    
    if st.button("Run Backtest"):
        df_backtest, trades = run_backtest(
            df, zscore_entry_threshold, zscore_exit_threshold,
            initial_capital, transaction_cost, max_position_size,
            stop_loss_pct, take_profit_pct
        )
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            final_value = df_backtest['portfolio_value'].iloc[-1]
            total_return = ((final_value - initial_capital) / initial_capital) * 100
            
            st.success(f"Backtest voltooid! Totaal rendement: {total_return:.2f}%")
            
            tab1, tab2 = st.tabs(["Performance Metrics", "Trade History"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Eindwaarde", f"${final_value:,.2f}")
                    st.metric("Aantal Trades", len(trades_df))
                    winning_trades = trades_df[trades_df['P&L'] > 0]
                    st.metric("Win Rate", f"{len(winning_trades)/len(trades_df)*100:.1f}%")
                
                with col2:
                    st.metric("Gem. Win Trade", f"${winning_trades['P&L'].mean():.2f}")
                    losing_trades = trades_df[trades_df['P&L'] < 0]
                    st.metric("Gem. Loss Trade", f"${losing_trades['P&L'].mean():.2f}")
                    st.metric("Profit Factor", f"{abs(winning_trades['P&L'].sum()/losing_trades['P&L'].sum()):.2f}")
            
            with tab2:
                st.dataframe(trades_df.sort_values('Exit Date', ascending=False))
            
            # Portfolio value grafiek
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_backtest.index,
                y=df_backtest['portfolio_value'],
                name='Portfolio Value'
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geen trades uitgevoerd tijdens backtest")

# Export functionaliteit
with st.expander("📤 Export", expanded=False):
    st.header("📤 Exporteer Data")
    if 'df_backtest' in locals():
        if st.button("Exporteer backtest resultaten"):
            csv = df_backtest.to_csv(index=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"backtest_results_{name1}_{name2}.csv",
                mime='text/csv'
            )
