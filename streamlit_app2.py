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
            
            # Coin 1 op primaire y-as (blauw)
            fig_prices.add_trace(go.Scatter(
                x=df.index, 
                y=df['price1'], 
                name=name1, 
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Coin 2 op secundaire y-as (rood/oranje)
            fig_prices.add_trace(go.Scatter(
                x=df.index, 
                y=df['price2'], 
                name=name2, 
                line=dict(color='#ff7f0e', width=2),
                yaxis='y2'
            ))
            
            # Layout configuratie met styling
            fig_prices.update_layout(
                title=dict(
                    text="Prijsverloop",
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Datum",
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis=dict(
                    title=f"{name1} Prijs (USD)",
                    titlefont=dict(color='#1f77b4'),
                    tickfont=dict(color='#1f77b4'),
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis2=dict(
                    title=f"{name2} Prijs (USD)",
                    titlefont=dict(color='#ff7f0e'),
                    tickfont=dict(color='#ff7f0e'),
                    overlaying='y',
                    side='right',
                    anchor='free',
                    position=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                margin=dict(l=50, r=50, b=50, t=50, pad=4),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            # Ratiografiek met groen gebied
            fig_ratio = go.Figure()
            
            fig_ratio.add_trace(go.Scatter(
                x=df.index, 
                y=df['ratio'], 
                name=f"{name1}/{name2} Ratio",
                line=dict(color='#9467bd'),
                fill='tozeroy',
                fillcolor='rgba(148, 103, 189, 0.2)'
            ))
            
            fig_ratio.update_layout(
                title=f"{name1}/{name2} Prijs Ratio",
                xaxis_title="Datum",
                yaxis_title="Ratio",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Z-score grafiek met entry/exit thresholds
            fig_zscore = go.Figure()
            
            fig_zscore.add_trace(go.Scatter(
                x=df.index, 
                y=df['zscore'], 
                name='Z-score',
                line=dict(color='#d62728')
            ))
            
            # Entry thresholds
            fig_zscore.add_hline(
                y=zscore_entry_threshold, 
                line=dict(color='#d62728', dash='dash', width=1), 
                annotation_text='Entry Threshold',
                annotation_position="top right"
            )
            fig_zscore.add_hline(
                y=-zscore_entry_threshold, 
                line=dict(color='#d62728', dash='dash', width=1), 
                annotation_text='Entry Threshold',
                annotation_position="bottom right"
            )
            
            # Exit thresholds
            fig_zscore.add_hline(
                y=zscore_exit_threshold, 
                line=dict(color='#17becf', dash='dot', width=1), 
                annotation_text='Exit Threshold',
                annotation_position="top right"
            )
            fig_zscore.add_hline(
                y=-zscore_exit_threshold, 
                line=dict(color='#17becf', dash='dot', width=1), 
                annotation_text='Exit Threshold',
                annotation_position="bottom right"
            )
            
            fig_zscore.update_layout(
                title="Z-score van de spread",
                xaxis_title="Datum",
                yaxis_title="Z-score",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlatiegrafiek
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df.index, 
                y=df['rolling_corr'], 
                name='Rolling Correlatie',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_corr.update_layout(
                title=f"Rolling Correlatie ({corr_window} dagen window)",
                xaxis_title="Datum",
                yaxis_title="Correlatie",
                yaxis_range=[-1, 1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Toon statistieken onder correlatiegrafiek
            st.subheader("Statistieken")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.metric("Pearson Correlatie", f"{pearson_corr:.4f}")
                st.metric("Rolling Correlatie (huidig)", f"{df['rolling_corr'].iloc[-1]:.4f}")
                st.metric("Beta (β)", f"{beta:.4f}")
            
            with stats_col2:
                st.metric("Alpha (α)", f"{alpha:.6f}")
                st.metric("R-squared", f"{r_squared:.4f}")
                st.metric("Spread Volatiliteit", f"{spread_std:.4f}")
        
        with col2:
            # Returns scatterplot
            fig_scatter = px.scatter(
                df.dropna(), 
                x='returns1', 
                y='returns2',
                title=f"Returns Scatterplot ({name1} vs {name2})",
                labels={'returns1': f'{name1} Returns', 'returns2': f'{name2} Returns'},
                color_discrete_sequence=['#17becf'],
                opacity=0.6
            )
            
            # Voeg regressielijn toe
            fig_scatter.add_trace(go.Scatter(
                x=np.linspace(df['returns1'].min(), df['returns1'].max(), 100),
                y=alpha + beta * np.linspace(df['returns1'].min(), df['returns1'].max(), 100),
                mode='lines',
                line=dict(color='#d62728', width=2),
                name='Regressielijn'
            ))
            
            fig_scatter.update_layout(
                xaxis_title=f"{name1} Returns",
                yaxis_title=f"{name2} Returns",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600,
                showlegend=True
            )
            
            # Pas marker grootte en stijl aan
            fig_scatter.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

# === BACKTESTING SECTIE ===
with st.expander("🔙 Backtesting", expanded=False):
    st.header("🔙 Backtesting")
    
    # Backtesting parameters
    with st.sidebar:
        st.markdown("---")
        st.header("🎯 Backtesting Instellingen")
        initial_capital = st.number_input("Startkapitaal (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
        transaction_cost = st.slider("Transactiekosten (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        max_position_size = st.slider("Max positie grootte (% van kapitaal)", min_value=10, max_value=100, value=50, step=10)
        
        st.subheader("🛡️ Risk Management")
        stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        take_profit_pct = st.slider("Take Profit (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    
    # Backtesting functie
    def run_backtest(df, entry_threshold, exit_threshold, initial_capital, transaction_cost, max_position_size, stop_loss_pct, take_profit_pct):
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
        
        cash = initial_capital
        position = 0
        coin1_shares = 0
        coin2_shares = 0
        entry_price1 = 0
        entry_price2 = 0
        entry_date = None
        position_value = 0
        
        trades = []
        portfolio_values = []
        positions = []
        
        max_position_value = (max_position_size / 100) * initial_capital
        
        for i in range(len(df)):
            current_zscore = df['zscore'].iloc[i]
            current_price1 = df['price1'].iloc[i]
            current_price2 = df['price2'].iloc[i]
            current_date = df.index[i]
            
            position_market_value = coin1_shares * current_price1 + coin2_shares * current_price2
            portfolio_value = cash + position_market_value
            
            if position == 0 and i > 0:
                if current_zscore < -entry_threshold:
                    position = 1
                    position_value = min(max_position_value, portfolio_value * 0.95)
                    
                    half_position = position_value / 2
                    coin2_shares = half_position / current_price2
                    coin1_shares = -half_position / current_price1
                    
                    entry_price1 = current_price1
                    entry_price2 = current_price2
                    entry_date = current_date
                    
                    transaction_costs = position_value * (transaction_cost / 100)
                    cash -= transaction_costs
                    
                elif current_zscore > entry_threshold:
                    position = -1
                    position_value = min(max_position_value, portfolio_value * 0.95)
                    
                    half_position = position_value / 2
                    coin1_shares = half_position / current_price1
                    coin2_shares = -half_position / current_price2
                    
                    entry_price1 = current_price1
                    entry_price2 = current_price2
                    entry_date = current_date
                    
                    transaction_costs = position_value * (transaction_cost / 100)
                    cash -= transaction_costs
            
            elif position != 0:
                exit_trade = False
                exit_reason = ""
                
                if abs(current_zscore) < exit_threshold:
                    exit_trade = True
                    exit_reason = "Z-score exit"
                
                current_position_value = abs(coin1_shares * current_price1) + abs(coin2_shares * current_price2)
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
                    final_pnl = pnl_dollar
                    exit_transaction_costs = current_position_value * (transaction_cost / 100)
                    final_pnl -= exit_transaction_costs
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
    
    # Voer backtest uit
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
    
    # Toon resultaten
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        # Portfolio metrics
        final_value = df_backtest['portfolio_value'].iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Layout met tabs
        tab1, tab2, tab3 = st.tabs(["Performance", "Trades", "Grafieken"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Totaal Rendement", f"{total_return:.2f}%")
                st.metric("Eindwaarde", f"${final_value:,.0f}")
                st.metric("Aantal Trades", len(trades_df))
            
            with col2:
                winning_trades = trades_df[trades_df['P&L'] > 0]
                losing_trades = trades_df[trades_df['P&L'] < 0]
                win_rate = (len(winning_trades) / len(trades_df)) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Gemiddelde Win", f"${winning_trades['P&L'].mean():.0f}")
                st.metric("Gemiddelde Loss", f"${losing_trades['P&L'].mean():.0f}")
            
            with col3:
                profit_factor = abs(winning_trades['P&L'].sum() / losing_trades['P&L'].sum()) if len(losing_trades) > 0 else float('inf')
                st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                returns = df_backtest['portfolio_value'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                max_drawdown = ((df_backtest['portfolio_value'].cummax() - df_backtest['portfolio_value']) / df_backtest['portfolio_value'].cummax()).max() * 100
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        with tab2:
            # Maak trades tabel meer leesbaar
            trades_display = trades_df.copy()
            trades_display['Entry Date'] = trades_display['Entry Date'].dt.strftime('%Y-%m-%d')
            trades_display['Exit Date'] = trades_display['Exit Date'].dt.strftime('%Y-%m-%d')
            trades_display['P&L'] = trades_display['P&L'].apply(lambda x: f"${x:,.0f}")
            trades_display['Position Size'] = trades_display['Position Size'].apply(lambda x: f"${x:,.0f}")
            trades_display['P&L %'] = trades_display['P&L %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(trades_display, use_container_width=True)
        
        with tab3:
            # Portfolio value grafiek
            fig_portfolio = go.Figure()
            fig_portfolio.add_trace(go.Scatter(
                x=df_backtest.index,
                y=df_backtest['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ))
            
            # Voeg trade markers toe
            for trade in trades:
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
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # P&L distributie
            fig_pnl = px.histogram(
                trades_df, 
                x='P&L %', 
                nbins=20,
                title="P&L Distributie (%)",
                labels={'P&L %': 'P&L Percentage', 'count': 'Aantal Trades'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.warning("Geen trades uitgevoerd in de backtesting periode. Probeer andere parameters.")

# Export functionaliteit
with st.expander("📤 Export", expanded=False):
    st.header("📤 Exporteer Data")
    
    if st.button("Exporteer analyse naar CSV"):
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
    
    if len(trades) > 0 and st.button("Exporteer trade history naar CSV"):
        trades_csv = pd.DataFrame(trades).to_csv(index=False)
        st.download_button(
            label="Download Trade History CSV",
            data=trades_csv,
            file_name=f"trade_history_{name1}_{name2}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
