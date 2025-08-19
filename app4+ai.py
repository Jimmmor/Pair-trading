# Fix 1: Add missing tickers dictionary at the top
# Add this after your imports if it's missing:
from constants.tickers import tickers

# If the constants/tickers.py file doesn't exist, create it or add this directly:
# tickers = {
#     'Bitcoin (BTC)': 'BTC-USD',
#     'Ethereum (ETH)': 'ETH-USD',
#     'Dogecoin (DOGE)': 'DOGE-USD',
#     'Cardano (ADA)': 'ADA-USD',
#     'Solana (SOL)': 'SOL-USD',
#     'Polygon (MATIC)': 'MATIC-USD',
#     'Chainlink (LINK)': 'LINK-USD',
#     'Litecoin (LTC)': 'LTC-USD',
#     'Bitcoin Cash (BCH)': 'BCH-USD',
#     'Avalanche (AVAX)': 'AVAX-USD'
# }

# Fix 2: Improved generate_trading_signals method to handle the TypeError
def generate_trading_signals(self, price1, price2, entry_zscore=2.0, exit_zscore=0.5, stop_zscore=3.5):
    """Generate trading signals with better error handling"""
    try:
        analysis_df, hedge_ratio = self.calculate_spread_zscore(price1, price2)
        
        if analysis_df.empty or len(analysis_df) < 10:
            st.warning("Insufficient data for signal generation")
            return pd.DataFrame()
        
        signals = []
        current_position = 0  # 0=flat, 1=long_spread, -1=short_spread
        
        for timestamp, row in analysis_df.iterrows():
            try:
                # Validate all numeric values with more robust checking
                zscore = float(row['zscore']) if pd.notna(row['zscore']) and np.isfinite(row['zscore']) else 0.0
                price1_val = float(row['price1']) if pd.notna(row['price1']) and np.isfinite(row['price1']) else 0.0
                price2_val = float(row['price2']) if pd.notna(row['price2']) and np.isfinite(row['price2']) else 0.0
                spread_val = float(row['spread']) if pd.notna(row['spread']) and np.isfinite(row['spread']) else 0.0
                
                # Skip if any critical values are zero or invalid
                if price1_val <= 0 or price2_val <= 0:
                    continue
                
                signal_data = {
                    'timestamp': timestamp,
                    'price1': price1_val,
                    'price2': price2_val,
                    'spread': spread_val,
                    'zscore': zscore,
                    'position': current_position,
                    'action': 'HOLD',
                    'entry_price': None,
                    'exit_price': None
                }
                
                # Signal generation logic (rest remains the same)
                if current_position == 0:  # No position
                    if zscore <= -entry_zscore:  # Enter long spread
                        signal_data['action'] = 'ENTER_LONG_SPREAD'
                        signal_data['entry_price'] = spread_val
                        current_position = 1
                        
                    elif zscore >= entry_zscore:  # Enter short spread
                        signal_data['action'] = 'ENTER_SHORT_SPREAD'
                        signal_data['entry_price'] = spread_val
                        current_position = -1
                
                elif current_position == 1:  # Long spread position
                    if zscore >= -exit_zscore:  # Normal exit
                        signal_data['action'] = 'EXIT_LONG_SPREAD'
                        signal_data['exit_price'] = spread_val
                        current_position = 0
                    elif zscore <= -stop_zscore:  # Stop loss
                        signal_data['action'] = 'STOP_LONG_SPREAD'
                        signal_data['exit_price'] = spread_val
                        current_position = 0
                
                elif current_position == -1:  # Short spread position
                    if zscore <= exit_zscore:  # Normal exit
                        signal_data['action'] = 'EXIT_SHORT_SPREAD'
                        signal_data['exit_price'] = spread_val
                        current_position = 0
                    elif zscore >= stop_zscore:  # Stop loss
                        signal_data['action'] = 'STOP_SHORT_SPREAD'
                        signal_data['exit_price'] = spread_val
                        current_position = 0
                
                signal_data['position'] = current_position
                signals.append(signal_data)
                
            except (ValueError, TypeError, KeyError) as e:
                # Log the specific row that caused issues and continue
                st.warning(f"Skipping invalid data row at {timestamp}: {e}")
                continue
        
        if not signals:
            st.warning("No valid signals could be generated from the data")
            return pd.DataFrame()
        
        return pd.DataFrame(signals)
    
    except Exception as e:
        st.error(f"Signal generation failed: {e}")
        return pd.DataFrame()

# Fix 3: Add data validation in the main analysis section
# Replace the data validation section around line 754 with:
if analyze_button or optimize_button:
    with st.spinner(f"Fetching data for {asset1_name} and {asset2_name}..."):
        try:
            price1 = system.fetch_price_data(asset1_ticker, data_period)
            price2 = system.fetch_price_data(asset2_ticker, data_period)
        except Exception as e:
            st.error(f"Data fetching failed: {e}")
            price1, price2 = pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Enhanced data validation
    min_data_points = 50
    if (isinstance(price1, pd.Series) and isinstance(price2, pd.Series) and 
        not price1.empty and not price2.empty and 
        len(price1) >= min_data_points and len(price2) >= min_data_points):
        
        # Additional data quality checks
        if price1.isna().sum() > len(price1) * 0.1:  # More than 10% NaN
            st.warning(f"{asset1_name} has {price1.isna().sum()} missing values")
        
        if price2.isna().sum() > len(price2) * 0.1:  # More than 10% NaN
            st.warning(f"{asset2_name} has {price2.isna().sum()} missing values")
        
        # Clean the data
        price1 = price1.dropna()
        price2 = price2.dropna()
        
        st.success(f"Data loaded: {asset1_name} ({len(price1)} points), {asset2_name} ({len(price2)} points)")
        
        # Continue with the rest of your analysis...
        # (correlation analysis, optimization, etc.)
    
    else:
        # Better error messaging with specific issues
        issues = []
        if not isinstance(price1, pd.Series) or price1.empty:
            issues.append(f"No data for {asset1_name} ({asset1_ticker})")
        elif len(price1) < min_data_points:
            issues.append(f"Insufficient {asset1_name} data: {len(price1)} points")
            
        if not isinstance(price2, pd.Series) or price2.empty:
            issues.append(f"No data for {asset2_name} ({asset2_ticker})")
        elif len(price2) < min_data_points:
            issues.append(f"Insufficient {asset2_name} data: {len(price2)} points")
        
        for issue in issues:
            st.error(issue)
        
        st.info("Try selecting different assets or a longer time period (1y or 2y)")
