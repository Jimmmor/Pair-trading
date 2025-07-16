import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Pairs Trading Strategy",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("🚀 Realistische Pairs Trading Strategie: €100 → €1000")

# Sidebar for phase selection
st.sidebar.header("Strategy Phases")
phase = st.sidebar.selectbox(
    "Select Phase",
    ["Fase 1: Foundation", "Fase 2: Growth", "Fase 3: Acceleration", "Fase 4: Final Push"]
)

# Main content based on phase
if phase == "Fase 1: Foundation":
    st.header("📊 Fase 1: Foundation (Maand 1-3)")
    st.write("**Target**: €100 → €200 (26% maandelijks compound return)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maand 1", "€126", "€26")
    with col2:
        st.metric("Maand 2", "€159", "€33")
    with col3:
        st.metric("Maand 3", "€200", "€41")
    
    st.subheader("Strategie Parameters")
    st.write("""
    - Position size: 8-12% per trade
    - Z-score threshold: 2.0
    - Max 2 posities tegelijk
    - Focus op ETH/BTC, SOL/ETH pairs
    """)

elif phase == "Fase 2: Growth":
    st.header("📈 Fase 2: Growth (Maand 4-9)")
    st.write("**Target**: €200 → €600 (20% maandelijks compound return)")
    
    # Create growth chart
    months = ["Maand 4", "Maand 5", "Maand 6", "Maand 7", "Maand 8", "Maand 9"]
    values = [240, 288, 346, 415, 498, 600]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=values,
        mode='lines+markers',
        name='Account Value',
        line=dict(color='green', width=3)
    ))
    fig.update_layout(
        title="Growth Phase Progress",
        xaxis_title="Month",
        yaxis_title="Account Value (€)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

elif phase == "Fase 3: Acceleration":
    st.header("🚀 Fase 3: Acceleration (Maand 10-15)")
    st.write("**Target**: €600 → €900 (15% maandelijks compound return)")
    
    st.subheader("Recommended Pairs")
    pairs_data = {
        "Pair": ["ATOM/NEAR", "APT/SUI", "LDO/AAVE", "CRV/1INCH"],
        "Type": ["Ecosystem plays", "New L1 competition", "DeFi yield pairs", "DEX pairs"],
        "Risk Level": ["Medium", "High", "Medium", "Medium"]
    }
    st.dataframe(pd.DataFrame(pairs_data))

else:  # Fase 4
    st.header("🎯 Fase 4: Final Push (Maand 16-18)")
    st.write("**Target**: €900 → €1000 (3.7% maandelijks compound return)")
    
    st.success("Conservative completion phase - Focus on capital preservation!")

# Position Size Calculator
st.header("💰 Position Size Calculator")
col1, col2 = st.columns(2)
with col1:
    account_value = st.number_input("Current Account Value (€)", min_value=100, max_value=1000, value=100)
with col2:
    selected_phase = st.selectbox(
        "Current Phase",
        ["Fase 1 (<€200)", "Fase 2 (€200-600)", "Fase 3 (€600-900)", "Fase 4 (>€900)"]
    )

# Calculate position size
def calculate_position_size(account_value, phase):
    if phase == "Fase 1 (<€200)":
        return min(account_value * 0.10, 20)
    elif phase == "Fase 2 (€200-600)":
        return min(account_value * 0.12, 70)
    elif phase == "Fase 3 (€600-900)":
        return min(account_value * 0.10, 90)
    else:  # Fase 4
        return min(account_value * 0.08, 80)

position_size = calculate_position_size(account_value, selected_phase)
st.metric("Recommended Position Size", f"€{position_size:.2f}")

# Trading Parameters
st.header("⚙️ Trading Parameters")
current_phase_num = 1 if "Fase 1" in selected_phase else 2 if "Fase 2" in selected_phase else 3 if "Fase 3" in selected_phase else 4

# Parameters based on phase
if current_phase_num == 1:
    params = {
        "Z-score Entry": 2.0,
        "Z-score Exit": 0.3,
        "Correlation Window": 20,
        "Stop Loss": "12%",
        "Min Correlation": 0.65
    }
elif current_phase_num == 2:
    params = {
        "Z-score Entry": 2.2,
        "Z-score Exit": 0.4,
        "Correlation Window": 18,
        "Stop Loss": "10%",
        "Min Correlation": 0.6
    }
elif current_phase_num == 3:
    params = {
        "Z-score Entry": 2.5,
        "Z-score Exit": 0.2,
        "Correlation Window": 15,
        "Stop Loss": "8%",
        "Min Correlation": 0.55
    }
else:
    params = {
        "Z-score Entry": 2.0,
        "Z-score Exit": 0.5,
        "Correlation Window": 25,
        "Stop Loss": "15%",
        "Min Correlation": 0.7
    }

# Display parameters in columns
cols = st.columns(len(params))
for i, (key, value) in enumerate(params.items()):
    with cols[i]:
        st.metric(key, value)

# Compound Growth Simulator
st.header("📊 Compound Growth Simulator")
months = list(range(1, 19))
account_values = [100]  # Starting value

monthly_rates = {
    1: 0.26, 2: 0.26, 3: 0.26,  # Fase 1
    4: 0.20, 5: 0.20, 6: 0.20, 7: 0.20, 8: 0.20, 9: 0.20,  # Fase 2
    10: 0.15, 11: 0.15, 12: 0.15, 13: 0.15, 14: 0.15, 15: 0.15,  # Fase 3
    16: 0.037, 17: 0.037, 18: 0.037  # Fase 4
}

for month in months:
    if month == 1:
        continue
    prev_value = account_values[-1]
    new_value = prev_value * (1 + monthly_rates[month])
    account_values.append(new_value)

# Create compound growth chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=months,
    y=account_values,
    mode='lines+markers',
    name='Account Growth',
    line=dict(color='blue', width=3)
))

# Add phase boundaries
fig.add_vline(x=3.5, line_dash="dash", line_color="red", annotation_text="Fase 1 → 2")
fig.add_vline(x=9.5, line_dash="dash", line_color="red", annotation_text="Fase 2 → 3")
fig.add_vline(x=15.5, line_dash="dash", line_color="red", annotation_text="Fase 3 → 4")

fig.update_layout(
    title="18-Month Compound Growth Plan",
    xaxis_title="Month",
    yaxis_title="Account Value (€)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Final metrics
st.header("🎊 Success Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Win Rate Target", "55-65%")
with col2:
    st.metric("Average Win", "6-10%")
with col3:
    st.metric("Average Loss", "3-5%")
with col4:
    st.metric("Max Drawdown", "<25%")

# Footer
st.markdown("---")
st.markdown("**Remember**: Consistent small wins compound better than seeking big trades!")
