import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ROAS Insights Dashboard",
    page_icon="💡",
    layout="wide"
)

st.title("💡 ROAS Insights Dashboard")
st.markdown(
    "Analyze your campaign performance, forecast returns, "
    "and simulate budget changes—all without external libraries."
)

# Sidebar: Upload
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Select your marketing CSV file",
        type=["csv"],
        help="Required columns: date, campaign, channel, spend, installs, conversions, revenue, roas_day_1"
    )
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

if not uploaded_file:
    st.info("📂 Please upload your campaign CSV to begin.")
    st.stop()

# Load and prepare data
df = pd.read_csv(uploaded_file)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date","spend","installs","conversions","revenue","roas_day_1"], inplace=True)

# Feature engineering
df["CAC"] = df["spend"] / df["installs"].replace(0, np.nan)
df["CAC"].fillna(0, inplace=True)
df["Weekend"] = df["date"].dt.weekday >= 5

required = ["date","campaign","channel","spend","installs","conversions","revenue","roas_day_1"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Sidebar: View options
with st.sidebar:
    st.header("2. View Options")
    view = st.selectbox(
        "Select view:",
        ["Overview", "Average ROAS", "Time Forecast", "Budget Simulator"]
    )

# 1. Overview
if view == "Overview":
    st.subheader("📊 Data Overview")
    st.metric("Days of Data", f"{len(df)}")
    st.metric("Campaigns", df["campaign"].nunique())
    st.metric("Channels", df["channel"].nunique())
    span = df["date"].max() - df["date"].min()
    st.metric("Date Range", f"{span.days} days")
    st.markdown("**Sample Records**")
    st.dataframe(df[required].head(8), use_container_width=True)

# 2. Average ROAS (in place of ML forecast)
elif view == "Average ROAS":
    st.subheader("📈 Day-1 ROAS by Campaign (Historical Average)")
    avg_roas = df.groupby("campaign")["roas_day_1"].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_roas.values, y=avg_roas.index, orientation="h",
        labels={"x":"Avg Day-1 ROAS (×)", "y":"Campaign"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "This uses each campaign’s historical Day-1 ROAS as the “forecast.”"
    )

# 3. Time Forecast (7-day moving average)
elif view == "Time Forecast":
    st.subheader("🔮 30-Day Revenue Forecast (Moving Average)")
    camp = st.selectbox("Select Campaign", df["campaign"].unique())
    chan = st.selectbox("Select Channel", df["channel"].unique())
    subset = (
        df[(df["campaign"]==camp)&(df["channel"]==chan)]
        .set_index("date").resample("D")["revenue"].sum()
    ).fillna(0)
    subset = subset.to_frame().assign(
        MA_7=subset.rolling(7, min_periods=1).mean()
    )
    last = subset.index.max()
    future = pd.date_range(last+pd.Timedelta(days=1), periods=30)
    forecast = pd.Series(subset["MA_7"].iloc[-7:].mean(), index=future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset.index, y=subset["revenue"], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Projected 30-day Revenue:** ${forecast.sum():,.0f}")

# 4. Budget Simulator
else:
    st.subheader("💰 Budget Reallocation Simulator")
    st.markdown("Adjust channel spend and see potential ROAS changes (using historical averages).")
    channels = df["channel"].unique()
    adjustments = {ch: st.sidebar.slider(f"{ch} ±%", -50,50,0)/100 for ch in channels}
    if st.button("Run Simulation"):
        df2 = df.copy()
        for ch,pct in adjustments.items():
            df2.loc[df2["channel"]==ch,"spend"] *= (1+pct)
        df2["roas_avg"] = df2.groupby("campaign")["roas_day_1"].transform("mean")
        orig = (df["roas_day_1"]*df["spend"]).sum()/df["spend"].sum()
        new = (df2["roas_avg"]*df2["spend"]).sum()/df2["spend"].sum()
        uplift = (new-orig)/orig*100
        st.metric("Original ROAS", f"{orig:.2f}×")
        st.metric("New ROAS", f"{new:.2f}×", delta=f"{new-orig:.2f}×")
        st.write(f"**ROAS Change:** {uplift:+.1f}%")

st.markdown("---")
st.caption("© Your Company — Streamlit-only Demo")
