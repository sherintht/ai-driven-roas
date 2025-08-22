import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import holidays
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ROAS Insights Dashboard",
    page_icon="💡",
    layout="wide"
)

# Title and description
st.title("💡 ROAS Insights Dashboard")
st.markdown(
    "Upload your campaign data to see expected returns, "
    "discover top-performing campaigns, and test budget changes—all in plain English."
)

# Sidebar: Upload
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Select your marketing CSV file",
        type=["csv"],
        help="Required: date, campaign, channel, spend, installs, conversions, revenue, roas_day_1"
    )
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

if not uploaded_file:
    st.info("📂 Please upload your campaign CSV to begin.")
    st.stop()

# Load and prepare data
df = pd.read_csv(uploaded_file)

@st.cache_data
def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "spend", "installs", "conversions", "revenue", "roas_day_1"], inplace=True)
    df["CAC"] = df["spend"] / df["installs"].replace(0, np.nan)
    df["CAC"].fillna(0, inplace=True)
    df["Weekend"] = df["date"].dt.weekday >= 5
    us_hols = holidays.US()
    df["Holiday"] = df["date"].isin(us_hols)
    df["CampCode"] = df["campaign"].astype("category").cat.codes
    df["ChanCode"] = df["channel"].astype("category").cat.codes
    df["Spend/Install"] = df["spend"] / (df["installs"] + 1)
    df["ConvRate"] = df["conversions"] / (df["installs"] + 1)
    df["Rev/Conv"] = df["revenue"] / (df["conversions"] + 1)
    return df

df = prepare(df)

required = ["date", "campaign", "channel", "spend", "installs", "conversions", "revenue", "roas_day_1"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Sidebar: Choose view
with st.sidebar:
    st.header("2. View Options")
    view = st.selectbox(
        "Select view:",
        ["Overview", "Predict ROAS", "Time Forecast", "Budget Simulator"]
    )

# Cache model training
@st.cache_resource
def train_model(data):
    feature_cols = [
        "spend", "installs", "conversions", "CAC",
        "Weekend", "Holiday", "CampCode", "ChanCode",
        "Spend/Install", "ConvRate", "Rev/Conv"
    ]
    X = data[feature_cols]
    y = data["roas_day_1"]
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols

model, features = train_model(df)

# 1. Overview
if view == "Overview":
    st.subheader("📊 Data Overview")
    st.metric("Days of Data", f"{len(df)}")
    st.metric("Unique Campaigns", df["campaign"].nunique())
    st.metric("Unique Channels", df["channel"].nunique())
    span = df["date"].max() - df["date"].min()
    st.metric("Date Range", f"{span.days} days")

    st.markdown("**Sample Records**")
    st.dataframe(df[required].head(8), use_container_width=True)

    st.markdown("**Key Averages**")
    st.write(f"- Average Daily Spend: ${df['spend'].mean():,.0f}")
    st.write(f"- Average Daily Conversions: {df['conversions'].mean():,.0f}")
    st.write(f"- Average Day-1 ROAS: {df['roas_day_1'].mean():.2f}×")

# 2. Predict ROAS
elif view == "Predict ROAS":
    st.subheader("📈 Day-1 ROAS Forecast")
    df["ROAS_Forecast"] = model.predict(df[features])

    # Compute accuracy metrics manually
    y_true = df["roas_day_1"]
    y_pred = df["ROAS_Forecast"]
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    st.markdown(f"**Forecast Accuracy:** Within {mape:.1f}% on average; RMSE = {rmse:.2f}")

    st.markdown("**Top 5 Campaigns by Forecasted Day-1 ROAS**")
    top = df.groupby("campaign")["ROAS_Forecast"].mean().nlargest(5)
    fig = px.bar(
        x=top.values, y=top.index, orientation="h",
        labels={"x": "Forecasted ROAS (×)", "y": "Campaign"}
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Time Forecast
elif view == "Time Forecast":
    st.subheader("🔮 30-Day Revenue Forecast (Moving Average)")
    camp = st.selectbox("Select Campaign", df["campaign"].unique())
    chan = st.selectbox("Select Channel", df["channel"].unique())
    
    subset = df[(df["campaign"] == camp) & (df["channel"] == chan)][["date", "revenue"]].sort_values("date")
    if subset.empty:
        st.warning("No data for that combination.")
    else:
        subset = subset.set_index("date").resample("D").sum().fillna(0)
        subset["7d_MA"] = subset["revenue"].rolling(7, min_periods=1).mean()
        
        last_date = subset.index.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
        forecast = pd.DataFrame(index=future_dates)
        forecast["revenue"] = np.nan
        forecast["MA_Forecast"] = subset["7d_MA"].iloc[-7:].mean()  # constant forecast
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset.index, y=subset["revenue"],
                                 mode="lines", name="Historical Revenue"))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast["MA_Forecast"],
                                 mode="lines", name="Forecasted Revenue"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Projected 30-day Revenue:** ${forecast['MA_Forecast'].sum():,.0f}")

# 4. Budget Simulator
else:
    st.subheader("💰 Budget Reallocation Simulator")
    st.markdown("Adjust each channel’s spend to see potential ROAS change.")
    channels = df["channel"].unique()
    adjustments = {}
    cols = st.columns(2)
    for i, ch in enumerate(channels):
        adjustments[ch] = cols[i % 2].slider(f"{ch} ±%", -50, 50, 0) / 100

    if st.button("Run Simulation"):
        df2 = df.copy()
        for ch, pct in adjustments.items():
            df2.loc[df2["channel"] == ch, "spend"] *= (1 + pct)
        df2 = prepare(df2)
        df2["ROAS_Forecast"] = model.predict(df2[features])

        orig = (df["roas_day_1"] * df["spend"]).sum() / df["spend"].sum()
        new = (df2["ROAS_Forecast"] * df2["spend"]).sum() / df2["spend"].sum()
        uplift = (new - orig) / orig * 100

        st.metric("Original Portfolio ROAS", f"{orig:.2f}×")
        st.metric("New Portfolio ROAS", f"{new:.2f}×", delta=f"{new - orig:.2f}×")
        st.write(f"**Total ROAS Change:** {uplift:+.1f}%")

st.markdown("---")
st.caption("© Your Company — Professional ROAS Insights")
