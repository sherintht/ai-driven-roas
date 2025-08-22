import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# Page configuration
st.set_page_config(
    page_title="AI-Powered ROAS Insights",
    page_icon="🤖",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error("⚠️ AI features unavailable. Please configure OpenAI API key.")
        return None

client = init_openai_client()

# Professional AI Insights Generator
class AIInsightsEngine:
    def __init__(self, client, campaign_data):
        self.client = client
        self.data = campaign_data
        self.context = self._build_context()
    
    def _build_context(self):
        """Build rich context from campaign data"""
        summary = {
            "total_campaigns": self.data["campaign"].nunique(),
            "total_channels": self.data["channel"].nunique(),
            "avg_daily_spend": self.data["spend"].mean(),
            "avg_roas": self.data["roas_day_1"].mean(),
            "top_campaign": self.data.groupby("campaign")["roas_day_1"].mean().idxmax(),
            "top_channel": self.data.groupby("channel")["roas_day_1"].mean().idxmax(),
            "date_range": f"{self.data['date'].min()} to {self.data['date'].max()}",
            "performance_trends": self._get_trends()
        }
        return summary
    
    def _get_trends(self):
        """Analyze performance trends"""
        daily_perf = self.data.groupby("date").agg({
            "spend": "sum",
            "revenue": "sum",
            "roas_day_1": "mean"
        }).reset_index()
        
        recent_roas = daily_perf.tail(7)["roas_day_1"].mean()
        older_roas = daily_perf.head(7)["roas_day_1"].mean()
        trend = "improving" if recent_roas > older_roas else "declining"
        
        return {
            "trend_direction": trend,
            "recent_avg_roas": recent_roas,
            "change_percentage": ((recent_roas - older_roas) / older_roas) * 100
        }
    
    def generate_insight(self, user_question, max_tokens=300):
        """Generate AI-powered insights with error handling"""
        if not self.client:
            return "❌ AI features are currently unavailable."
        
        if not user_question.strip():
            return "💡 Please ask a specific question about your campaigns."
        
        try:
            # Professional system prompt
            system_prompt = f"""You are a senior marketing data analyst and ROAS optimization expert. 
            
            Campaign Context:
            - {self.context['total_campaigns']} campaigns across {self.context['total_channels']} channels
            - Average daily spend: ${self.context['avg_daily_spend']:,.0f}
            - Average ROAS: {self.context['avg_roas']:.2f}×
            - Top performing campaign: {self.context['top_campaign']}
            - Top performing channel: {self.context['top_channel']}
            - Performance trend: {self.context['performance_trends']['trend_direction']}
            - Recent ROAS change: {self.context['performance_trends']['change_percentage']:+.1f}%
            
            Provide concise, actionable insights in professional language. Include specific numbers and recommendations when relevant."""
            
            # Rate limiting check
            if 'last_request_time' not in st.session_state:
                st.session_state.last_request_time = 0
            
            import time
            if time.time() - st.session_state.last_request_time < 2:
                return "⏳ Please wait a moment between requests."
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Cost-effective choice
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Lower for more consistent business advice
            )
            
            st.session_state.last_request_time = time.time()
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Unable to generate insight: {str(e)[:100]}..."
    
    def get_automated_recommendations(self):
        """Generate automatic strategic recommendations"""
        recommendations = []
        
        # Channel performance analysis
        channel_roas = self.data.groupby("channel")["roas_day_1"].mean()
        best_channel = channel_roas.idxmax()
        worst_channel = channel_roas.idxmin()
        
        recommendations.append({
            "title": "Channel Optimization",
            "insight": f"Consider shifting budget from {worst_channel} (ROAS: {channel_roas[worst_channel]:.2f}×) to {best_channel} (ROAS: {channel_roas[best_channel]:.2f}×)",
            "impact": "Potential 15-25% ROAS improvement"
        })
        
        # Weekend performance
        weekend_roas = self.data[self.data["Weekend"]]["roas_day_1"].mean()
        weekday_roas = self.data[~self.data["Weekend"]]["roas_day_1"].mean()
        
        if weekend_roas > weekday_roas:
            recommendations.append({
                "title": "Weekend Opportunity",
                "insight": f"Weekend ROAS ({weekend_roas:.2f}×) outperforms weekdays ({weekday_roas:.2f}×)",
                "impact": "Scale weekend campaigns for higher returns"
            })
        
        return recommendations

# Title and description
st.title("🤖 AI-Powered ROAS Insights Dashboard")
st.markdown("Upload your campaign data to get AI-driven insights, forecasts, and optimization recommendations.")

# Sidebar: Upload
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Select your marketing CSV file",
        type=["csv"],
        help="Required: date,campaign,channel,spend,installs,conversions,revenue,roas_day_1"
    )
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

if not uploaded_file:
    st.info("📂 Please upload your campaign CSV to begin.")
    st.stop()

# Load and prepare data (same as before)
df = pd.read_csv(uploaded_file)

@st.cache_data
def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date","spend","installs","conversions","revenue","roas_day_1"], inplace=True)
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

# Initialize AI Engine
ai_engine = AIInsightsEngine(client, df) if client else None

# Sidebar: Choose view
with st.sidebar:
    st.header("2. View Options")
    view = st.selectbox(
        "Select view:",
        ["AI Insights", "Overview", "Predict ROAS", "Time Forecast", "Budget Simulator"]
    )

# AI Insights View
if view == "AI Insights":
    st.subheader("🤖 AI Marketing Consultant")
    
    if not ai_engine:
        st.error("⚠️ AI features require OpenAI API configuration.")
        st.stop()
    
    # Automated recommendations
    st.markdown("### 🎯 Automated Recommendations")
    recommendations = ai_engine.get_automated_recommendations()
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"💡 {rec['title']}", expanded=i==0):
            st.write(f"**Insight:** {rec['insight']}")
            st.write(f"**Potential Impact:** {rec['impact']}")
    
    # Interactive Q&A
    st.markdown("### 💬 Ask the AI Consultant")
    
    # Suggested questions
    suggested_questions = [
        "What's my best performing campaign and why?",
        "Which channel should I invest more budget in?",
        "How can I improve my overall ROAS?",
        "What trends do you see in my campaign performance?"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask any question about your campaigns:",
            placeholder="e.g., Which campaigns should I scale up?"
        )
    
    with col2:
        st.markdown("**Quick Questions:**")
        for q in suggested_questions:
            if st.button(q, key=f"suggest_{hash(q)}"):
                user_question = q
    
    if st.button("🔍 Get AI Insight", type="primary"):
        if user_question.strip():
            with st.spinner("AI is analyzing your data..."):
                insight = ai_engine.generate_insight(user_question)
                st.markdown("### 🤖 AI Response")
                st.markdown(insight)
        else:
            st.warning("Please enter a question first.")

# Rest of your existing views (Overview, Predict ROAS, etc.) remain the same
# ... (include all previous view code here)

st.markdown("---")
st.caption("© Your Company — AI-Powered ROAS Insights | Powered by OpenAI GPT")