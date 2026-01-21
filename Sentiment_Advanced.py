import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import random
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sentimind AI Enterprise",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #F4F6F9; color: #212529; }
        
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border-left: 5px solid #2C3E50;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }

        h1, h2, h3 { font-family: 'Inter', sans-serif; color: #111827; font-weight: 700; }
        
        section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 15px; border-bottom: 2px solid #E5E7EB; }
        .stTabs [data-baseweb="tab"] { height: 50px; border: none; font-weight: 600; color: #6B7280; }
        .stTabs [aria-selected="true"] { color: #2563EB; border-bottom: 3px solid #2563EB; }
        
        .alert-box {
            padding: 15px;
            background-color: #FEF2F2;
            color: #991B1B;
            border-left: 5px solid #EF4444;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

class SentimentPro:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.colors = {'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#3B82F6'}

    def generate_dummy_data(self):
        base_date = datetime.now() - timedelta(days=30)
        
        feedback_samples = [
            ("The new UI is blazing fast and smooth.", "Positive"),
            ("Login page is crashing repeatedly.", "Negative"),
            ("Support team helped me fix the issue quickly.", "Positive"),
            ("Subscription pricing is too high for these features.", "Negative"),
            ("Dark mode looks amazing on mobile.", "Positive"),
            ("Documentation is confusing and needs update.", "Neutral"),
            ("API response time is very slow today.", "Negative"),
            ("Great update, loved the new dashboard widgets.", "Positive"),
            ("Cannot export PDF reports, getting 404 error.", "Negative"),
            ("Python SDK integration was seamless.", "Positive"),
            ("The app is okay, but lacks some customization.", "Neutral"),
            ("Security features are top-notch.", "Positive"),
            ("My account got locked without reason. Terrible.", "Negative"),
            ("Just a regular update, nothing special.", "Neutral")
        ]
        
        data = []
        for i in range(50):
            text, sentiment_bias = random.choice(feedback_samples)
            date = base_date + timedelta(days=random.randint(0, 30))
            data.append({'date': date, 'posts': text})
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df.sort_values('date')

    def load_data(self, uploaded_file):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'posts' not in df.columns:
                    st.error("Error: CSV must contain a 'posts' column.")
                    return None
                if 'date' not in df.columns:
                    base_date = datetime.now()
                    df['date'] = [base_date - timedelta(days=x) for x in range(len(df))]
                df['date'] = pd.to_datetime(df['date']).dt.date
                return df
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None
        else:
            return self.generate_dummy_data()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def analyze(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        blob = TextBlob(text)
        nouns = [w for w, t in blob.tags if t.startswith('NN')] if blob.tags else []
        
        if compound >= 0.05:
            return 'Positive', compound, nouns
        elif compound <= -0.05:
            return 'Negative', compound, nouns
        else:
            return 'Neutral', compound, nouns

    def run(self):
        with st.sidebar:
            st.title("Sentiment")
            st.markdown("Enterprise Intelligence")
            st.divider()
            
            uploaded_file = st.file_uploader("Upload Data (CSV)", type=['csv'])
            
            st.subheader("Filters")
            sentiment_filter = st.multiselect("Filter by Sentiment", ['Positive', 'Negative', 'Neutral'], default=['Positive', 'Negative', 'Neutral'])
            
            st.divider()
            st.info("System Ready")

        st.markdown("## Executive Analytics Dashboard")
        st.markdown(f"**Live Overview:** {datetime.now().strftime('%B %d, %Y')}")
        st.write("")

        df = self.load_data(uploaded_file)
        
        if df is not None:
            df['Cleaned_Posts'] = df['posts'].apply(self.clean_text)
            
            analysis = df['Cleaned_Posts'].apply(lambda x: self.analyze(x))
            df['Sentiment'] = [x[0] for x in analysis]
            df['Score'] = [x[1] for x in analysis]
            df['Topics'] = [x[2] for x in analysis]

            filtered_df = df[df['Sentiment'].isin(sentiment_filter)]

            c1, c2, c3, c4 = st.columns(4)
            total = len(filtered_df)
            pos_count = len(filtered_df[filtered_df['Sentiment']=='Positive'])
            neg_count = len(filtered_df[filtered_df['Sentiment']=='Negative'])
            
            c1.metric("Total Reviews", total, "Posts")
            c2.metric("Positive", pos_count, f"{pos_count/total*100:.1f}%" if total else "0%", delta_color="normal")
            c3.metric("Negative", neg_count, f"{neg_count/total*100:.1f}%" if total else "0%", delta_color="inverse")
            
            avg = filtered_df['Score'].mean()
            delta = "High" if avg > 0.5 else "Low"
            c4.metric("Avg Quality Score", f"{avg:.2f}", delta)

            st.markdown("---")

            if neg_count > (total * 0.3) and total > 0:
                st.markdown(f"""
                <div class="alert-box">
                    <strong>CRITICAL ALERT:</strong> Negative sentiment is above 30%. Immediate attention required on recent feedback.
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Trends & Topics", "Deep Analysis", "Raw Data"])

            with tab1:
                col_trend, col_topic = st.columns([2, 1])
                
                with col_trend:
                    st.subheader("Sentiment Trends Over Time")
                    trend_data = filtered_df.groupby(['date', 'Sentiment']).size().reset_index(name='count')
                    
                    fig_trend = px.line(trend_data, x='date', y='count', color='Sentiment', 
                                        color_discrete_map=self.colors, markers=True,
                                        title="Daily Sentiment Volume")
                    fig_trend.update_layout(plot_bgcolor="white", hovermode="x unified")
                    st.plotly_chart(fig_trend, use_container_width=True)

                with col_topic:
                    st.subheader("Top Keywords")
                    all_topics = [item for sublist in filtered_df['Topics'] for item in sublist]
                    if all_topics:
                        topic_freq = pd.Series(all_topics).value_counts().head(7)
                        
                        fig_topic = px.bar(x=topic_freq.values, y=topic_freq.index, orientation='h',
                                         title="Most Discussed Terms",
                                         labels={'x': 'Mentions', 'y': 'Topic'},
                                         color_discrete_sequence=['#2C3E50'])
                        fig_topic.update_layout(plot_bgcolor="white")
                        st.plotly_chart(fig_topic, use_container_width=True)
                    else:
                        st.info("No sufficient topics extracted.")

            with tab2:
                st.subheader("Lab Investigation")
                txt = st.text_area("Analyze specific feedback:", placeholder="Paste customer email or chat here...")
                
                if st.button("Run Diagnostics"):
                    if txt:
                        sent, score, _ = self.analyze(self.clean_text(txt))
                        
                        fig_g = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = score,
                            title = {'text': "Sentiment Intensity"},
                            delta = {'reference': 0},
                            gauge = {
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "black"},
                                'steps': [
                                    {'range': [-1, -0.05], 'color': "#EF4444"},
                                    {'range': [-0.05, 0.05], 'color': "#3B82F6"},
                                    {'range': [0.05, 1], 'color': "#10B981"}]
                            }
                        ))
                        st.plotly_chart(fig_g, use_container_width=True)
                        
                        st.success(f"Verdict: {sent}")

            with tab3:
                st.dataframe(filtered_df[['date', 'posts', 'Sentiment', 'Score']], use_container_width=True)
                st.download_button("Download Report", df.to_csv().encode('utf-8'), "report.csv")

if __name__ == "__main__":
    app = SentimentPro()
    app.run()