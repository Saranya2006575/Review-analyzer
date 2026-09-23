"""
Review and Reputation Analyzer Web Application
A Data Mining and Text Mining System built with Streamlit, Scikit-Learn, and Plotly.
"""

import os
import sys
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page layout configuration
st.set_page_config(
    page_title="Review & Reputation Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-quality, professional styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-subtitle {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 6px;
    }
    
    /* Badges */
    .badge-pos {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-neu {
        background-color: #374151;
        color: #E5E7EB;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #991B1B;
        color: #FEE2E2;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-suspicious {
        background-color: #7C2D12;
        color: #FFEDD5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 15px;
        padding-bottom: 6px;
        border-bottom: 2px solid #334155;
    }
    
    /* Info callouts */
    .callout-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules
from modules.preprocessor import TextPreprocessor
from modules.sentiment import SentimentAnalyzer
from modules.classifier import ReviewClassifierPipeline
from modules.clustering import ReviewClusteringPipeline
from modules.frequency import FrequencyMiner
from modules.association import AssociationMiner
from modules.anomaly_detector import ReviewAnomalyDetector
from modules.topics import TopicAnalyzer
from modules.reputation import ReputationEngine
from modules.report_generator import generate_executive_report


# ==========================================
# Caching & State Management
# ==========================================

@st.cache_resource
def get_preprocessor():
    return TextPreprocessor()

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def get_frequency_miner():
    return FrequencyMiner()

@st.cache_resource
def get_association_miner():
    return AssociationMiner()

@st.cache_resource
def get_anomaly_detector():
    return ReviewAnomalyDetector()

@st.cache_resource
def get_topic_analyzer():
    return TopicAnalyzer()

@st.cache_resource
def get_reputation_engine():
    return ReputationEngine()


def load_default_dataset() -> pd.DataFrame:
    """Loads default benchmark sample reviews CSV."""
    default_path = os.path.join(os.path.dirname(__file__), "data", "sample_reviews.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return pd.DataFrame()


def process_dataset(df: pd.DataFrame, text_col: str, rating_col: str, date_col: str) -> pd.DataFrame:
    """Runs complete end-to-end preprocessing, sentiment, and anomaly pipelines."""
    preprocessor = get_preprocessor()
    sentiment_analyzer = get_sentiment_analyzer()
    anomaly_detector = get_anomaly_detector()
    topics_analyzer = get_topic_analyzer()

    # 1. Clean and deduplicate
    clean_df, prep_summary = preprocessor.preprocess_dataframe(
        df, text_col=text_col, rating_col=rating_col, dedup_exact=True
    )

    # 2. Sentiment scoring
    scored_df = sentiment_analyzer.add_sentiment_columns(clean_df, text_col=text_col)

    # 3. Anomaly audit
    audited_df, anomaly_summary = anomaly_detector.audit_dataframe(
        scored_df, text_col=text_col, rating_col=rating_col, sentiment_score_col='Sentiment_Score'
    )

    # 4. Aspect detection
    audited_df['Aspects'] = audited_df[text_col].apply(topics_analyzer.extract_review_aspect_tags)

    st.session_state['prep_summary'] = prep_summary
    st.session_state['anomaly_summary'] = anomaly_summary

    return audited_df


# ==========================================
# Application State Initialization
# ==========================================
if 'df_raw' not in st.session_state:
    default_df = load_default_dataset()
    st.session_state['df_raw'] = default_df
    if not default_df.empty:
        st.session_state['df_processed'] = process_dataset(
            default_df, text_col='Review Text', rating_col='Rating', date_col='Date'
        )
    else:
        st.session_state['df_processed'] = pd.DataFrame()


# ==========================================
# Sidebar Navigation & Filter Controls
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ Review & Reputation")
    st.markdown("<span style='color: #94A3B8; font-size: 0.85rem;'>Data & Text Mining Analytics Engine</span>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "📁 Upload Reviews",
            "🔍 Analyze Single Review",
            "⛏️ Data Mining Results",
            "📑 Reputation Report",
            "📚 Methodology & Architecture"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🎯 Active Scope Filter")

    df_active = st.session_state.get('df_processed', pd.DataFrame())

    if not df_active.empty and 'Product/Business Name' in df_active.columns:
        products = ["All Products / Businesses"] + sorted(list(df_active['Product/Business Name'].dropna().unique()))
        selected_product = st.selectbox("Filter by Product/Brand", products)
        if selected_product != "All Products / Businesses":
            df_filtered = df_active[df_active['Product/Business Name'] == selected_product].copy()
        else:
            df_filtered = df_active.copy()
    else:
        selected_product = "All Products / Businesses"
        df_filtered = df_active.copy()

    st.markdown("---")
    st.markdown(f"**Loaded Reviews**: `{len(df_filtered)}`")
    if not df_filtered.empty and 'Is_Suspicious' in df_filtered.columns:
        susp_count = int(df_filtered['Is_Suspicious'].sum())
        st.markdown(f"**Flagged Suspicious**: `{susp_count}` ({round(susp_count/len(df_filtered)*100, 1)}%)")

    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748B; margin-top: 20px;'>
        PBL Project: Advanced Text Mining<br>
        Framework: Streamlit & Scikit-Learn
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: 📊 EXECUTIVE DASHBOARD
# ==============================================================================
if page == "📊 Executive Dashboard":
    st.title("📊 Customer Reputation & Review Analytics Dashboard")
    st.markdown(f"Real-time Data Mining summary for: **{selected_product}**")

    if df_filtered.empty:
        st.warning("⚠️ No reviews loaded. Please navigate to 'Upload Reviews' to load a dataset.")
        st.stop()

    rep_engine = get_reputation_engine()
    rep_results = rep_engine.calculate_reputation(
        df_filtered, rating_col='Rating', sentiment_col='Sentiment', date_col='Date', suspicious_col='Is_Suspicious'
    )

    metrics = rep_results['metrics']
    rep_score = rep_results['reputation_score']
    grade = rep_results['grade']
    verdict = rep_results['verdict']
    grade_color = rep_results['color']

    # --- Top Metric Cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reputation Score</div>
            <div class="metric-value" style="color: {grade_color};">{rep_score} <span style="font-size: 1.1rem; color: #94A3B8;">/ 100</span></div>
            <div class="metric-subtitle">Grade: <strong>{grade}</strong> ({verdict})</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_r = metrics['avg_rating']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Rating</div>
            <div class="metric-value" style="color: #FBBF24;">{avg_r} <span style="font-size: 1.1rem; color: #94A3B8;">★</span></div>
            <div class="metric-subtitle">Across {metrics['total_reviews']} total reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Reviews</div>
            <div class="metric-value" style="color: #10B981;">{metrics['pct_positive']}%</div>
            <div class="metric-subtitle">{metrics['pos_count']} positive entries</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Negative Reviews</div>
            <div class="metric-value" style="color: #EF4444;">{metrics['pct_negative']}%</div>
            <div class="metric-subtitle">{metrics['neg_count']} customer complaints</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        trend_d = metrics['trend_delta']
        trend_color = "#10B981" if trend_d > 0 else ("#EF4444" if trend_d < 0 else "#94A3B8")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recent Momentum</div>
            <div class="metric-value" style="color: {trend_color};">{'+' if trend_d > 0 else ''}{trend_d}★</div>
            <div class="metric-subtitle">Trend: <strong>{metrics['trend_direction']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # --- Reputation Gauge & Score Breakdown ---
    st.markdown("<div class='section-header'>🧭 Reputation Health & Component Breakdown</div>", unsafe_allow_html=True)
    gauge_col, break_col = st.columns([1.2, 1.8])

    with gauge_col:
        # Plotly Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rep_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Reputation Score ({grade})", 'font': {'size': 18, 'color': '#E2E8F0'}},
            delta={'reference': 70, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                'bar': {'color': grade_color, 'thickness': 0.25},
                'bgcolor': "#1E293B",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 45], 'color': '#7F1D1D'},
                    {'range': [45, 60], 'color': '#7C2D12'},
                    {'range': [60, 75], 'color': '#78350F'},
                    {'range': [75, 90], 'color': '#064E3B'},
                    {'range': [90, 100], 'color': '#065F46'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3},
                    'thickness': 0.8,
                    'value': rep_score
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#E2E8F0"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with break_col:
        comp = rep_results['components']
        comp_df = pd.DataFrame([
            {"Component Factor": "Normalized Average Rating (35%)", "Score": comp['Rating Score (35%)'], "Max": 100},
            {"Component Factor": "Net Sentiment Distribution (30%)", "Score": comp['Sentiment Score (30%)'], "Max": 100},
            {"Component Factor": "Positive / Negative Ratio (15%)", "Score": comp['Pos/Neg Ratio Score (15%)'], "Max": 100},
            {"Component Factor": "Volume Reliability Confidence (10%)", "Score": comp['Volume Confidence (10%)'], "Max": 100},
            {"Component Factor": "Recent Trend Momentum (10%)", "Score": comp['Recent Trend Score (10%)'], "Max": 100}
        ])
        fig_comp = px.bar(
            comp_df, x="Score", y="Component Factor", orientation="h", text="Score",
            color="Score", color_continuous_scale="Viridis", range_x=[0, 105]
        )
        fig_comp.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#E2E8F0"}, coloraxis_showscale=False
        )
        fig_comp.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Charts Row: Sentiment & Rating Distributions ---
    st.markdown("<div class='section-header'>📈 Sentiment Polarity & Star Rating Distributions</div>", unsafe_allow_html=True)
    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        st.markdown("##### Sentiment Distribution")
        sent_df = df_filtered['Sentiment'].value_counts().reset_index()
        sent_df.columns = ['Sentiment', 'Count']
        colors = {'Positive': '#10B981', 'Neutral': '#94A3B8', 'Negative': '#EF4444'}
        fig_sent = px.pie(
            sent_df, values='Count', names='Sentiment', hole=0.55,
            color='Sentiment', color_discrete_map=colors
        )
        fig_sent.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#E2E8F0"})
        st.plotly_chart(fig_sent, use_container_width=True)

    with chart2:
        st.markdown("##### Rating Distribution (1 to 5 Stars)")
        rating_counts = df_filtered['Rating'].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_rating = px.bar(
            rating_counts, x='Rating', y='Count',
            color='Rating', color_continuous_scale='Turbo',
            text='Count'
        )
        fig_rating.update_layout(
            height=280, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#E2E8F0"}, coloraxis_showscale=False
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with chart3:
        st.markdown("##### Historical Reputation Trend")
        if 'Date' in df_filtered.columns:
            try:
                df_trend = df_filtered.copy()
                df_trend['dt'] = pd.to_datetime(df_trend['Date'], errors='coerce')
                df_trend = df_trend.dropna(subset=['dt']).sort_values('dt')
                df_trend['Month'] = df_trend['dt'].dt.to_period('M').astype(str)
                trend_grouped = df_trend.groupby('Month')['Rating'].mean().reset_index()
                fig_trend = px.line(
                    trend_grouped, x='Month', y='Rating', markers=True,
                    line_shape='spline', title="Monthly Average Rating"
                )
                fig_trend.update_layout(
                    height=280, margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#E2E8F0"}, yaxis_range=[1, 5.2]
                )
                fig_trend.update_traces(line_color='#3B82F6', line_width=3)
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception:
                st.info("Time-series trend requires valid date formats.")
        else:
            st.info("Date column not provided in dataset.")

    # --- Keywords & Topics Overview ---
    st.markdown("<div class='section-header'>🔑 Trending Keywords & Extracted Topics</div>", unsafe_allow_html=True)
    kw_col1, kw_col2 = st.columns(2)

    freq_miner = get_frequency_miner()
    aspect_insights = freq_miner.analyze_sentiment_keywords(
        df_filtered, text_col='Clean_Review', sentiment_col='Sentiment', top_n=8
    )

    with kw_col1:
        st.markdown("##### ⭐ Top Praised Features & Aspects")
        praised_df = aspect_insights['praised_aspects']
        if not praised_df.empty:
            fig_p = px.bar(
                praised_df.head(6), x='Frequency', y='Ngram', orientation='h',
                color_discrete_sequence=['#10B981'], text='Frequency'
            )
            fig_p.update_layout(
                height=240, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"}, yaxis={'autorange': 'reversed'}
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Insufficient positive reviews for phrase extraction.")

    with kw_col2:
        st.markdown("##### ⚠️ Top Customer Complaints & Issues")
        complaints_df = aspect_insights['common_complaints']
        if not complaints_df.empty:
            fig_c = px.bar(
                complaints_df.head(6), x='Frequency', y='Ngram', orientation='h',
                color_discrete_sequence=['#EF4444'], text='Frequency'
            )
            fig_c.update_layout(
                height=240, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"}, yaxis={'autorange': 'reversed'}
            )
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("Insufficient negative reviews for complaint extraction.")

    # --- Review Classification Table ---
    st.markdown("<div class='section-header'>📋 Review Classification & Audit Table</div>", unsafe_allow_html=True)

    filter_f1, filter_f2, filter_f3 = st.columns(3)
    with filter_f1:
        filter_sent = st.multiselect("Filter by Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    with filter_f2:
        filter_susp = st.selectbox("Filter Suspicious Status", ["All Reviews", "Potentially Suspicious Only", "Normal Only"])
    with filter_f3:
        search_query = st.text_input("🔍 Search within reviews", "")

    table_df = df_filtered.copy()
    if filter_sent:
        table_df = table_df[table_df['Sentiment'].isin(filter_sent)]
    if filter_susp == "Potentially Suspicious Only":
        table_df = table_df[table_df['Is_Suspicious'] == True]
    elif filter_susp == "Normal Only":
        table_df = table_df[table_df['Is_Suspicious'] == False]
    if search_query:
        table_df = table_df[table_df['Review Text'].str.contains(search_query, case=False, na=False)]

    display_cols = [
        col for col in [
            'Review ID', 'Customer Name', 'Rating', 'Sentiment',
            'Sentiment_Score', 'Aspects', 'Suspicious_Indicator',
            'Suspicion_Reasons', 'Review Text'
        ] if col in table_df.columns
    ]

    st.dataframe(
        table_df[display_cols],
        use_container_width=True,
        height=380
    )


# ==============================================================================
# PAGE 2: 📁 UPLOAD & PREPROCESS REVIEWS
# ==============================================================================
elif page == "📁 Upload Reviews":
    st.title("📁 Upload & Preprocess Customer Reviews")
    st.markdown("Upload your custom CSV dataset or reset to the benchmark dataset.")

    upload_tab1, upload_tab2 = st.tabs(["📤 Upload Custom CSV", "🔄 Load Benchmark Dataset"])

    with upload_tab1:
        uploaded_file = st.file_uploader(
            "Select CSV file containing customer reviews",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Detected {len(user_df)} rows and {len(user_df.columns)} columns.")

                st.markdown("#### ⚙️ Column Mapping")
                st.markdown("Map your dataset's columns to the standard review attributes:")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                all_cols = list(user_df.columns)
                with col_m1:
                    text_choice = st.selectbox("Review Text Column*", all_cols, index=0 if len(all_cols) > 0 else 0)
                with col_m2:
                    rating_choice = st.selectbox("Star Rating Column*", all_cols, index=min(1, len(all_cols) - 1))
                with col_m3:
                    date_choice = st.selectbox("Date Column (Optional)", ["None"] + all_cols)
                with col_m4:
                    prod_choice = st.selectbox("Product / Brand Column (Optional)", ["None"] + all_cols)

                if st.button("🚀 Process & Preprocess Dataset", type="primary"):
                    with st.spinner("Executing Data Preprocessing, Tokenization, Lemmatization, and Sentiment pipelines..."):
                        # Format mapping
                        df_to_proc = user_df.copy()
                        df_to_proc['Review Text'] = df_to_proc[text_choice]
                        df_to_proc['Rating'] = pd.to_numeric(df_to_proc[rating_choice], errors='coerce').fillna(3.0)
                        if date_choice != "None":
                            df_to_proc['Date'] = df_to_proc[date_choice]
                        if prod_choice != "None":
                            df_to_proc['Product/Business Name'] = df_to_proc[prod_choice]
                        else:
                            df_to_proc['Product/Business Name'] = "Custom Product / Business"

                        processed = process_dataset(df_to_proc, text_col='Review Text', rating_col='Rating', date_col='Date')
                        st.session_state['df_raw'] = user_df
                        st.session_state['df_processed'] = processed
                        st.success(f"Processing Complete! {len(processed)} clean reviews ready for Data Mining analysis.")
                        st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with upload_tab2:
        st.markdown("#### Pre-loaded Multi-Product Benchmark Dataset")
        st.markdown("""
        Includes **200+ realistic reviews** across 4 diverse industries:
        - 🎧 **NovaSound Pro Headphones** (Consumer Electronics)
        - ☁️ **CloudSync Cloud Storage** (B2B / SaaS)
        - 🍔 **SwiftEats Express Delivery** (Food & Logistics)
        - ⌚ **ApexFit Pulse Smartwatch** (Wearable Tech)
        - 🤖 *Synthetic Anomaly Injections* (Spam bots, duplicate reviews, rating contradictions)
        """)

        if st.button("🔄 Reset to Benchmark Dataset"):
            bench_df = load_default_dataset()
            st.session_state['df_raw'] = bench_df
            st.session_state['df_processed'] = process_dataset(
                bench_df, text_col='Review Text', rating_col='Rating', date_col='Date'
            )
            st.success("Successfully restored default benchmark dataset!")
            st.rerun()

    # Preprocessing Summary
    st.markdown("<div class='section-header'>🧹 Data Preprocessing & Cleaning Audit</div>", unsafe_allow_html=True)
    summary = st.session_state.get('prep_summary', {})

    if summary:
        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            st.metric("Raw Rows Submitted", summary.get("initial_rows", 0))
        with ps2:
            st.metric("Clean Reviews Retained", summary.get("final_rows", 0))
        with ps3:
            st.metric("Duplicates Purged", summary.get("duplicates_removed", 0))
        with ps4:
            st.metric("Vocabulary Reduction", f"{summary.get('vocab_reduction_pct', 0)}%")

        st.markdown("""
        <div class="callout-box">
            <strong>Preprocessing Pipeline Steps Applied:</strong><br>
            1. <strong>Deduplication</strong>: Exact duplicate review text identification and purge.<br>
            2. <strong>Missing Value Imputation</strong>: Handled missing ratings with median and dropped empty text rows.<br>
            3. <strong>Normalization</strong>: Lowercased, stripped HTML tags, URLs, and punctuation.<br>
            4. <strong>Tokenization & Lemmatization</strong>: Applied WordNet lemmatizer and NLTK stopwords reduction.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: 🔍 ANALYZE SINGLE REVIEW
# ==============================================================================
elif page == "🔍 Analyze Single Review":
    st.title("🔍 Analyze Individual Customer Review")
    st.markdown("Test the real-time Data Mining & Text Mining pipeline on custom text.")

    sample_prompts = [
        "Select a sample or enter your own...",
        "The active noise cancellation on these NovaSound headphones is phenomenal! Battery lasts over 35 hours.",
        "Terrible experience. Left ear cup stopped working after two weeks. Customer support refused to help.",
        "Best product ever loved it highly recommend everyone buy now excellent quality superb fast delivery!",
        "bad bad bad bad bad terrible terrible terrible awful awful awful terrible bad bad bad",
        "Horrible dreadful product. Broke immediately upon opening box. Complete waste of money.",
        "good"
    ]

    selected_sample = st.selectbox("Quick-fill sample review:", sample_prompts)
    default_text = "" if selected_sample == sample_prompts[0] else selected_sample

    review_input = st.text_area(
        "Enter Customer Review Text:",
        value=default_text,
        height=130,
        placeholder="Type or paste a customer review here to analyze..."
    )

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        rating_input = st.slider("Customer Star Rating (1 to 5)", 1, 5, 5)

    if st.button("🔬 Run Deep Text Mining & Anomaly Audit", type="primary"):
        if not review_input.strip():
            st.warning("Please enter review text to analyze.")
        else:
            preprocessor = get_preprocessor()
            sentiment_analyzer = get_sentiment_analyzer()
            anomaly_detector = get_anomaly_detector()
            topic_analyzer = get_topic_analyzer()

            # 1. Step by step preprocessing
            steps = preprocessor.get_step_by_step(review_input)

            # 2. Sentiment inference
            sent_res = sentiment_analyzer.analyze_review(review_input)

            # 3. Anomaly scan
            is_rep, rep_msg = anomaly_detector.check_lexical_repetition(review_input)
            is_len, len_msg = anomaly_detector.check_length_anomaly(len(review_input.split()), 25, 15)
            is_mis, mis_msg = anomaly_detector.check_rating_sentiment_mismatch(rating_input, sent_res['sentiment_score'])

            suspicion_reasons = []
            if is_rep:
                suspicion_reasons.append(rep_msg)
            if is_len:
                suspicion_reasons.append(len_msg)
            if is_mis:
                suspicion_reasons.append(mis_msg)

            is_suspicious = len(suspicion_reasons) > 0

            # 4. Aspect tags
            aspects = topic_analyzer.extract_review_aspect_tags(review_input)

            # --- Results Presentation ---
            st.markdown("<div class='section-header'>📊 Analysis Results</div>", unsafe_allow_html=True)
            res_c1, res_c2, res_c3 = st.columns(3)

            with res_c1:
                s_label = sent_res['sentiment']
                s_badge = "badge-pos" if s_label == "Positive" else ("badge-neg" if s_label == "Negative" else "badge-neu")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Detected Sentiment</div>
                    <div class="metric-value"><span class="{s_badge}">{s_label}</span></div>
                    <div class="metric-subtitle">Score: <strong>{sent_res['sentiment_score']}</strong> (range: -1.0 to +1.0)</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c2:
                susp_badge = "badge-suspicious" if is_suspicious else "badge-pos"
                susp_label = "Potentially Suspicious" if is_suspicious else "Normal"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Anomaly Status</div>
                    <div class="metric-value"><span class="{susp_badge}">{susp_label}</span></div>
                    <div class="metric-subtitle">{'Flagged by heuristic rules' if is_suspicious else 'Passes data mining checks'}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Aspect Categories</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: #38BDF8;">{aspects}</div>
                    <div class="metric-subtitle">Identified functional aspects</div>
                </div>
                """, unsafe_allow_html=True)

            if is_suspicious:
                st.error("⚠️ **Anomaly Reasons Detected:**\n- " + "\n- ".join(suspicion_reasons))

            # --- Preprocessing Pipeline Inspection ---
            st.markdown("<div class='section-header'>🔬 Step-by-Step Text Mining Transformation</div>", unsafe_allow_html=True)
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                st.markdown("**1. Raw Input Text:**")
                st.code(steps['Original Text'])
                st.markdown("**2. Cleaned & Lowercased:**")
                st.code(steps['Cleaned Lowercase'])
                st.markdown("**3. Tokenization:**")
                st.write(steps['Tokens'])

            with t_col2:
                st.markdown("**4. Stop Words Purged:**")
                st.write(steps['Stopwords Removed'])
                st.markdown("**5. Lemmatized Tokens:**")
                st.write(steps['Lemmatized Tokens'])
                st.markdown("**6. Final Normalized Model Feature:**")
                st.code(steps['Final Clean Text'])


# ==============================================================================
# PAGE 4: ⛏️ DATA MINING RESULTS
# ==============================================================================
elif page == "⛏️ Data Mining Results":
    st.title("⛏️ Data Mining & Pattern Discovery Deep Dive")
    st.markdown("Explore Supervised Classification, Unsupervised Clustering, Frequency Analysis, Association Rules, and Anomaly Detection.")

    if df_filtered.empty:
        st.warning("Please upload or load reviews first.")
        st.stop()

    dm_tab1, dm_tab2, dm_tab3, dm_tab4, dm_tab5 = st.tabs([
        "🤖 Supervised Classification",
        "🎯 Unsupervised Clustering & PCA",
        "📊 Frequency & N-Gram Mining",
        "🔗 Association Rule Mining",
        "🚨 Anomaly & Outlier Audit"
    ])

    # -------------------------------------------------------------
    # TAB 1: CLASSIFICATION
    # -------------------------------------------------------------
    with dm_tab1:
        st.markdown("### 🤖 Supervised Machine Learning Benchmark")
        st.markdown("Compares 4 classification algorithms trained on TF-IDF features with stratified 80/20 train/test split:")

        if 'classifier_pipeline' not in st.session_state:
            clf_pipe = ReviewClassifierPipeline()
            with st.spinner("Training classification models (Naive Bayes, Logistic Regression, Random Forest, SVM)..."):
                eval_res = clf_pipe.train_and_evaluate(
                    texts=df_filtered['Clean_Review'].tolist(),
                    labels=df_filtered['Sentiment'].tolist()
                )
                st.session_state['classifier_pipeline'] = clf_pipe
                st.session_state['classifier_eval'] = eval_res
        else:
            clf_pipe = st.session_state['classifier_pipeline']
            eval_res = st.session_state['classifier_eval']

        # Comparison Table
        summary_table = eval_res['summary_table']
        st.dataframe(summary_table, use_container_width=True)

        # Visual Comparison Bar Chart
        fig_clf = px.bar(
            summary_table, x="Algorithm", y=["Accuracy (%)", "F1-Score (%)", "Precision (%)", "Recall (%)"],
            barmode="group", title="Model Performance Metric Comparison",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_clf.update_layout(
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#E2E8F0"}
        )
        st.plotly_chart(fig_clf, use_container_width=True)

        # Confusion Matrix Inspector
        st.markdown("#### 🎯 Interactive Confusion Matrix")
        chosen_model = st.selectbox(
            "Select model to view Confusion Matrix:",
            list(clf_pipe.models.keys()),
            index=1
        )

        model_eval = eval_res['metrics'].get(chosen_model, {})
        if model_eval:
            cm = np.array(model_eval['confusion_matrix'])
            classes = model_eval['classes']

            fig_cm = px.imshow(
                cm, text_auto=True,
                x=classes, y=classes,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual True Class", color="Count")
            )
            fig_cm.update_layout(
                height=350, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)', font={'color': "#E2E8F0"}
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: CLUSTERING & PCA
    # -------------------------------------------------------------
    with dm_tab2:
        st.markdown("### 🎯 Unsupervised Review Clustering (K-Means)")
        st.markdown("Groups similar reviews into thematic clusters using TF-IDF representation and visualizes them in 2D using Principal Component Analysis (PCA).")

        cluster_col1, cluster_col2 = st.columns([1, 3])
        with cluster_col1:
            k_val = st.slider("Select Number of Clusters (k):", min_value=2, max_value=6, value=4)
            cluster_pipe = ReviewClusteringPipeline(n_clusters=k_val)
            df_clustered, cluster_meta = cluster_pipe.fit_and_cluster(
                df_filtered, text_col='Clean_Review', k=k_val
            )

            st.metric("Silhouette Score", cluster_meta['silhouette_score'])
            st.markdown("""
            <div style='font-size: 0.8rem; color: #94A3B8;'>
                Silhouette score measures cluster cohesion and separation (-1 to +1).
            </div>
            """, unsafe_allow_html=True)

        with cluster_col2:
            # Interactive 2D PCA Scatter
            fig_pca = px.scatter(
                df_clustered, x="PCA_1", y="PCA_2",
                color="Cluster_Theme",
                hover_data=["Rating", "Sentiment", "Review Text"],
                title="2D PCA Projection of Customer Review Clusters",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_pca.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"}
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        # Cluster Themes Cards
        st.markdown("#### 🏷️ Discovered Cluster Themes")
        c_cards = st.columns(k_val)
        for cid, info in cluster_meta['cluster_info'].items():
            if cid < len(c_cards):
                with c_cards[cid]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{info['theme_name']}</div>
                        <div class="metric-value" style="font-size: 1.4rem;">{info['size']} <span style="font-size: 0.9rem; color: #94A3B8;">({info['pct']}%)</span></div>
                        <div class="metric-subtitle">Top Terms: <em>{', '.join(info['top_keywords'][:4])}</em></div>
                    </div>
                    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: FREQUENCY & N-GRAMS
    # -------------------------------------------------------------
    with dm_tab3:
        st.markdown("### 📊 Frequency Analysis & N-Gram Mining")
        st.markdown("Identifies high-salience terms, bigrams, and trigrams across positive and negative feedback.")

        freq_miner = get_frequency_miner()

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            ngram_opt = st.selectbox("Select N-Gram Order:", ["Unigrams (Single Words)", "Bigrams (2-word phrases)", "Trigrams (3-word phrases)"])
            n_range = (1, 1) if "Unigrams" in ngram_opt else ((2, 2) if "Bigrams" in ngram_opt else (3, 3))

        with f_col2:
            sentiment_slice = st.selectbox("Filter by Sentiment Subset:", ["All Reviews", "Positive Only", "Negative Only"])

        if sentiment_slice == "Positive Only":
            texts_to_mine = df_filtered[df_filtered['Sentiment'] == 'Positive']['Clean_Review'].tolist()
        elif sentiment_slice == "Negative Only":
            texts_to_mine = df_filtered[df_filtered['Sentiment'] == 'Negative']['Clean_Review'].tolist()
        else:
            texts_to_mine = df_filtered['Clean_Review'].tolist()

        ngram_df = freq_miner.get_top_ngrams(texts_to_mine, ngram_range=n_range, top_n=15)

        if not ngram_df.empty:
            fig_freq = px.bar(
                ngram_df, x="Frequency", y="Ngram", orientation="h",
                text="Frequency", color="Frequency",
                color_continuous_scale="Viridis",
                title=f"Top 15 {ngram_opt} in {sentiment_slice}"
            )
            fig_freq.update_layout(
                height=420, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"}, yaxis={'autorange': 'reversed'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.info("No N-grams met the extraction criteria.")

    # -------------------------------------------------------------
    # TAB 4: ASSOCIATION RULE MINING
    # -------------------------------------------------------------
    with dm_tab4:
        st.markdown("### 🔗 Association Analysis & Co-occurrence Mining")
        st.markdown("Discovers relationship rules between functional product aspects and sentiment/rating outcomes: `Aspect` ➔ `Outcome`.")

        assoc_miner = get_association_miner()

        ar_c1, ar_c2 = st.columns(2)
        with ar_c1:
            min_sup = st.slider("Minimum Support Threshold (%)", min_value=1.0, max_value=25.0, value=4.0, step=1.0) / 100.0
        with ar_c2:
            min_conf = st.slider("Minimum Confidence Threshold (%)", min_value=10.0, max_value=90.0, value=25.0, step=5.0) / 100.0

        rules_df = assoc_miner.mine_rules(
            df_filtered, text_col='Review Text', min_support=min_sup, min_confidence=min_conf
        )

        if not rules_df.empty:
            st.dataframe(rules_df, use_container_width=True)

            # Scatter plot of Support vs Confidence vs Lift
            fig_rules = px.scatter(
                rules_df, x="Support (%)", y="Confidence (%)", size="Lift",
                color="Antecedent", hover_data=["Consequent", "Lift", "Co-occurrences"],
                title="Association Rules: Support vs Confidence (Bubble Size = Lift)"
            )
            fig_rules.update_layout(
                height=380, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#E2E8F0"}
            )
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.info("No association rules found at the selected thresholds. Try lowering Support or Confidence.")

    # -------------------------------------------------------------
    # TAB 5: ANOMALY DETECTION
    # -------------------------------------------------------------
    with dm_tab5:
        st.markdown("### 🚨 Anomaly & Suspicious Review Detection")
        st.markdown("Identifies anomalous, bot-generated, or inconsistent reviews using heuristic and statistical data mining rules.")

        anomaly_sum = st.session_state.get('anomaly_summary', {})
        ab = anomaly_sum.get('anomaly_breakdown', {})

        an1, an2, an3, an4 = st.columns(4)
        with an1:
            st.metric("Near-Duplicate Spam", ab.get('duplicate_similarity', 0))
        with an2:
            st.metric("Excessive Word Repetition", ab.get('lexical_repetition', 0))
        with an3:
            st.metric("Length Anomalies", ab.get('length_anomalies', 0))
        with an4:
            st.metric("Rating-Sentiment Mismatch", ab.get('rating_sentiment_mismatch', 0))

        st.markdown("#### 📋 Flagged Review Audit Log")
        flagged_df = df_filtered[df_filtered['Is_Suspicious'] == True]

        if not flagged_df.empty:
            st.dataframe(
                flagged_df[['Review ID', 'Customer Name', 'Rating', 'Sentiment', 'Suspicion_Score', 'Suspicion_Reasons', 'Review Text']],
                use_container_width=True,
                height=360
            )
        else:
            st.success("✅ No suspicious reviews detected in the current filter scope.")


# ==============================================================================
# PAGE 5: 📑 REPUTATION REPORT
# ==============================================================================
elif page == "📑 Reputation Report":
    st.title("📑 Comprehensive Executive Reputation Report")
    st.markdown("Generate and export a full audit report detailing data mining discoveries, risk indicators, and strategic recommendations.")

    if df_filtered.empty:
        st.warning("Please upload reviews first.")
        st.stop()

    rep_engine = get_reputation_engine()
    rep_results = rep_engine.calculate_reputation(
        df_filtered, rating_col='Rating', sentiment_col='Sentiment', date_col='Date', suspicious_col='Is_Suspicious'
    )

    clf_eval = st.session_state.get('classifier_eval', {})
    prep_sum = st.session_state.get('prep_summary', {})
    anom_sum = st.session_state.get('anomaly_summary', {})

    # Clustering meta
    cluster_pipe = ReviewClusteringPipeline(n_clusters=4)
    _, cluster_meta = cluster_pipe.fit_and_cluster(df_filtered, text_col='Clean_Review', k=4)

    # Association
    assoc_miner = get_association_miner()
    assoc_df = assoc_miner.mine_rules(df_filtered, min_support=0.03, min_confidence=0.25)

    # Frequency aspects
    freq_miner = get_frequency_miner()
    aspect_insights = freq_miner.analyze_sentiment_keywords(df_filtered)

    # Compile report text
    report_text = generate_executive_report(
        product_name=selected_product,
        reputation_data=rep_results,
        preprocessing_data=prep_sum,
        classification_data=clf_eval,
        clustering_data=cluster_meta,
        association_data=assoc_df,
        anomaly_data=anom_sum,
        top_aspects=aspect_insights
    )

    # Download Button
    st.download_button(
        label="📥 Download Executive Report (.md)",
        data=report_text,
        file_name=f"reputation_audit_report_{selected_product.replace(' ', '_').lower()}.md",
        mime="text/markdown"
    )

    # Display rendered markdown report
    st.markdown("---")
    st.markdown(report_text)


# ==============================================================================
# ==============================================================================
# PAGE 6: 📚 METHODOLOGY & ARCHITECTURE
# ==============================================================================
elif page == "📚 Methodology & Architecture":
    st.title("📚 Data Mining Methodology & Mathematical Foundations")
    st.markdown("Technical documentation explaining the algorithms, formulas, and architecture implemented in this system.")

    st.markdown("<div class='section-header'>1. The Knowledge Discovery in Databases (KDD) Pipeline</div>", unsafe_allow_html=True)
    st.markdown("This application strictly implements the formal **Knowledge Discovery in Databases (KDD)** stages:")

    kdd_col1, kdd_col2 = st.columns(2)

    with kdd_col1:
        st.markdown("""
        #### 1. Data Cleaning & Preprocessing
        - **Missing Value Imputation**: Median imputation for missing star ratings; deletion of null text records.
        - **Deduplication**: Purging exact matching strings to eliminate bot spamming.
        - **Text Normalization**: Case folding, URL/HTML tag removal, punctuation stripping, and tokenization.
        - **Stopwords & Lemmatization**: Reduction of lexical noise using NLTK WordNetLemmatizer and Porter Stemmer.

        #### 2. Pattern Discovery & Frequency Mining
        - Extraction of Unigrams, Bigrams, and Trigrams across sentiment subsets.
        - Term Frequency - Inverse Document Frequency (TF-IDF) feature matrix creation.

        #### 3. Supervised Classification
        - Four algorithms trained and compared:
          - **Multinomial Naive Bayes (MNB)**: Probabilistic generative model with Laplace smoothing.
          - **Logistic Regression (LR)**: Linear discriminative model with L2 regularization.
          - **Random Forest (RF)**: Ensemble of decision trees mitigating overfitting.
          - **Support Vector Machine (LinearSVC)**: Maximum-margin hyper-plane separator.
        - Full evaluation via Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
        """)

    with kdd_col2:
        st.markdown("""
        #### 4. Unsupervised Clustering
        - **K-Means Clustering**: Partitioning review TF-IDF vectors into $k$ discrete thematic clusters by minimizing within-cluster sum of squares (WCSS).
        - **Principal Component Analysis (PCA)**: Linear dimensionality reduction projecting high-dimensional TF-IDF vectors into 2 principal orthogonal axes for scatter visualization.
        - **Silhouette Coefficient**: Evaluating cluster compactness and separation.

        #### 5. Association Rule Mining
        - Discovers relational rules of the form:
        """)
        st.latex(r"A \implies B \quad (\text{e.g., } \{\text{Aspect: Battery}\} \implies \{\text{Sentiment: Negative}\})")
        st.markdown("- **Support**: Probability of co-occurrence:")
        st.latex(r"\text{Support}(A \implies B) = P(A \cap B) = \frac{\text{count}(A \cup B)}{N}")
        st.markdown("- **Confidence**: Conditional probability of outcome:")
        st.latex(r"\text{Confidence}(A \implies B) = P(B \mid A) = \frac{\text{count}(A \cup B)}{\text{count}(A)}")
        st.markdown("- **Lift**: Rule strength relative to random chance:")
        st.latex(r"\text{Lift}(A \implies B) = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)}")

        st.markdown("""
        #### 6. Anomaly & Outlier Detection
        - **TF-IDF Cosine Similarity**: Flags near-duplicate review bursts (similarity $\ge 0.85$).
        - **Type-Token Ratio (TTR)**: Flags excessive word repetition ($TTR < 0.42$).
        - **Length Z-score Outliers**: Flags reviews $\le 3$ words or $> 3.0\sigma$ above mean length.
        - **Rating-Sentiment Discrepancy**: Flags 5-star ratings with caustic negative text or 1-star ratings with glowing praise.
        """)

    st.markdown("<div class='section-header'>2. Reputation Score Formulation (R)</div>", unsafe_allow_html=True)
    st.markdown("The reputation score maps brand health to a unified index from **0 to 100**:")

    st.latex(r"R = \max\left(0, \min\left(100, \sum_{i=1}^{5} w_i \cdot S_i - P_{\text{suspicious}}\right)\right)")

    st.markdown("#### Mathematical Component Breakdown:")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**1. Average Rating Score ($S_1$, weight: 35%)**")
        st.latex(r"S_1 = \left(\frac{\mu_{\text{rating}}}{5.0}\right) \times 100")

        st.markdown("**2. Net Sentiment Distribution ($S_2$, weight: 30%)**")
        st.latex(r"S_2 = \frac{\%Pos - \%Neg + 100}{2} \in [0, 100]")

        st.markdown("**3. Positive-to-Negative Ratio ($S_3$, weight: 15%)**")
        st.latex(r"S_3 = \frac{\text{Count}(Pos)}{\text{Count}(Pos) + \text{Count}(Neg)} \times 100")

    with rc2:
        st.markdown("**4. Volume Reliability Factor ($S_4$, weight: 10%)**")
        st.latex(r"S_4 = \min\left(1.0, \frac{\log_{10}(N+1)}{\log_{10}(50)}\right) \times 100")

        st.markdown("**5. Recent Trend Momentum ($S_5$, weight: 10%)**")
        st.markdown("Time-decayed rating average comparing the most recent 30% of reviews against historical baseline.")

        st.markdown("**6. Suspicious Activity Penalty ($P_{\\text{suspicious}}$)**")
        st.latex(r"P_{\text{suspicious}} = \min(15.0, \text{Flagged}\% \times 0.35)")

    st.markdown("#### Letter Grade & Verdict System:")
    grade_c1, grade_c2, grade_c3 = st.columns(3)
    with grade_c1:
        st.markdown("""
        - `90 - 100`: **A+** (Stellar Reputation)
        - `80 - 89`: **A** (Strong Brand Reputation)
        """)
    with grade_c2:
        st.markdown("""
        - `70 - 79`: **B** (Good Reputation)
        - `60 - 69`: **C** (Moderate / Fair Reputation)
        """)
    with grade_c3:
        st.markdown("""
        - `45 - 59`: **D** (At-Risk Brand Reputation)
        - `< 45`: **F** (Critical / Damaged Reputation)
        """)
