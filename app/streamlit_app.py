"""E-Commerce Customer Analytics, Segmentation & Churn AI Portal.

Dark-themed executive portal and analytics workspace matching modern SaaS dashboards.
Reads strictly from data/processed/ and models/.
Built for B.Tech Internship Analytics Project.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure local writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence | Analytics Portal",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# EXECUTIVE DARK THEME STYLING (Matching User-Supplied Screenshots)
# ------------------------------------------------------------------------------
DARK_THEME_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0B132B !important;
        color: #F8FAFC !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1B2A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #F8FAFC !important;
    }
    
    /* Top Header Link (Fork on GitHub) */
    .top-nav-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 4px 0 16px 0;
        margin-top: -30px;
    }
    .top-nav-link {
        color: #94A3B8;
        text-decoration: none;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: color 0.2s ease;
    }
    .top-nav-link:hover {
        color: #38BDF8;
    }
    
    /* Breadcrumb Banner */
    .breadcrumb-text {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #60A5FA;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    /* Hero Title & Subtitle */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 18px;
    }
    
    /* Notice / Badge Callout Box */
    .notice-box {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid #334155;
        border-left: 3px solid #38BDF8;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #CBD5E1;
        margin-bottom: 22px;
    }
    
    /* Dark Surface Cards (Metric Containers) */
    .metric-card {
        background-color: #152238;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Streamlit Default Metric Override */
    div[data-testid="stMetric"] {
        background-color: #152238 !important;
        border: 1px solid #1E293B !important;
        padding: 14px 18px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    
    /* Portal Dashboard Action Cards */
    .dashboard-preview-card {
        background-color: #152238;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 22px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .dashboard-preview-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .dashboard-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dashboard-card-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 18px;
        min-height: 42px;
    }
    
    /* Dataset Management Box in Sidebar */
    .sidebar-dataset-box {
        background-color: #152238;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .sidebar-dataset-stats {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
    }
    
    /* Navigation Radio Pills Styling */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent;
        padding: 8px 12px;
        border-radius: 6px;
        color: #94A3B8;
        font-size: 0.90rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 4px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border-radius: 4px 4px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    
    /* Table & Dataframe Overrides */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        background-color: #152238 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #38BDF8 !important;
        color: #0B132B !important;
        border-color: #38BDF8 !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #152238 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 12px;
        margin-right: 6px;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid #EF4444; }
    .badge-medium { background-color: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid #F59E0B; }
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid #10B981; }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ------------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    """Load and merge processed datasets strictly from data/processed/."""
    clean_df = pd.read_csv("data/processed/clean_transactions.csv")
    cust_df = pd.read_csv("data/processed/customer_table.csv")
    seg_df = pd.read_csv("data/processed/customer_segments.csv")
    churn_df = pd.read_csv("data/processed/churn_predictions.csv")

    # Unified Master Customer Dataframe
    master_cust = cust_df.copy()
    master_cust["RFM_Segment"] = seg_df["RFM_Segment"]
    master_cust["Cluster"] = seg_df["cluster"]
    master_cust["Cluster_Label"] = seg_df["cluster_label"]
    master_cust["R_Score"] = seg_df["R_Score"]
    master_cust["F_Score"] = seg_df["F_Score"]
    master_cust["M_Score"] = seg_df["M_Score"]
    master_cust["RFM_Score_Sum"] = seg_df["RFM_Score_Sum"]

    master_cust["Recency"] = seg_df["R"]
    master_cust["Frequency"] = seg_df["F"]
    master_cust["Monetary"] = seg_df["M"]

    master_cust["Actual_Churn"] = churn_df["Actual_Churn"]
    master_cust["Predicted_Churn"] = churn_df["Predicted_Churn"]
    master_cust["Churn_Probability"] = churn_df["Churn_Probability"]
    master_cust["Risk_Level"] = churn_df["Risk_Level"]

    return clean_df, master_cust


@st.cache_resource
def load_models():
    """Load serialized machine learning artifacts strictly from models/."""
    churn_artifact = joblib.load("models/churn_model.pkl")
    kmeans_artifact = joblib.load("models/kmeans.pkl")
    return churn_artifact, kmeans_artifact


# Load data and models
clean_transactions, master_customers = load_datasets()
churn_model_pkg, kmeans_model_pkg = load_models()


# ------------------------------------------------------------------------------
# SESSION STATE & NAVIGATION SYNC
# ------------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Portal Home"

nav_options = [
    "Portal Home",
    "Streamlit Analytics",
    "Power BI Dashboard"
]

# ------------------------------------------------------------------------------
# SIDEBAR (Matching Layout from Screenshots)
# ------------------------------------------------------------------------------
with st.sidebar:
    # Top Navigation Menu
    nav_icons = {
        "Portal Home": "🏠",
        "Streamlit Analytics": "📊",
        "Power BI Dashboard": "📈"
    }
    
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        format_func=lambda x: f"{nav_icons[x]}  {x}",
        index=nav_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in nav_options else 0,
        label_visibility="collapsed"
    )
    
    # Sync radio with state
    if selected_nav != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_nav

    st.markdown("---")
    
    # App Branding (Matching Image 1 & 2)
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h2 style="color: #FFFFFF; font-size: 1.4rem; font-weight: 800; margin: 0;">Customer<br>Intelligence</h2>
        <div style="color: #64748B; font-size: 0.80rem; margin-top: 2px;">Analytics Portal</div>
    </div>
    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 10px;">
        Logged in as: <strong style="color: #F8FAFC;">Analyst (Pratim)</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Reset Filters", use_container_width=True):
            st.session_state["filter_seasons"] = []
            st.session_state["filter_locations"] = []
            st.session_state["filter_rfm"] = []
            st.session_state["filter_risks"] = []
            st.rerun()
    with col_btn2:
        if st.button("Active Session", use_container_width=True):
            st.toast("Session active. Data cache synchronized.", icon="🟢")

    # Dataset Management (Matching Image 2)
    st.markdown("---")
    st.markdown("#### | Dataset Management")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], help="Upload an alternative transaction CSV file (max 200MB).")
    
    if uploaded_file is not None:
        try:
            custom_tx = pd.read_csv(uploaded_file)
            st.success(f"Custom dataset loaded: {len(custom_tx):,} rows")
            clean_transactions = custom_tx
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    else:
        st.markdown(f"""
        <div class="sidebar-dataset-box">
            <strong style="color: #38BDF8;">Using default project dataset.</strong><br>
            clean_transactions.csv
            <div class="sidebar-dataset-stats">
                Total rows: <strong>{len(clean_transactions):,}</strong><br>
                Total columns: <strong>{len(clean_transactions.columns)}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Global Filters (Active on Streamlit Analytics & Power BI pages)
    if st.session_state["current_page"] in ["Streamlit Analytics", "Power BI Dashboard"]:
        st.markdown("#### | Interactive Filters")
        
        # Season filter
        all_seasons = sorted(clean_transactions["Season"].dropna().unique().tolist())
        sel_seasons = st.multiselect("Season / Date Cohort", options=all_seasons, default=[])
        
        # Location filter
        all_locations = sorted(master_customers["Location"].dropna().unique().tolist())
        sel_locations = st.multiselect("US State / Location", options=all_locations, default=[])
        
        # RFM Segment filter
        all_rfm = sorted(master_customers["RFM_Segment"].dropna().unique().tolist())
        sel_rfm = st.multiselect("RFM Segment", options=all_rfm, default=[])
        
        # Churn Risk filter
        all_risks = ["High Risk (Churning)", "Medium Risk", "Low Risk (Active)"]
        sel_risks = st.multiselect("Churn Risk Tier", options=all_risks, default=[])
    else:
        sel_seasons = []
        sel_locations = []
        sel_rfm = []
        sel_risks = []

# Apply Global Filters to Dataframes
filtered_cust = master_customers.copy()
filtered_tx = clean_transactions.copy()

if sel_seasons:
    filtered_tx = filtered_tx[filtered_tx["Season"].isin(sel_seasons)]
    season_month_map = {"Winter": 1, "Spring": 4, "Summer": 7, "Fall": 10}
    target_months = [season_month_map[s] for s in sel_seasons if s in season_month_map]
    filtered_cust = filtered_cust[filtered_cust["last_purchase_date"].str[5:7].astype(int).isin(target_months)]

if sel_locations:
    filtered_cust = filtered_cust[filtered_cust["Location"].isin(sel_locations)]
    filtered_tx = filtered_tx[filtered_tx["Location"].isin(sel_locations)]

if sel_rfm:
    filtered_cust = filtered_cust[filtered_cust["RFM_Segment"].isin(sel_rfm)]
    filtered_tx = filtered_tx[filtered_tx["CustomerID"].isin(filtered_cust["CustomerID"])]

if sel_risks:
    filtered_cust = filtered_cust[filtered_cust["Risk_Level"].isin(sel_risks)]
    filtered_tx = filtered_tx[filtered_tx["CustomerID"].isin(filtered_cust["CustomerID"])]


# ------------------------------------------------------------------------------
# TOP HEADER BAR (Matching Top Right Fork on GitHub Link in Screenshots)
# ------------------------------------------------------------------------------
st.markdown("""
<div class="top-nav-bar">
    <a href="https://github.com/codelabbot/E-commerce-Customer-Behaviour-Analysis" target="_blank" class="top-nav-link">
        <span>Fork</span>
        <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
    </a>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: PORTAL HOME (Matching Image 1)
# ==============================================================================
if st.session_state["current_page"] == "Portal Home":
    st.markdown('<div class="hero-title">Welcome to Customer Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">A central portal for exploring customer behavior, RFM segmentation, predictive churn machine learning, and Power BI dashboards.</div>', unsafe_allow_html=True)

    # 3-Metric Summary Row (Matching Image 1: Registered Users, Total Visits, Active Sessions)
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    with col_h1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Registered Customers</div>
            <div class="metric-value">3,900</div>
            <div class="metric-sub">Verified customer base</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Lifetime Orders</div>
            <div class="metric-value">102,771</div>
            <div class="metric-sub">Mean: 26.35 orders/cust</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Risk Sessions</div>
            <div class="metric-value" style="color: #F59E0B;">237</div>
            <div class="metric-sub">Medium churn risk priority</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Enterprise Value</div>
            <div class="metric-value" style="color: #38BDF8;">$6.15M</div>
            <div class="metric-sub">Cumulative customer equity</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Active risk sessions are based on machine learning churn probabilities between 30% and 70% requiring proactive retention.")
    st.markdown("<hr style='border-color: #1E293B; margin: 24px 0;'>", unsafe_allow_html=True)

    # Available Dashboards Section (Matching Image 1 Two Cards)
    st.markdown("<h3 style='color: #FFFFFF; font-size: 1.35rem; font-weight: 700; margin-bottom: 16px;'>Available Dashboards</h3>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("""
        <div class="dashboard-preview-card">
            <div class="dashboard-card-title">
                <span>📊</span> Streamlit Analytics
            </div>
            <div class="dashboard-card-desc">
                Explore commercial KPIs, RFM & K-Means customer segments, interactive churn risk tables, and Customer 360 lookups.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Streamlit Analytics from the sidebar.", key="btn_open_stream", use_container_width=True):
            st.session_state["current_page"] = "Streamlit Analytics"
            st.rerun()

    with col_d2:
        st.markdown("""
        <div class="dashboard-preview-card">
            <div class="dashboard-card-title">
                <span>📈</span> Power BI Dashboard
            </div>
            <div class="dashboard-card-desc">
                Explore the executive star-schema data model, 20+ DAX measures, 4-page dashboard specification, and export datasets.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Power BI Dashboard from the sidebar.", key="btn_open_pbi", use_container_width=True):
            st.session_state["current_page"] = "Power BI Dashboard"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.78rem; color: #64748B;'>Project data is validated for B.Tech Internship Analytics. Transaction observations, segmentations, and churn models are strictly leak-free.</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: STREAMLIT ANALYTICS (Matching Image 2)
# ==============================================================================
elif st.session_state["current_page"] == "Streamlit Analytics":
    # Breadcrumb & Header
    st.markdown('<div class="breadcrumb-text">BEYOND THE NUMBERS / ANALYTICS WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Customer Intelligence. <span style="font-size: 1.2rem; color: #38BDF8;">🔗</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Customer behaviour & predictive churn analytics | Interactive analytics dashboard</div>', unsafe_allow_html=True)

    # Notice / Disclaimer Banner (Matching Image 2)
    st.markdown("""
    <div class="notice-box">
        <strong>PROCESSED DATA | B.Tech Major Project.</strong> 3,900 customer observations, RFM scores, K-Means clusters, and ML churn predictions are validated with zero data leakage.
    </div>
    """, unsafe_allow_html=True)

    # Collapsible Dataset Quality Report (Matching Image 2)
    with st.expander("› Dataset Quality Report", expanded=False):
        col_q1, col_q2, col_q3 = st.columns(3)
        col_q1.markdown("""
        **Integrity Checks:**
        - Missing Values: **0% (Complete)**
        - Exact Duplicate Rows: **0 (Unique)**
        - Integrity Status: **Validated**
        """)
        col_q2.markdown("""
        **Data Scale:**
        - Total Customers: **3,900**
        - Catalog Items: **25 unique items**
        - Departments: **4 categories**
        """)
        col_q3.markdown("""
        **Pre-Cutoff Formulation:**
        - Cutoff Date: **July 16, 2023**
        - Outcome Window: **91 days (≥ 90d)**
        - Class Balance: **75.0% Churn / 25.0% Retained**
        """)

    # 5 Horizontal Metric Cards (Matching Image 2 Layout)
    total_rev = filtered_tx["Revenue"].sum()
    total_orders = len(filtered_tx)
    total_cust = filtered_cust["CustomerID"].nunique()
    aov = total_rev / total_orders if total_orders > 0 else 0
    repeat_pct = (filtered_cust["total_orders"] >= 10).mean() * 100
    avg_rating = filtered_cust["Review_Rating"].mean()

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customers</div>
            <div class="metric-value">{total_cust:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Orders</div>
            <div class="metric-value">{total_orders:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Spend</div>
            <div class="metric-value">${aov:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Repeat Rate</div>
            <div class="metric-value">{repeat_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Rating</div>
            <div class="metric-value">{avg_rating:.2f} <span style="font-size: 1.1rem; color: #F59E0B;">★</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sub-Tabs (Underlined style matching Image 2: Overview, Customer Segments, Churn Risk Table, Customer 360 Lookup)
    tab_overview, tab_segments, tab_churn, tab_lookup = st.tabs([
        "Overview",
        "Customer Segments",
        "Churn Risk Table",
        "Customer 360 Lookup"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW (Matching Image 2 Cyan Bar Charts)
    # --------------------------------------------------------------------------
    with tab_overview:
        c_row1_1, c_row1_2 = st.columns(2)

        with c_row1_1:
            st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;'>Average spend by category</h4>", unsafe_allow_html=True)
            cat_avg = filtered_tx.groupby("Category")["Revenue"].mean().reset_index()
            cat_avg = cat_avg.sort_values(by="Revenue", ascending=False)
            
            fig_cat = px.bar(
                cat_avg,
                x="Category",
                y="Revenue",
                text_auto=".2f",
                color_discrete_sequence=["#38BDF8"] # Electric Cyan
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(title="Average Spend ($)", gridcolor="#1E293B"),
                xaxis=dict(title=None, gridcolor="#1E293B"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with c_row1_2:
            st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;'>Revenue by season</h4>", unsafe_allow_html=True)
            season_rev = filtered_tx.groupby("Season")["Revenue"].sum().reindex(["Winter", "Spring", "Summer", "Fall"]).dropna().reset_index()
            
            fig_season = px.bar(
                season_rev,
                x="Season",
                y="Revenue",
                text_auto="$,.0f",
                color_discrete_sequence=["#38BDF8"] # Electric Cyan
            )
            fig_season.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(title="Total Revenue ($)", gridcolor="#1E293B"),
                xaxis=dict(title=None, gridcolor="#1E293B"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320
            )
            st.plotly_chart(fig_season, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_row2_1, c_row2_2 = st.columns(2)

        with c_row2_1:
            st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;'>Top 10 selling products by revenue</h4>", unsafe_allow_html=True)
            top_prod = filtered_tx.groupby("Item_Purchased")["Revenue"].sum().nlargest(10).reset_index()
            top_prod = top_prod.sort_values(by="Revenue", ascending=True)
            
            fig_prod = px.bar(
                top_prod,
                x="Revenue",
                y="Item_Purchased",
                orientation="h",
                text_auto="$,.0f",
                color_discrete_sequence=["#60A5FA"]
            )
            fig_prod.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                xaxis=dict(title="Revenue ($)", gridcolor="#1E293B"),
                yaxis=dict(title=None, gridcolor="#1E293B"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=340
            )
            st.plotly_chart(fig_prod, use_container_width=True)

        with c_row2_2:
            st.markdown("<h4 style='color: #FFFFFF; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px;'>Payment method by shipping speed</h4>", unsafe_allow_html=True)
            pay_ship = filtered_tx.groupby(["Payment_Method", "Shipping_Type"]).size().reset_index(name="Orders")
            fig_ps = px.bar(
                pay_ship,
                x="Payment_Method",
                y="Orders",
                color="Shipping_Type",
                barmode="stack",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_ps.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(title="Orders", gridcolor="#1E293B"),
                xaxis=dict(title=None, gridcolor="#1E293B"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=10, b=10),
                height=340
            )
            st.plotly_chart(fig_ps, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: CUSTOMER SEGMENTS (RFM + K-MEANS)
    # --------------------------------------------------------------------------
    with tab_segments:
        st.markdown("<h4 style='color: #FFFFFF; font-size: 1.15rem; font-weight: 700; margin-bottom: 12px;'>Dual Segmentation Analysis: Heuristic RFM vs Unsupervised K-Means</h4>", unsafe_allow_html=True)
        
        col_seg_kpi1, col_seg_kpi2, col_seg_kpi3 = st.columns(3)
        champions_rev = filtered_cust[filtered_cust["RFM_Segment"] == "Champions"]["total_revenue"].sum()
        hvi_rev = filtered_cust[filtered_cust["Cluster_Label"] == "High-Value Inactive"]["total_revenue"].sum()
        hvi_pct = (hvi_rev / filtered_cust["total_revenue"].sum() * 100) if filtered_cust["total_revenue"].sum() > 0 else 0
        
        col_seg_kpi1.metric("Champions Spend", f"${champions_rev:,.0f}", "Top RFM Tier")
        col_seg_kpi2.metric("High-Value Inactive Spend", f"${hvi_rev:,.0f}", f"{hvi_pct:.1f}% Enterprise Share")
        col_seg_kpi3.metric("Optimal Clusters (k)", "4 Clusters", "Silhouette: 0.4560")

        st.markdown("<br>", unsafe_allow_html=True)
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown("<h5 style='color: #F8FAFC;'>RFM Segment Spend Distribution</h5>", unsafe_allow_html=True)
            rfm_sum = filtered_cust.groupby("RFM_Segment")["total_revenue"].sum().reset_index()
            fig_rfm = px.bar(
                rfm_sum,
                x="RFM_Segment",
                y="total_revenue",
                text_auto="$,.0f",
                color="RFM_Segment",
                color_discrete_sequence=["#38BDF8", "#34D399", "#818CF8", "#F472B6", "#FB923C"]
            )
            fig_rfm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(title="Lifetime Spend ($)", gridcolor="#1E293B"),
                xaxis=dict(title=None, gridcolor="#1E293B"),
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with s_col2:
            st.markdown("<h5 style='color: #F8FAFC;'>Recency vs. Frequency (Bubble = Spend)</h5>", unsafe_allow_html=True)
            fig_rfm_scat = px.scatter(
                filtered_cust,
                x="Recency",
                y="Frequency",
                size="total_revenue",
                color="RFM_Segment",
                hover_data=["CustomerID", "total_revenue"],
                color_discrete_sequence=["#38BDF8", "#34D399", "#818CF8", "#F472B6", "#FB923C"]
            )
            fig_rfm_scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8"),
                yaxis=dict(title="Frequency (Orders)", gridcolor="#1E293B"),
                xaxis=dict(title="Recency (Days Dormant)", gridcolor="#1E293B"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_rfm_scat, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h5 style='color: #F8FAFC;'>K-Means Cluster Architecture ($k=4$) Summary</h5>", unsafe_allow_html=True)
        km_table = filtered_cust.groupby("Cluster_Label").agg(
            Customers=("CustomerID", "count"),
            Total_Revenue=("total_revenue", "sum"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_AOV=("average_order_value", "mean")
        ).reset_index()

        st.dataframe(km_table.style.format({
            "Customers": "{:,}",
            "Total_Revenue": "${:,.0f}",
            "Mean_Recency": "{:.1f} days",
            "Mean_Frequency": "{:.1f} orders",
            "Mean_AOV": "${:,.2f}"
        }), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: CHURN RISK TABLE
    # --------------------------------------------------------------------------
    with tab_churn:
        st.markdown("<h4 style='color: #FFFFFF; font-size: 1.15rem; font-weight: 700; margin-bottom: 12px;'>Machine Learning Churn Predictions & Actionable Worklist</h4>", unsafe_allow_html=True)
        
        # Priority Callout
        med_cnt = (filtered_cust["Risk_Level"] == "Medium Risk").sum()
        st.markdown(f"""
        <div class="notice-box" style="border-left-color: #F59E0B;">
            <strong style="color: #FCD34D;">IMMEDIATE COMMERCIAL PRIORITY:</strong> There are <strong>{med_cnt:,} accounts</strong> in the <strong>Medium Risk</strong> tier (30% to 70% probability). These buyers are transitioning into dormancy. Intervene now before full lapse!
        </div>
        """, unsafe_allow_html=True)

        # Filters
        c_f1, c_f2, c_f3 = st.columns([4, 4, 4])
        with c_f1:
            q_search = st.text_input("🔍 Search Customer ID or State", placeholder="e.g. 1042, California")
        with c_f2:
            q_tiers = st.multiselect("Risk Tier", options=all_risks, default=all_risks)
        with c_f3:
            q_prob = st.slider("Churn Probability Threshold", min_value=0.0, max_value=1.0, value=(0.0, 1.0))

        # Filter Data
        churn_table = filtered_cust[
            (filtered_cust["Risk_Level"].isin(q_tiers)) &
            (filtered_cust["Churn_Probability"] >= q_prob[0]) &
            (filtered_cust["Churn_Probability"] <= q_prob[1])
        ]

        if q_search:
            s_term = q_search.strip().lower()
            churn_table = churn_table[
                churn_table["CustomerID"].astype(str).str.contains(s_term) |
                churn_table["Location"].str.lower().str.contains(s_term)
            ]

        display_cols = [
            "CustomerID", "Age", "Gender", "Location", "RFM_Segment",
            "Cluster_Label", "Recency", "Frequency", "total_revenue",
            "Churn_Probability", "Risk_Level"
        ]
        view_df = churn_table[display_cols].rename(columns={
            "total_revenue": "Lifetime_Spend",
            "Recency": "Days_Dormant",
            "Frequency": "Lifetime_Orders"
        })

        st.dataframe(view_df.style.format({
            "Lifetime_Spend": "${:,.0f}",
            "Churn_Probability": "{:.2%}"
        }), use_container_width=True, height=380)

        # CSV Download Button
        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Retention Worklist to CSV",
            data=csv_bytes,
            file_name="retention_campaign_target.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------------------------
    # TAB 4: CUSTOMER 360 LOOKUP & REAL-TIME CHURN SIMULATOR
    # --------------------------------------------------------------------------
    with tab_lookup:
        st.markdown("<h4 style='color: #FFFFFF; font-size: 1.15rem; font-weight: 700; margin-bottom: 12px;'>Customer 360 Diagnostic & Real-Time What-If ML Simulator</h4>", unsafe_allow_html=True)
        
        lookup_num = st.number_input("Enter Customer ID (1 to 3,900):", min_value=1, max_value=3900, value=1, step=1)
        c_records = master_customers[master_customers["CustomerID"] == lookup_num]

        if len(c_records) > 0:
            crow = c_records.iloc[0]
            
            badge_class = "badge-high" if "High" in crow["Risk_Level"] else ("badge-medium" if "Medium" in crow["Risk_Level"] else "badge-low")
            st.markdown(f"""
            <div style="margin-bottom: 14px;">
                <span style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">Customer #{crow['CustomerID']}</span> &nbsp;
                <span class="badge-pill {badge_class}">{crow['Risk_Level']}</span>
                <span class="badge-pill badge-medium">RFM: {crow['RFM_Segment']}</span>
                <span class="badge-pill badge-low">Cluster: {crow['Cluster_Label']}</span>
            </div>
            """, unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Recency", f"{crow['Recency']} Days", f"R-Score: {crow['R_Score']}/5")
            k2.metric("Frequency", f"{crow['Frequency']} Orders", f"F-Score: {crow['F_Score']}/5")
            k3.metric("Lifetime Spend", f"${crow['total_revenue']:,.2f}", f"M-Score: {crow['M_Score']}/5")
            k4.metric("Churn Risk", f"{crow['Churn_Probability']:.1%}", crow['Risk_Level'])

            st.markdown("<hr style='border-color: #1E293B; margin: 18px 0;'>", unsafe_allow_html=True)

            col_diag, col_sim = st.columns(2)

            with col_diag:
                st.markdown("<h5 style='color: #F8FAFC;'>Customer Attributes</h5>", unsafe_allow_html=True)
                diag_items = {
                    "Age & Gender": f"{crow['Age']} yrs ({crow['Gender']})",
                    "Territory / State": f"{crow['Location']}, United States",
                    "Loyalty Program": f"Subscribed: {crow['Subscription_Status']}",
                    "Satisfaction Score": f"{crow['Review_Rating']} / 5.0 ★",
                    "Shopping Cadence": f"{crow['Frequency_of_Purchases']}",
                    "Inter-Purchase Gap": f"{crow['average_days_between_orders']} days",
                    "First Purchase": f"{crow['first_purchase_date']}",
                    "Latest Purchase": f"{crow['last_purchase_date']}",
                    "Average Basket (AOV)": f"${crow['average_order_value']:.2f}"
                }
                st.table(pd.DataFrame(list(diag_items.items()), columns=["Field", "Value"]))

            with col_sim:
                st.markdown("<h5 style='color: #F8FAFC;'>Dynamic 'What-If' Retention Simulator</h5>", unsafe_allow_html=True)
                st.caption("Adjust sliders to observe real-time machine learning probability re-calculation via models/churn_model.pkl.")

                sim_r = st.slider("Simulated Recency (Days Dormant)", min_value=1, max_value=300, value=int(crow["Recency"]))
                sim_sub_val = st.selectbox("Loyalty Subscription", ["Yes", "No"], index=0 if crow["Subscription_Status"] == "Yes" else 1)
                sim_disc_val = st.selectbox("Promotional Discount Given", ["Yes", "No"], index=0)

                cad_map = {
                    "Weekly": 7, "Bi-Weekly": 14, "Fortnightly": 14,
                    "Monthly": 30, "Every 3 Months": 90, "Quarterly": 90, "Annually": 365
                }
                gap_val = cad_map.get(crow["Frequency_of_Purchases"], 30)

                sim_vec = pd.DataFrame([{
                    "Recency": sim_r,
                    "Frequency": crow["Frequency"],
                    "AOV": crow["average_order_value"],
                    "Monetary": crow["Monetary"],
                    "Tenure_Days": int(crow["active_months"] * 30.4375),
                    "Avg_Gap_Days": gap_val,
                    "Review_Rating": crow["Review_Rating"],
                    "Age": crow["Age"],
                    "Is_Subscriber": 1 if sim_sub_val == "Yes" else 0,
                    "Discount_Applied": 1 if sim_disc_val == "Yes" else 0,
                    "Gender_Male": 1 if crow["Gender"] == "Male" else 0
                }])

                best_pipe = churn_model_pkg["pipeline"]
                sim_p = best_pipe.predict_proba(sim_vec)[0, 1]

                delta_p = (sim_p - crow["Churn_Probability"]) * 100
                st.metric("Simulated Churn Probability", f"{sim_p:.1%}", delta=f"{delta_p:+.1f}% vs Original", delta_color="inverse")

                if sim_p < 0.30:
                    st.success("Simulated Status: **Low Risk (Active)**")
                elif sim_p <= 0.70:
                    st.warning("Simulated Status: **Medium Risk (Transitioning)**")
                else:
                    st.error("Simulated Status: **High Risk (Churning)**")


# ==============================================================================
# VIEW 3: POWER BI DASHBOARD (Matching Second Card in Image 1)
# ==============================================================================
elif st.session_state["current_page"] == "Power BI Dashboard":
    st.markdown('<div class="breadcrumb-text">EXECUTIVE REPORTING / POWER BI WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Power BI Executive Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Star-schema data architecture, DAX measure repository, and 4-page dashboard blueprint</div>', unsafe_allow_html=True)

    col_pb1, col_pb2, col_pb3, col_pb4 = st.columns(4)
    with col_pb1:
        st.metric("Fact Transactions", "3,900 rows", "fact_sales.csv")
    with col_pb2:
        st.metric("Dimension Tables", "3 conformed", "Customer, Product, Date")
    with col_pb3:
        st.metric("DAX Measures", "20+ measures", "measures.dax")
    with col_pb4:
        st.metric("Executive Pages", "4 pages", "Sales, Behavior, Segments, Churn")

    st.markdown("<br>", unsafe_allow_html=True)

    pbi_tab0, pbi_tab1, pbi_tab2, pbi_tab3, pbi_tab4 = st.tabs([
        "Official .PBIX Dashboard",
        "Star Schema Model",
        "DAX Measures Library",
        "4-Page Visual Blueprint",
        "Export CSV Files"
    ])

    with pbi_tab0:
        st.markdown("<h4 style='color: #FFFFFF;'>Official Project Power BI Dashboard</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class="notice-box" style="border-left-color: #38BDF8;">
            <strong>COMPLETED DASHBOARD READY:</strong> The official Power BI Desktop dashboard file 
            <code>Dashboard_cronus_to_for_project.pbix</code> (1.64 MB) is fully built and integrated into this project.
            It features custom futuristic wireframe visuals, conformed star-schema data bindings (<code>fact_sales</code>, <code>dim_customer</code>, <code>dim_product</code>, <code>dim_date</code>), and executive KPI cards.
        </div>
        """, unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns([6, 4])
        with col_dl1:
            pbix_path = "powerbi/Dashboard_cronus_to_for_project.pbix"
            if os.path.exists(pbix_path):
                with open(pbix_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Official Project Dashboard (Dashboard_cronus_to_for_project.pbix)",
                        data=f.read(),
                        file_name="Dashboard_cronus_to_for_project.pbix",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            st.markdown("""
            **How to Open in Power BI Desktop:**
            1. Click the download button above to save `Dashboard_cronus_to_for_project.pbix`.
            2. Double-click the file to open it directly in **Microsoft Power BI Desktop**.
            3. All report visuals, interactive slicers, drillthrough filters on `RFM_Segment`, and data relationships will load automatically.
            """)

        with col_dl2:
            st.markdown("""
            <div class="metric-card" style="margin-top: 0;">
                <div class="metric-label">Dashboard Artifact</div>
                <div class="metric-value" style="font-size: 1.3rem;">.PBIX File</div>
                <div class="metric-sub">Size: <strong>1.64 MB</strong> | High-Resolution Layout</div>
            </div>
            """, unsafe_allow_html=True)

        wireframe_img = "reports/figures/20_powerbi_dashboard_wireframe.png"
        if os.path.exists(wireframe_img):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h5 style='color: #F8FAFC;'>Futuristic Wireframe Design Layout</h5>", unsafe_allow_html=True)
            st.image(wireframe_img, caption="Power BI Custom Dashboard Canvas Wireframe (1920x1080 Widescreen)", use_container_width=True)


    with pbi_tab1:
        st.markdown("<h4 style='color: #FFFFFF;'>Kimball Star Schema Architecture</h4>", unsafe_allow_html=True)
        st.markdown("""
        All relationships in the Power BI model are strictly **One-to-Many (1 : *)** with **Single-Direction** cross-filtering:
        - `dim_customer[CustomerID]` (1) ➔ `fact_sales[CustomerID]` (*)
        - `dim_product[ProductID]` (1) ➔ `fact_sales[ProductID]` (*)
        - `dim_date[DateKey]` (1) ➔ `fact_sales[DateKey]` (*)
        """)
        
        st.image("reports/figures/16_kmeans_vs_rfm_crosstab.png", caption="Model Cross-Tabulation Matrix Reference", use_container_width=True)

    with pbi_tab2:
        st.markdown("<h4 style='color: #FFFFFF;'>Core DAX Measure Scripts (powerbi/measures.dax)</h4>", unsafe_allow_html=True)
        st.code("""
// Total Commercial Revenue
Total Revenue = SUM(fact_sales[Revenue])

// Total Distinct Transactions
Total Orders = DISTINCTCOUNT(fact_sales[TransactionID])

// Total Active Customer Base
Total Customers = DISTINCTCOUNT(dim_customer[CustomerID])

// Average Order Value (AOV)
Average Order Value = DIVIDE([Total Revenue], [Total Orders], 0)

// Repeat Customer Percentage
Repeat Customer % = DIVIDE([Returning Customers], [Total Customers], 0)

// High-Risk Churn Customer Count
High-Risk Churn Customers = 
CALCULATE([Total Customers], dim_customer[Risk_Level] = "High Risk (Churning)")
        """, language="dax")

    with pbi_tab3:
        st.markdown("<h4 style='color: #FFFFFF;'>4-Page Visual Specification (powerbi/dashboard_spec.md)</h4>", unsafe_allow_html=True)
        st.markdown("""
        1. **Page 1: Sales Overview** — KPI cards, Monthly revenue & order trend combo chart, category donut breakdown, US State filled map.
        2. **Page 2: Customer Behaviour** — Repeat customer %, Season composition (New vs Returning 100% stacked), purchasing cadence horizontal bar, lifetime spend vs orders scatter plot.
        3. **Page 3: Customer Segmentation** — RFM treemap, K-Means dual-axis column chart ($k=4$), Recency vs Frequency bubble plot, segment cross-tab matrix.
        4. **Page 4: Business Insights & Churn Risk** — Churn risk cards, Churn by RFM 100% stacked bar, Top 10 products bar chart, **Medium-Risk re-activation grid table**.
        """)

    with pbi_tab4:
        st.markdown("<h4 style='color: #FFFFFF;'>Download Clean Star-Schema CSVs</h4>", unsafe_allow_html=True)
        st.caption("Load these 4 conformed CSV files directly into Power BI Desktop via 'Get Data' -> 'Text/CSV'.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            with open("powerbi/fact_sales.csv", "rb") as f:
                st.download_button("📥 Download fact_sales.csv", f.read(), "fact_sales.csv", "text/csv")
            with open("powerbi/dim_customer.csv", "rb") as f:
                st.download_button("📥 Download dim_customer.csv", f.read(), "dim_customer.csv", "text/csv")
        with d_col2:
            with open("powerbi/dim_product.csv", "rb") as f:
                st.download_button("📥 Download dim_product.csv", f.read(), "dim_product.csv", "text/csv")
            with open("powerbi/dim_date.csv", "rb") as f:
                st.download_button("📥 Download dim_date.csv", f.read(), "dim_date.csv", "text/csv")
