"""E-Commerce Customer Analytics, Segmentation & Churn Prediction Dashboard.

Companion Streamlit Application reading strictly from data/processed/ and models/.
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
# PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Customer Intelligence & Churn AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #4A5568 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #1F4E79 !important;
    }
    /* Headers & Dividers */
    h1, h2, h3 {
        color: #1F4E79;
        font-family: 'Segoe UI', sans-serif;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 12px;
        margin-right: 6px;
    }
    .badge-high { background-color: #FADBD8; color: #78281F; border: 1px solid #E6B0AA; }
    .badge-medium { background-color: #FCF3CF; color: #7D6608; border: 1px solid #F9E79F; }
    .badge-low { background-color: #D4EFDF; color: #1E8449; border: 1px solid #A9DFBF; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
# SIDEBAR FILTERS & NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=64)
st.sidebar.title("E-Commerce Intelligence")
st.sidebar.caption("B.Tech Internship Analytics Suite")

navigation = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Overview KPIs",
        "🎯 Customer Segments (RFM + K-Means)",
        "⚠️ Churn Risk (Searchable Table)",
        "🔍 Customer Lookup by ID"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

# Filter 1: Season
all_seasons = sorted(clean_transactions["Season"].unique().tolist())
selected_seasons = st.sidebar.multiselect("Season / Date Cohort", options=all_seasons, default=[])

# Filter 2: Location
all_locations = sorted(master_customers["Location"].unique().tolist())
selected_locations = st.sidebar.multiselect("US State / Location", options=all_locations, default=[])

# Filter 3: RFM Segment
all_rfm = sorted(master_customers["RFM_Segment"].dropna().unique().tolist())
selected_rfm = st.sidebar.multiselect("RFM Segment", options=all_rfm, default=[])

# Filter 4: Churn Risk Tier
all_risks = ["High Risk (Churning)", "Medium Risk", "Low Risk (Active)"]
selected_risks = st.sidebar.multiselect("Churn Risk Tier", options=all_risks, default=[])

# Apply filters dynamically
filtered_cust = master_customers.copy()
filtered_tx = clean_transactions.copy()

if selected_seasons:
    filtered_tx = filtered_tx[filtered_tx["Season"].isin(selected_seasons)]
    # Filter customers whose latest purchase was in the selected season
    season_month_map = {"Winter": 1, "Spring": 4, "Summer": 7, "Fall": 10}
    target_months = [season_month_map[s] for s in selected_seasons if s in season_month_map]
    filtered_cust = filtered_cust[filtered_cust["last_purchase_date"].str[5:7].astype(int).isin(target_months)]

if selected_locations:
    filtered_cust = filtered_cust[filtered_cust["Location"].isin(selected_locations)]
    filtered_tx = filtered_tx[filtered_tx["Location"].isin(selected_locations)]

if selected_rfm:
    filtered_cust = filtered_cust[filtered_cust["RFM_Segment"].isin(selected_rfm)]
    filtered_tx = filtered_tx[filtered_tx["CustomerID"].isin(filtered_cust["CustomerID"])]

if selected_risks:
    filtered_cust = filtered_cust[filtered_cust["Risk_Level"].isin(selected_risks)]
    filtered_tx = filtered_tx[filtered_tx["CustomerID"].isin(filtered_cust["CustomerID"])]

st.sidebar.info(f"Showing **{len(filtered_cust):,}** customers ({len(filtered_tx):,} transactions)")


# ==============================================================================
# PAGE 1: OVERVIEW KPIS
# ==============================================================================
if navigation == "📊 Overview KPIs":
    st.title("Executive Sales & Transaction Overview")
    st.markdown("Commercial performance metrics, category distributions, and multi-channel order volume.")

    # Top KPI Banner
    total_rev = filtered_tx["Revenue"].sum()
    total_orders = len(filtered_tx)
    total_cust = filtered_cust["CustomerID"].nunique()
    aov = total_rev / total_orders if total_orders > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_rev:,.0f}", help="Cumulative sales volume in scope")
    col2.metric("Total Orders", f"{total_orders:,}", help="Distinct purchase orders completed")
    col3.metric("Active Customers", f"{total_cust:,}", help="Distinct active buyers in scope")
    col4.metric("Average Order Value (AOV)", f"${aov:,.2f}", help="Average financial spend per transaction")

    st.markdown("---")

    # Row 1: Monthly Sales Trend & Category Share
    r1_col1, r1_col2 = st.columns([6, 4])

    with r1_col1:
        st.subheader("Seasonal Revenue & Order Trend")
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        trend_df = filtered_tx.groupby("Season").agg(
            Revenue=("Revenue", "sum"),
            Orders=("Revenue", "count")
        ).reindex(season_order).dropna().reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=trend_df["Season"],
            y=trend_df["Revenue"],
            name="Revenue ($)",
            marker_color="#1F4E79"
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_df["Season"],
            y=trend_df["Orders"],
            name="Order Count",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#E67E22", width=3)
        ))
        fig_trend.update_layout(
            yaxis=dict(title="Revenue ($)", showgrid=True),
            yaxis2=dict(title="Order Count", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with r1_col2:
        st.subheader("Revenue by Department")
        cat_df = filtered_tx.groupby("Category")["Revenue"].sum().reset_index()
        fig_cat = px.pie(
            cat_df,
            values="Revenue",
            names="Category",
            color_discrete_sequence=["#1F4E79", "#2B5C8F", "#5DADE2", "#A9DFBF"],
            hole=0.45
        )
        fig_cat.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Row 2: Top Selling Products & Payment/Shipping Matrix
    r2_col1, r2_col2 = st.columns([5, 5])

    with r2_col1:
        st.subheader("Top 10 Selling Products by Revenue")
        top_prod = filtered_tx.groupby("Item_Purchased")["Revenue"].sum().nlargest(10).reset_index()
        top_prod = top_prod.sort_values("Revenue", ascending=True)
        fig_prod = px.bar(
            top_prod,
            x="Revenue",
            y="Item_Purchased",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Blues"
        )
        fig_prod.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with r2_col2:
        st.subheader("Fulfillment Speed by Payment Method")
        pay_ship = filtered_tx.groupby(["Payment_Method", "Shipping_Type"]).size().reset_index(name="Orders")
        fig_ps = px.bar(
            pay_ship,
            x="Payment_Method",
            y="Orders",
            color="Shipping_Type",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_ps.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360)
        st.plotly_chart(fig_ps, use_container_width=True)


# ==============================================================================
# PAGE 2: CUSTOMER SEGMENTS (RFM + K-MEANS)
# ==============================================================================
elif navigation == "🎯 Customer Segments (RFM + K-Means)":
    st.title("Customer Segmentation Intelligence")
    st.markdown("Comparative analysis of heuristic RFM segmentation and unsupervised K-Means clusters.")

    # Top Metrics
    total_lifetime_rev = filtered_cust["total_revenue"].sum()
    avg_clv = filtered_cust["total_revenue"].mean() if len(filtered_cust) > 0 else 0
    champions_cnt = (filtered_cust["RFM_Segment"] == "Champions").sum()
    hva_cnt = (filtered_cust["Cluster_Label"] == "High-Value Active").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Lifetime Value", f"${total_lifetime_rev:,.0f}")
    col2.metric("Average Customer CLV", f"${avg_clv:,.2f}")
    col3.metric("Champions (RFM)", f"{champions_cnt:,}")
    col4.metric("High-Value Active (K-Means)", f"{hva_cnt:,}")

    st.markdown("---")

    tab1, tab2 = st.tabs(["🏷️ RFM Heuristic Segmentation", "🤖 K-Means Clustering"])

    with tab1:
        st.subheader("RFM Segment Composition & Spending Distribution")
        rfm_summary = filtered_cust.groupby("RFM_Segment").agg(
            Customers=("CustomerID", "count"),
            Total_Spend=("total_revenue", "sum"),
            Avg_AOV=("average_order_value", "mean"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean")
        ).reset_index()

        c1, c2 = st.columns([5, 5])
        with c1:
            fig_rfm_bar = px.bar(
                rfm_summary,
                x="RFM_Segment",
                y="Total_Spend",
                color="RFM_Segment",
                text_auto=".2s",
                title="Lifetime Spend by RFM Segment",
                color_discrete_sequence=["#1b9e77", "#2b5c8f", "#7570b3", "#e7298a", "#d95f02"]
            )
            fig_rfm_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_rfm_bar, use_container_width=True)

        with c2:
            fig_rfm_scat = px.scatter(
                filtered_cust,
                x="Recency",
                y="Frequency",
                size="total_revenue",
                color="RFM_Segment",
                hover_data=["CustomerID", "total_revenue"],
                title="Recency vs. Frequency (Bubble Size = Lifetime Spend)",
                color_discrete_sequence=["#1b9e77", "#2b5c8f", "#7570b3", "#e7298a", "#d95f02"]
            )
            fig_rfm_scat.update_layout(height=350)
            st.plotly_chart(fig_rfm_scat, use_container_width=True)

        st.dataframe(rfm_summary.style.format({
            "Customers": "{:,}",
            "Total_Spend": "${:,.0f}",
            "Avg_AOV": "${:,.2f}",
            "Avg_Recency": "{:.1f} days",
            "Avg_Frequency": "{:.1f} orders"
        }), use_container_width=True)

    with tab2:
        st.subheader("K-Means Cluster Architecture ($k=4$)")
        km_summary = filtered_cust.groupby("Cluster_Label").agg(
            Customers=("CustomerID", "count"),
            Revenue=("total_revenue", "sum"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean")
        ).reset_index()

        k1, k2 = st.columns([5, 5])
        with k1:
            fig_km = px.bar(
                km_summary,
                x="Cluster_Label",
                y="Customers",
                color="Cluster_Label",
                text_auto=True,
                title="Customer Distribution across K-Means Clusters",
                color_discrete_sequence=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
            )
            fig_km.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_km, use_container_width=True)

        with k2:
            fig_km_rev = px.pie(
                km_summary,
                names="Cluster_Label",
                values="Revenue",
                title="Revenue Contribution by Cluster",
                hole=0.4,
                color_discrete_sequence=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
            )
            fig_km_rev.update_layout(height=350)
            st.plotly_chart(fig_km_rev, use_container_width=True)

        st.dataframe(km_summary.style.format({
            "Customers": "{:,}",
            "Revenue": "${:,.0f}",
            "Mean_Recency": "{:.1f} days",
            "Mean_Frequency": "{:.1f} orders",
            "Mean_Monetary": "${:,.2f}"
        }), use_container_width=True)


# ==============================================================================
# PAGE 3: CHURN RISK (SEARCHABLE CUSTOMER TABLE)
# ==============================================================================
elif navigation == "⚠️ Churn Risk (Searchable Table)":
    st.title("Customer Churn Risk & Retention Intelligence")
    st.markdown("Machine learning churn forecasts with actionable operational risk classification.")

    high_risk_cnt = (filtered_cust["Risk_Level"] == "High Risk (Churning)").sum()
    med_risk_cnt = (filtered_cust["Risk_Level"] == "Medium Risk").sum()
    low_risk_cnt = (filtered_cust["Risk_Level"] == "Low Risk (Active)").sum()
    total_active = len(filtered_cust)
    churn_rate = (high_risk_cnt / total_active) * 100 if total_active > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk Accounts", f"{high_risk_cnt:,}", f"{churn_rate:.1f}% Churn Rate")
    col2.metric("Medium Risk (Opportunity)", f"{med_risk_cnt:,}", "Prime Rescue Cohort")
    col3.metric("Low Risk (Active)", f"{low_risk_cnt:,}", "Loyal Active Buyers")
    col4.metric("Avg Churn Probability", f"{filtered_cust['Churn_Probability'].mean():.2%}")

    st.markdown("---")

    # Priority Action Alert
    st.info(f"💡 **Immediate Commercial Priority:** There are **{med_risk_cnt:,} customers** in the **Medium Risk** tier. These buyers are transitioning into inactivity but have not fully lapsed. Prioritize these accounts for targeted, low-cost re-engagement nudges.")

    # Search and Filter Toolbar
    c_search, c_tier, c_prob = st.columns([4, 4, 4])
    with c_search:
        search_query = st.text_input("🔍 Search Customer ID or State", placeholder="e.g., 1042, California")
    with c_tier:
        table_tiers = st.multiselect("Filter Risk Tier", options=all_risks, default=all_risks)
    with c_prob:
        prob_range = st.slider("Churn Probability Threshold", min_value=0.0, max_value=1.0, value=(0.0, 1.0))

    # Apply Search & Table Filters
    table_df = filtered_cust[
        (filtered_cust["Risk_Level"].isin(table_tiers)) &
        (filtered_cust["Churn_Probability"] >= prob_range[0]) &
        (filtered_cust["Churn_Probability"] <= prob_range[1])
    ]

    if search_query:
        q = search_query.strip().lower()
        table_df = table_df[
            table_df["CustomerID"].astype(str).str.contains(q) |
            table_df["Location"].str.lower().str.contains(q)
        ]

    # Display Columns
    view_cols = [
        "CustomerID",
        "Age",
        "Gender",
        "Location",
        "RFM_Segment",
        "Cluster_Label",
        "Recency",
        "Frequency",
        "total_revenue",
        "Churn_Probability",
        "Risk_Level"
    ]
    renamed_view = table_df[view_cols].rename(columns={
        "total_revenue": "Lifetime_Spend",
        "Recency": "Days_Dormant",
        "Frequency": "Lifetime_Orders"
    })

    st.subheader(f"Search Results ({len(renamed_view):,} accounts matching filters)")
    st.dataframe(
        renamed_view.style.format({
            "Lifetime_Spend": "${:,.0f}",
            "Churn_Probability": "{:.2%}"
        }),
        use_container_width=True,
        height=400
    )

    # Download Button
    csv_bytes = renamed_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Cohort to CSV",
        data=csv_bytes,
        file_name="churn_risk_campaign_target.csv",
        mime="text/csv"
    )


# ==============================================================================
# PAGE 4: CUSTOMER LOOKUP BY ID & CHURN SIMULATOR
# ==============================================================================
elif navigation == "🔍 Customer Lookup by ID":
    st.title("Customer 360 Profile & Real-Time Churn Predictor")
    st.markdown("Detailed individual account diagnosis and dynamic 'What-If' churn simulation using the trained Machine Learning pipeline.")

    lookup_id = st.number_input(
        "Enter Customer ID (1 to 3,900)",
        min_value=1,
        max_value=3900,
        value=1,
        step=1
    )

    cust_records = master_customers[master_customers["CustomerID"] == lookup_id]

    if len(cust_records) == 0:
        st.error(f"Customer #{lookup_id} not found.")
    else:
        c_row = cust_records.iloc[0]

        # Profile Header Banner
        risk_class = "badge-high" if "High" in c_row["Risk_Level"] else ("badge-medium" if "Medium" in c_row["Risk_Level"] else "badge-low")
        st.markdown(f"""
        ### Customer #{c_row['CustomerID']} &mdash; Profile Summary
        <span class="badge-pill {risk_class}">{c_row['Risk_Level']}</span>
        <span class="badge-pill badge-medium">Segment: {c_row['RFM_Segment']}</span>
        <span class="badge-pill badge-low">Cluster: {c_row['Cluster_Label']}</span>
        <br><br>
        """, unsafe_allow_html=True)

        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recency", f"{c_row['Recency']} Days", f"R-Score: {c_row['R_Score']}/5")
        m2.metric("Frequency", f"{c_row['Frequency']} Orders", f"F-Score: {c_row['F_Score']}/5")
        m3.metric("Lifetime Spend", f"${c_row['total_revenue']:,.2f}", f"M-Score: {c_row['M_Score']}/5")
        m4.metric("Churn Risk", f"{c_row['Churn_Probability']:.1%}", c_row['Risk_Level'])

        st.markdown("---")

        # Two-Column Account Details & Dynamic Simulator
        col_details, col_sim = st.columns([5, 5])

        with col_details:
            st.subheader("Account Diagnostics")
            details_dict = {
                "Age & Gender": f"{c_row['Age']} years ({c_row['Gender']})",
                "Territory / Location": f"{c_row['Location']}, United States",
                "Subscription Status": f"{c_row['Subscription_Status']}",
                "Satisfaction Rating": f"{c_row['Review_Rating']} / 5.0 ★",
                "Stated Shopping Cadence": f"{c_row['Frequency_of_Purchases']}",
                "Expected Inter-Purchase Gap": f"{c_row['average_days_between_orders']} days",
                "First Purchase Date": f"{c_row['first_purchase_date']}",
                "Last Purchase Date": f"{c_row['last_purchase_date']}",
                "Average Basket Size (AOV)": f"${c_row['average_order_value']:.2f}"
            }
            st.table(pd.DataFrame(list(details_dict.items()), columns=["Attribute", "Value"]))

            # Prescription Box
            if "High" in c_row["Risk_Level"]:
                st.error("🚨 **Prescription:** High dormancy detected. Immediate automated win-back campaign with a 20% discount code or personalized reactivation email recommended.")
            elif "Medium" in c_row["Risk_Level"]:
                st.warning("⚠️ **Prescription:** Approaching critical lapse threshold. Send cadence-based push reminder or recommendations based on preferred category.")
            else:
                st.success("✅ **Prescription:** Healthy active buyer. Cross-sell new arrivals and reward with loyalty bonus points without discounting margins.")

        with col_sim:
            st.subheader("Dynamic 'What-If' Churn Simulator")
            st.caption("Test how re-engagement actions alter the customer's live predicted churn probability.")

            sim_recency = st.slider(
                "Simulated Recency (Days since last purchase)",
                min_value=1,
                max_value=300,
                value=int(c_row["Recency"])
            )
            sim_sub = st.selectbox(
                "Loyalty Subscription Status",
                options=["Yes", "No"],
                index=0 if c_row["Subscription_Status"] == "Yes" else 1
            )
            sim_discount = st.selectbox(
                "Promotional Discount Provided",
                options=["Yes", "No"],
                index=0
            )

            # Build feature vector for ML pipeline
            cadence_map = {
                "Weekly": 7, "Bi-Weekly": 14, "Fortnightly": 14,
                "Monthly": 30, "Every 3 Months": 90, "Quarterly": 90, "Annually": 365
            }
            gap_days = cadence_map.get(c_row["Frequency_of_Purchases"], 30)

            sim_vector = pd.DataFrame([{
                "Recency": sim_recency,
                "Frequency": c_row["Frequency"],
                "AOV": c_row["average_order_value"],
                "Monetary": c_row["Monetary"],
                "Tenure_Days": int(c_row["active_months"] * 30.4375),
                "Avg_Gap_Days": gap_days,
                "Review_Rating": c_row["Review_Rating"],
                "Age": c_row["Age"],
                "Is_Subscriber": 1 if sim_sub == "Yes" else 0,
                "Discount_Applied": 1 if sim_discount == "Yes" else 0,
                "Gender_Male": 1 if c_row["Gender"] == "Male" else 0
            }])

            # Execute real-time inference using the loaded model pipeline
            best_pipe = churn_model_pkg["pipeline"]
            sim_prob = best_pipe.predict_proba(sim_vector)[0, 1]

            st.markdown("#### Simulation Outcome:")
            delta_prob = (sim_prob - c_row["Churn_Probability"]) * 100
            st.metric(
                label="Simulated Churn Probability",
                value=f"{sim_prob:.1%}",
                delta=f"{delta_prob:+.1f}% vs Original",
                delta_color="inverse"
            )

            if sim_prob < 0.30:
                st.success("Resulting Status: **Low Risk (Active)**")
            elif sim_prob <= 0.70:
                st.warning("Resulting Status: **Medium Risk**")
            else:
                st.error("Resulting Status: **High Risk (Churning)**")
