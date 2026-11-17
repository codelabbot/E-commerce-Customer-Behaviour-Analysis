"""Customer behavioral analytics and customer-level table generation.

Computes lifetime order metrics, purchase cadences, repeat purchase rates,
Pareto revenue concentration (80/20 rule), and customer growth trends.
"""

import os
import pandas as pd
import numpy as np

# Use local matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

CADENCE_DAYS_MAP = {
    "Weekly": 7,
    "Bi-Weekly": 14,
    "Fortnightly": 14,
    "Monthly": 30,
    "Every 3 Months": 90,
    "Quarterly": 90,
    "Annually": 365
}


def build_customer_table(
    df: pd.DataFrame,
    output_path: str = "data/processed/customer_table.csv"
) -> pd.DataFrame:
    """Construct customer-level behavioral summary table and save to disk.
    
    Required attributes:
    - total_orders
    - total_revenue
    - average_order_value
    - items_per_order
    - first_purchase_date
    - last_purchase_date
    - average_days_between_orders
    - active_months
    """
    df_work = df.copy()

    # 1. Map purchase frequency cadence to days
    if "Frequency_of_Purchases" in df_work.columns:
        df_work["average_days_between_orders"] = df_work["Frequency_of_Purchases"].map(CADENCE_DAYS_MAP).fillna(30).astype(int)
    else:
        df_work["average_days_between_orders"] = 30

    # 2. Total lifetime orders = historical previous purchases + current order
    if "Previous_Purchases" in df_work.columns:
        df_work["total_orders"] = df_work["Previous_Purchases"] + 1
    else:
        df_work["total_orders"] = 1

    # 3. Average Order Value (AOV) and Total Lifetime Revenue
    if "Purchase_Amount_USD" in df_work.columns:
        df_work["average_order_value"] = df_work["Purchase_Amount_USD"].round(2)
    elif "Revenue" in df_work.columns:
        df_work["average_order_value"] = df_work["Revenue"].round(2)
    else:
        df_work["average_order_value"] = 50.0

    df_work["total_revenue"] = (df_work["total_orders"] * df_work["average_order_value"]).round(2)

    # 4. Items per order
    df_work["items_per_order"] = df_work.get("Quantity", 1.0)

    # 5. Last and First purchase dates
    year = df_work.get("Year", 2023)
    month = df_work.get("Month", 6)
    df_work["last_purchase_date"] = pd.to_datetime(
        year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-15"
    )

    history_span_days = pd.to_timedelta((df_work["total_orders"] - 1) * df_work["average_days_between_orders"], unit="D")
    df_work["first_purchase_date"] = df_work["last_purchase_date"] - history_span_days
    df_work["active_months"] = ((df_work["last_purchase_date"] - df_work["first_purchase_date"]).dt.days / 30.4375).clip(lower=1).round().astype(int)

    # Additional contextual attributes
    month_names = {1: "Jan (Winter)", 4: "Apr (Spring)", 7: "Jul (Summer)", 10: "Oct (Fall)"}
    df_work["Month_Label"] = df_work["last_purchase_date"].dt.month.map(month_names).fillna("Other")
    df_work["Cohort_Month"] = df_work["Month_Label"]
    df_work["Loyalty_Type"] = np.where(df_work["total_orders"] >= 10, "Returning Established (>=10 orders)", "New / Early-Stage (<10 orders)")

    # Format dates as strings YYYY-MM-DD
    df_work["first_purchase_date"] = df_work["first_purchase_date"].dt.strftime("%Y-%m-%d")
    df_work["last_purchase_date"] = df_work["last_purchase_date"].dt.strftime("%Y-%m-%d")

    # Select columns
    base_cols = [
        "CustomerID",
        "total_orders",
        "total_revenue",
        "average_order_value",
        "items_per_order",
        "first_purchase_date",
        "last_purchase_date",
        "average_days_between_orders",
        "active_months"
    ]
    context_cols = [c for c in ["Age", "Gender", "Category", "Location", "Review_Rating", "Subscription_Status", "Frequency_of_Purchases", "Month_Label", "Cohort_Month", "Loyalty_Type"] if c in df_work.columns]
    customer_table = df_work[base_cols + context_cols].copy()

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer_table.to_csv(output_path, index=False)
    print(f"Customer table saved successfully to: {output_path} ({len(customer_table):,} rows)")

    return customer_table


def plot_purchase_frequency(df_cust: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot purchase frequency distribution and repeat-purchase metrics."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "07_purchase_frequency_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Cadence frequency
    freq_order = ["Weekly", "Bi-Weekly", "Fortnightly", "Monthly", "Every 3 Months", "Quarterly", "Annually"]
    freq_counts = df_cust["Frequency_of_Purchases"].value_counts().reindex(freq_order)
    sns.barplot(x=freq_counts.index, y=freq_counts.values, hue=freq_counts.index, legend=False, ax=axes[0], palette="viridis")
    axes[0].set_title("Customer Distribution by Stated Purchase Cadence", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Customer Count", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # 2. Lifetime Order Volume Tiers
    bins = [1, 5, 15, 30, 50, 100]
    labels = ["2-5 Orders (Newer)", "6-15 Orders (Growing)", "16-30 Orders (Frequent)", "31-50 Orders (Loyal)", "50+ Orders (Champions)"]
    tier_counts = pd.cut(df_cust["total_orders"], bins=bins, labels=labels).value_counts().sort_index()

    sns.barplot(x=tier_counts.index, y=tier_counts.values, hue=tier_counts.index, legend=False, ax=axes[1], palette="crest")
    axes[1].set_title("Customer Breakdown by Total Lifetime Order Volume", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Customer Count", fontsize=11)
    axes[1].tick_params(axis="x", rotation=25)
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height())}\n({p.get_height()/len(df_cust)*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_new_vs_returning_monthly(df_cust: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Analyze new vs returning customers per monthly transaction cohort."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "08_new_vs_returning_monthly.png")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if "Loyalty_Type" not in df_cust.columns:
        df_cust["Loyalty_Type"] = np.where(df_cust["total_orders"] >= 10, "Returning Established (>=10 orders)", "New / Early-Stage (<10 orders)")

    if "Month_Label" not in df_cust.columns:
        month_names = {1: "Jan (Winter)", 4: "Apr (Spring)", 7: "Jul (Summer)", 10: "Oct (Fall)"}
        df_cust["Month_Label"] = df_cust["last_purchase_date"].str[5:7].astype(int).map(month_names).fillna("Other")

    cohort = pd.crosstab(df_cust["Month_Label"], df_cust["Loyalty_Type"], normalize="index") * 100

    cohort.plot(kind="bar", stacked=True, ax=ax, color=["#3182bd", "#9ecae1"], edgecolor="white", width=0.5)
    ax.set_title("Monthly Active Customer Composition: Returning vs. Early-Stage Shoppers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Transaction Cohort", fontsize=11)
    ax.set_ylabel("Customer Share (%)", fontsize=11)
    ax.legend(title="Customer Cohort", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)

    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", label_type="center", fontsize=9.5, color="white", weight="bold")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_spending_and_pareto(df_cust: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot spending distribution and Pareto 80/20 cumulative revenue concentration curve."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "09_pareto_revenue_concentration.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Total Lifetime Revenue Distribution
    sns.histplot(df_cust["total_revenue"], bins=25, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].axvline(df_cust["total_revenue"].mean(), color="red", linestyle="--", linewidth=1.8,
                    label=f"Mean: ${df_cust['total_revenue'].mean():,.0f}")
    axes[0].axvline(df_cust["total_revenue"].median(), color="green", linestyle=":", linewidth=2,
                    label=f"Median: ${df_cust['total_revenue'].median():,.0f}")
    axes[0].set_title("Distribution of Customer Lifetime Spend ($)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Total Customer Revenue ($)", fontsize=11)
    axes[0].set_ylabel("Number of Customers", fontsize=11)
    axes[0].xaxis.set_major_formatter("${x:,.0f}")
    axes[0].legend()

    # 2. Pareto 80/20 Concentration Curve
    sorted_rev = df_cust["total_revenue"].sort_values(ascending=False).values
    cum_rev = np.cumsum(sorted_rev) / np.sum(sorted_rev) * 100
    cum_cust = np.linspace(0, 100, len(sorted_rev))

    # Exact Top 20% point
    top20_idx = int(len(sorted_rev) * 0.20)
    top20_rev_share = cum_rev[top20_idx]

    axes[1].plot(cum_cust, cum_rev, color="#d95f02", linewidth=2.5, label="Actual Revenue Curve")
    axes[1].plot([0, 100], [0, 100], color="gray", linestyle="--", alpha=0.7, label="Equal Distribution Line")

    # Mark 20% point
    axes[1].scatter([20], [top20_rev_share], color="red", s=80, zorder=5)
    axes[1].axvline(20, color="red", linestyle=":", alpha=0.6)
    axes[1].axhline(top20_rev_share, color="red", linestyle=":", alpha=0.6)
    axes[1].annotate(f"Top 20% Customers generate\n{top20_rev_share:.1f}% of Revenue",
                     (20, top20_rev_share), xytext=(35, top20_rev_share - 15),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="orange", lw=1))

    axes[1].set_title("Pareto Analysis: Cumulative Customer vs. Revenue Share", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Cumulative % of Customers", fontsize=11)
    axes[1].set_ylabel("Cumulative % of Total Revenue", fontsize=11)
    axes[1].legend(loc="lower right")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_monthly_customer_growth(df_cust: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot monthly active and cumulative customer acquisition growth."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "10_monthly_customer_growth.png")

    if "Cohort_Month" not in df_cust.columns:
        month_names = {1: "Jan (Winter)", 4: "Apr (Spring)", 7: "Jul (Summer)", 10: "Oct (Fall)"}
        df_cust["Cohort_Month"] = df_cust["last_purchase_date"].str[5:7].astype(int).map(month_names).fillna("Other")

    growth = df_cust.groupby("Cohort_Month").agg(
        Active_Customers=("CustomerID", "count"),
        Revenue=("total_revenue", "sum")
    ).reindex(["Jan (Winter)", "Apr (Spring)", "Jul (Summer)", "Oct (Fall)"]).reset_index()

    growth["Cumulative_Customers"] = growth["Active_Customers"].cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2 = ax1.twinx()

    color1, color2 = "#2b5c8f", "#1b9e77"
    bars = ax1.bar(growth["Cohort_Month"], growth["Active_Customers"], color=color1, alpha=0.7, width=0.45, label="Active Customers in Cohort")
    lines = ax2.plot(growth["Cohort_Month"], growth["Cumulative_Customers"], color=color2, marker="o", linewidth=2.5, markersize=8, label="Cumulative Base")

    ax1.set_title("Customer Activity and Cumulative Base Growth Over Time", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Seasonal Milestone", fontsize=11)
    ax1.set_ylabel("Active Customers", fontsize=11, color=color1)
    ax2.set_ylabel("Cumulative Unique Customer Base", fontsize=11, color=color2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}", (bar.get_x() + bar.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    for i, txt in enumerate(growth["Cumulative_Customers"]):
        ax2.annotate(f"{txt:,}", (growth["Cohort_Month"][i], txt),
                     ha="center", va="bottom", fontsize=9, color=color2, weight="bold",
                     xytext=(0, 5), textcoords="offset points")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def run_all_customer_analytics(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """Generate customer-level table and all four behavioral visualizations."""
    cust_table = build_customer_table(df)
    return {
        "customer_table": cust_table,
        "fig_frequency": plot_purchase_frequency(cust_table, output_dir),
        "fig_new_vs_returning": plot_new_vs_returning_monthly(cust_table, output_dir),
        "fig_pareto": plot_spending_and_pareto(cust_table, output_dir),
        "fig_growth": plot_monthly_customer_growth(cust_table, output_dir)
    }
