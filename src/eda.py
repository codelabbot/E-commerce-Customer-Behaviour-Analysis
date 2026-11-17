"""Exploratory Data Analysis (EDA) plotting and analytical modules.

Generates publication-quality visualizations for revenue trends, product rankings,
geographic distributions, order distributions, and category keyword extraction.
Saves all generated figures to reports/figures/.
"""

import os
import pandas as pd
import numpy as np

# Ensure matplotlib uses local writable directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True


def plot_monthly_trend(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot monthly revenue and order count trends."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "01_monthly_revenue_order_trend.png")

    monthly = df.groupby("Month").agg(
        Revenue=("Revenue", "sum"),
        Order_Count=("CustomerID", "count")
    ).reset_index()

    month_names = {1: "Jan (Winter)", 4: "Apr (Spring)", 7: "Jul (Summer)", 10: "Oct (Fall)"}
    monthly["Month_Label"] = monthly["Month"].map(lambda m: month_names.get(m, f"M{m}"))

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2 = ax1.twinx()

    color1, color2 = "#2b5c8f", "#d95f02"
    bars = ax1.bar(monthly["Month_Label"], monthly["Revenue"], color=color1, alpha=0.75, width=0.45, label="Revenue ($)")
    lines = ax2.plot(monthly["Month_Label"], monthly["Order_Count"], color=color2, marker="o", linewidth=2.5, markersize=8, label="Order Count")

    ax1.set_title("Monthly Revenue & Order Volume Trends", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Season / Month", fontsize=11, labelpad=10)
    ax1.set_ylabel("Total Revenue ($)", fontsize=11, color=color1)
    ax2.set_ylabel("Total Orders", fontsize=11, color=color2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax2.grid(False)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"${h:,.0f}", (bar.get_x() + bar.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_day_and_hour_patterns(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot transaction distribution by Day of Week and Hour of Day."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "02_day_and_hour_patterns.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Day of Week
    dow_counts = df["DayOfWeek"].value_counts().reindex(days_order)
    sns.barplot(x=dow_counts.index, y=dow_counts.values, hue=dow_counts.index, legend=False, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Order Volume by Day of Week", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Order Count", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # Hour of Day
    hour_counts = df["Hour"].value_counts().sort_index()
    sns.lineplot(x=hour_counts.index, y=hour_counts.values, ax=axes[1], marker="s", color="#e7298a", linewidth=2.5)
    axes[1].fill_between(hour_counts.index, hour_counts.values, color="#e7298a", alpha=0.15)
    axes[1].set_title("Customer Transaction Velocity by Hour of Day", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (24-Hour Clock)", fontsize=11)
    axes[1].set_ylabel("Order Count", fontsize=11)
    axes[1].set_xticks(hour_counts.index)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_top_products(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot Top 10 products by revenue and by order quantity."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "03_top_products_revenue_quantity.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Top 10 by Revenue
    top_rev = df.groupby("Item_Purchased")["Revenue"].sum().nlargest(10).sort_values(ascending=True)
    axes[0].barh(top_rev.index, top_rev.values, color="#1b9e77", alpha=0.85)
    axes[0].set_title("Top 10 Products by Total Revenue", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Total Sales Revenue ($)", fontsize=11)
    axes[0].xaxis.set_major_formatter("${x:,.0f}")
    for i, v in enumerate(top_rev.values):
        axes[0].text(v + 100, i, f"${v:,.0f}", va="center", fontsize=9)

    # Top 10 by Quantity / Order Volume
    top_qty = df.groupby("Item_Purchased")["Quantity"].sum().nlargest(10).sort_values(ascending=True)
    axes[1].barh(top_qty.index, top_qty.values, color="#7570b3", alpha=0.85)
    axes[1].set_title("Top 10 Products by Order Frequency / Quantity", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Total Units Ordered", fontsize=11)
    for i, v in enumerate(top_qty.values):
        axes[1].text(v + 1, i, f"{int(v):,} orders", va="center", fontsize=9)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_geographic_distribution(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot revenue across top geographic markets and comparison."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "04_revenue_by_location.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Top 10 US States by Revenue
    top_states = df.groupby("Location")["Revenue"].sum().nlargest(10).sort_values(ascending=True)
    axes[0].barh(top_states.index, top_states.values, color="#2b5c8f", alpha=0.85)
    axes[0].set_title("Top 10 Geographic Locations by Revenue ($)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Revenue ($)", fontsize=11)
    axes[0].xaxis.set_major_formatter("${x:,.0f}")
    for i, v in enumerate(top_states.values):
        axes[0].text(v + 50, i, f"${v:,.0f}", va="center", fontsize=9)

    # Geographic Market Share Concentration
    state_totals = df.groupby("Location")["Revenue"].sum().sort_values(ascending=False)
    top_5_share = state_totals.head(5).sum()
    rest_share = state_totals.iloc[5:].sum()
    share_series = pd.Series([top_5_share, rest_share], index=["Top 5 States", "Remaining 45 States"])

    axes[1].pie(share_series, labels=share_series.index, autopct="%1.1f%%",
               colors=["#2b5c8f", "#9ecae1"], startangle=140, explode=(0.06, 0),
               wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Revenue Concentration: Top 5 States vs. Other 45 States", fontsize=13, fontweight="bold")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_order_value_and_basket_distributions(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Plot distribution of order values (revenue) and customer historical order volumes."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "05_order_value_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Order Value (Spend) Distribution
    sns.histplot(df["Revenue"], bins=20, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].axvline(df["Revenue"].mean(), color="red", linestyle="--", linewidth=1.8,
                    label=f"Mean: ${df['Revenue'].mean():.2f}")
    axes[0].axvline(df["Revenue"].median(), color="green", linestyle=":", linewidth=2,
                    label=f"Median: ${df['Revenue'].median():.2f}")
    axes[0].set_title("Distribution of Order Spend Value ($)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Order Value ($ USD)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(fontsize=10)

    # Customer Lifetime Order Distribution (Previous Purchases)
    sns.histplot(df["Previous_Purchases"], bins=20, kde=True, ax=axes[1], color="#e7298a")
    axes[1].axvline(df["Previous_Purchases"].mean(), color="red", linestyle="--", linewidth=1.8,
                    label=f"Mean: {df['Previous_Purchases'].mean():.1f}")
    axes[1].axvline(df["Previous_Purchases"].median(), color="green", linestyle=":", linewidth=2,
                    label=f"Median: {df['Previous_Purchases'].median():.1f}")
    axes[1].set_title("Distribution of Customer Historical Purchase Counts", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Prior Purchases Completed", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend(fontsize=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_category_and_keyword_analysis(df: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Extract and plot top product categories and description keyword tokens."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "06_category_keyword_analysis.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # 1. Category Revenue Share
    cat_rev = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    cat_pct = (cat_rev / cat_rev.sum()) * 100
    sns.barplot(x=cat_rev.index, y=cat_rev.values, hue=cat_rev.index, legend=False, ax=axes[0], palette="Set2")
    axes[0].set_title("Revenue Contribution by Merchandise Category", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Total Revenue ($)", fontsize=11)
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    for p, pct in zip(axes[0].patches, cat_pct.values):
        axes[0].annotate(f"${p.get_height():,.0f}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # 2. Text Keyword Frequency (Derived from Item Descriptions)
    tokens = []
    for item in df["Item_Purchased"].dropna():
        words = str(item).lower().split()
        tokens.extend(words)

    stop_words = {"and", "the", "with", "of", "in", "a", "an", "for"}
    filtered_tokens = [w.capitalize() for w in tokens if w not in stop_words and len(w) > 2]
    token_series = pd.Series(filtered_tokens).value_counts().head(10).sort_values(ascending=True)

    axes[1].barh(token_series.index, token_series.values, color="#e6ab02", alpha=0.85)
    axes[1].set_title("Top 10 Merchandise Keywords (Derived from Text Descriptions)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Mention Frequency in Orders", fontsize=11)
    for i, v in enumerate(token_series.values):
        axes[1].text(v + 3, i, f"{v:,}", va="center", fontsize=9)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def run_all_eda(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """Run all EDA plotting routines and return the mapping of saved chart paths."""
    os.makedirs(output_dir, exist_ok=True)
    return {
        "monthly_trend": plot_monthly_trend(df, output_dir),
        "day_hour_patterns": plot_day_and_hour_patterns(df, output_dir),
        "top_products": plot_top_products(df, output_dir),
        "geographic_distribution": plot_geographic_distribution(df, output_dir),
        "order_distributions": plot_order_value_and_basket_distributions(df, output_dir),
        "category_keywords": plot_category_and_keyword_analysis(df, output_dir)
    }
