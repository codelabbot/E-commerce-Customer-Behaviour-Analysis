"""RFM (Recency, Frequency, Monetary) segmentation engine.

Computes snapshot-relative recency, frequency, and monetary metrics,
applies tie-safe quantile scoring (1-5), assigns business customer personas,
and generates diagnostic heatmaps, treemaps, and segment distribution charts.
"""

import os
import pandas as pd
import numpy as np

# Ensure local writable cache for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

SEGMENT_RULES = [
    {
        "Segment": "Champions",
        "R_Score": "4 - 5",
        "F_Score": "4 - 5",
        "Description": "Bought recently, buy most often, and spend the most. Core brand advocates.",
        "Action_Strategy": "Reward with VIP perks, early product access, and brand ambassador programs."
    },
    {
        "Segment": "Loyal Customers",
        "R_Score": "3 - 5",
        "F_Score": "3 - 5",
        "Description": "Consistent repeat purchasers with high spending and steady purchasing cadences.",
        "Action_Strategy": "Upsell higher-value products, offer loyalty points, and seek user reviews."
    },
    {
        "Segment": "Potential Loyalists",
        "R_Score": "4 - 5",
        "F_Score": "1 - 3",
        "Description": "Recent buyers with moderate order frequency and growing lifetime spend.",
        "Action_Strategy": "Offer personalized onboarding recommendations and bundle discounts to drive repeat visits."
    },
    {
        "Segment": "Need Attention",
        "R_Score": "3",
        "F_Score": "1 - 2",
        "Description": "Above-average recency but lower historical frequency; at a critical retention crossroads.",
        "Action_Strategy": "Re-engage with limited-time promotional incentives and personalized product suggestions."
    },
    {
        "Segment": "About to Sleep",
        "R_Score": "1 - 2",
        "F_Score": "3",
        "Description": "Below-average recency with moderate past frequency; at risk of lapsing into dormancy.",
        "Action_Strategy": "Deliver aggressive reactivation discounts, win-back emails, and surveys to identify friction."
    },
    {
        "Segment": "At Risk",
        "R_Score": "1 - 2",
        "F_Score": "4 - 5",
        "Description": "High historical order volume and spend, but haven't purchased recently. High-priority churn risk.",
        "Action_Strategy": "High-priority direct outreach, personalized re-activation deals, and customer success contact."
    },
    {
        "Segment": "Hibernating / Lost",
        "R_Score": "1 - 2",
        "F_Score": "1 - 2",
        "Description": "Lowest recency and low frequency; dormant shoppers who have likely churned.",
        "Action_Strategy": "Automated low-cost email reactivation or exclude from paid remarketing to conserve ad spend."
    }
]


def compute_rfm_metrics(
    df: pd.DataFrame,
    date_col: str = "last_purchase_date",
    freq_col: str = "total_orders",
    monetary_col: str = "total_revenue"
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute Recency, Frequency, and Monetary metrics for each customer.
    
    Snapshot date = last invoice date + 1 day.
    """
    df_rfm = df.copy()

    # Determine snapshot date
    max_date = pd.to_datetime(df_rfm[date_col]).max()
    snapshot_date = max_date + pd.Timedelta(days=1)

    # Recency: days elapsed since last purchase
    df_rfm["Recency"] = (snapshot_date - pd.to_datetime(df_rfm[date_col])).dt.days
    df_rfm["Frequency"] = df_rfm[freq_col].astype(int)
    df_rfm["Monetary"] = df_rfm[monetary_col].astype(float)

    return df_rfm, snapshot_date


def score_rfm_quantiles(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """Score Recency, Frequency, and Monetary metrics 1-5 using quantiles.
    
    Uses rank(method='first') prior to qcut to safely handle ties in discrete values.
    - Recency: 5 = Most recent (lowest days), 1 = Least recent (highest days).
    - Frequency: 5 = Most frequent, 1 = Least frequent.
    - Monetary: 5 = Highest spend, 1 = Lowest spend.
    """
    df_scored = df_rfm.copy()

    # R Score: lower recency is better
    df_scored["R_Score"] = pd.qcut(
        df_scored["Recency"].rank(method="first", ascending=False),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # F Score: higher frequency is better
    df_scored["F_Score"] = pd.qcut(
        df_scored["Frequency"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # M Score: higher spend is better
    df_scored["M_Score"] = pd.qcut(
        df_scored["Monetary"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Composite RFM String and Numeric Score
    df_scored["RFM_Segment_Code"] = (
        df_scored["R_Score"].astype(str) +
        df_scored["F_Score"].astype(str) +
        df_scored["M_Score"].astype(str)
    )
    df_scored["RFM_Score_Sum"] = df_scored["R_Score"] + df_scored["F_Score"] + df_scored["M_Score"]

    return df_scored


def assign_rfm_segments(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Map scored customers into business segments based on documented rule table."""
    def segment_logic(row):
        r, f = row["R_Score"], row["F_Score"]
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 3:
            return "Potential Loyalists"
        elif r == 3 and f <= 2:
            return "Need Attention"
        elif r <= 2 and f >= 4:
            return "At Risk"
        elif r <= 2 and f == 3:
            return "About to Sleep"
        else:
            return "Hibernating / Lost"

    df_segmented = df_scored.copy()
    df_segmented["Segment"] = df_segmented.apply(segment_logic, axis=1)
    return df_segmented


def get_segment_summary(df_segmented: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate count, revenue share, and average R/F/M metrics per segment."""
    summary = df_segmented.groupby("Segment").agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    total_customers = len(df_segmented)
    total_rev = summary["Total_Revenue"].sum()

    summary["Customer_Share_%"] = (summary["Customer_Count"] / total_customers * 100).round(2)
    summary["Revenue_Share_%"] = (summary["Total_Revenue"] / total_rev * 100).round(2)
    summary["Avg_Recency"] = summary["Avg_Recency"].round(1)
    summary["Avg_Frequency"] = summary["Avg_Frequency"].round(1)
    summary["Avg_Monetary"] = summary["Avg_Monetary"].round(2)
    summary["Total_Revenue"] = summary["Total_Revenue"].round(2)

    return summary.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)


def plot_rfm_heatmap(df_segmented: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Generate 5x5 Recency vs Frequency matrix heatmap showing Average Monetary Spend."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "11_rfm_heatmap.png")

    rf_matrix = df_segmented.pivot_table(
        index="R_Score",
        columns="F_Score",
        values="Monetary",
        aggfunc="mean"
    ).sort_index(ascending=False)

    plt.figure(figsize=(9, 7), dpi=150)
    ax = sns.heatmap(
        rf_matrix,
        annot=True,
        fmt=",.0f",
        cmap="YlGnBu",
        cbar_kws={"label": "Average Monetary Spend ($)"},
        linewidths=1.5,
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.title("RFM Heatmap: Average Customer Lifetime Spend ($)\nby Recency Score (Row) vs. Frequency Score (Col)",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Frequency Score (1 = Low, 5 = High)", fontsize=11, labelpad=10)
    plt.ylabel("Recency Score (5 = Recent, 1 = Lapsed)", fontsize=11, labelpad=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_segment_distributions(
    df_segmented: pd.DataFrame,
    output_dir: str = "reports/figures"
) -> str:
    """Plot customer count and revenue share comparisons per segment."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "12_rfm_segments_treemap_bar.png")

    summary = get_segment_summary(df_segmented)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Left: Customer Count Share
    sns.barplot(
        data=summary,
        x="Customer_Share_%",
        y="Segment",
        hue="Segment",
        legend=False,
        palette="Blues_r",
        ax=axes[0]
    )
    axes[0].set_title("Customer Volume Share (%) by Segment", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Customer Share (%)", fontsize=11)
    axes[0].set_ylabel("Customer Segment", fontsize=11)
    for i, row in summary.iterrows():
        axes[0].text(row["Customer_Share_%"] + 0.3, i,
                     f"{row['Customer_Share_%']:.1f}% ({row['Customer_Count']:,})",
                     va="center", fontsize=9.5, fontweight="bold")

    # Right: Revenue Share
    sns.barplot(
        data=summary,
        x="Revenue_Share_%",
        y="Segment",
        hue="Segment",
        legend=False,
        palette="Greens_r",
        ax=axes[1]
    )
    axes[1].set_title("Total Revenue Contribution (%) by Segment", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Revenue Share (%)", fontsize=11)
    axes[1].set_ylabel("")
    for i, row in summary.iterrows():
        axes[1].text(row["Revenue_Share_%"] + 0.3, i,
                     f"{row['Revenue_Share_%']:.1f}% (${row['Total_Revenue']:,.0f})",
                     va="center", fontsize=9.5, fontweight="bold")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_rfm_treemap(df_segmented: pd.DataFrame, output_dir: str = "reports/figures") -> str:
    """Draw a proportional rectangular Treemap of RFM segments by Revenue."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "13_rfm_treemap.png")

    summary = get_segment_summary(df_segmented)
    total_rev = summary["Total_Revenue"].sum()
    shares = (summary["Total_Revenue"] / total_rev).tolist()
    labels = summary["Segment"].tolist()
    revenues = summary["Total_Revenue"].tolist()
    cust_counts = summary["Customer_Count"].tolist()

    # Layout rectangles
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # Simple slice partitioning
    y_current = 0
    # Left column: top 3 segments (At Risk, Champions, Loyal Customers)
    top_3_weight = sum(shares[:3])
    x_split = 65  # 65% width for top 3
    y_curr = 0
    for i in range(3):
        h = (shares[i] / top_3_weight) * 100
        rect = patches.Rectangle((0, y_curr), x_split, h, facecolor=colors[i], edgecolor="white", linewidth=2.5, alpha=0.88)
        ax.add_patch(rect)
        ax.text(x_split / 2, y_curr + h / 2,
                f"{labels[i]}\n${revenues[i]:,.0f}\n({shares[i]*100:.1f}% Revenue | {cust_counts[i]:,} Customers)",
                ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        y_curr += h

    # Right column: remaining 4 segments
    rest_weight = sum(shares[3:])
    y_curr = 0
    for i in range(3, len(shares)):
        h = (shares[i] / rest_weight) * 100
        rect = patches.Rectangle((x_split, y_curr), 100 - x_split, h, facecolor=colors[i], edgecolor="white", linewidth=2.5, alpha=0.88)
        ax.add_patch(rect)
        ax.text(x_split + (100 - x_split) / 2, y_curr + h / 2,
                f"{labels[i]}\n${revenues[i]:,.0f} ({shares[i]*100:.1f}%)",
                ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        y_curr += h

    plt.title("RFM Customer Segments Treemap (Proportional by Revenue)", fontsize=14, fontweight="bold", pad=12)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def run_rfm_pipeline(
    df_clean: pd.DataFrame,
    output_csv: str = "data/processed/rfm_segments.csv",
    figures_dir: str = "reports/figures"
) -> dict:
    """Execute complete RFM scoring, segmentation, table export, and visualization."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Compute raw metrics
    df_rfm, snapshot = compute_rfm_metrics(df_clean)

    # 2. Score quantiles 1-5
    df_scored = score_rfm_quantiles(df_rfm)

    # 3. Assign segments
    df_segmented = assign_rfm_segments(df_scored)

    # 4. Save to CSV
    df_segmented.to_csv(output_csv, index=False)
    print(f"Saved rfm_segments.csv to {output_csv} ({len(df_segmented):,} records)")

    # 5. Summaries and plots
    summary = get_segment_summary(df_segmented)
    heatmap_path = plot_rfm_heatmap(df_segmented, figures_dir)
    barchart_path = plot_segment_distributions(df_segmented, figures_dir)
    treemap_path = plot_rfm_treemap(df_segmented, figures_dir)

    return {
        "segmented_df": df_segmented,
        "snapshot_date": snapshot,
        "summary_table": summary,
        "heatmap_file": heatmap_path,
        "distribution_file": barchart_path,
        "treemap_file": treemap_path,
        "saved_csv": output_csv
    }
