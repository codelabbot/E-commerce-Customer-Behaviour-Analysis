"""Power BI Data Export Engine.

Exports star-schema dimensional tables for Power BI Desktop:
1. fact_sales.csv
2. dim_customer.csv (with RFM segments, K-means cluster labels, and churn probabilities)
3. dim_product.csv
4. dim_date.csv
"""

import os
import pandas as pd
import numpy as np


def export_dim_product(clean_df: pd.DataFrame, output_dir: str = "powerbi") -> pd.DataFrame:
    """Create and export dim_product dimension table."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dim_product.csv")

    # Extract distinct products and their categories
    prod_df = clean_df.groupby(["Item_Purchased", "Category"]).agg(
        Typical_Price=("Purchase_Amount_USD", "mean"),
        Total_Units_Sold=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum")
    ).reset_index()

    prod_df = prod_df.sort_values(by=["Category", "Item_Purchased"]).reset_index(drop=True)
    prod_df["ProductID"] = [f"P{i+1:02d}" for i in range(len(prod_df))]
    prod_df["Typical_Price"] = prod_df["Typical_Price"].round(2)

    cols = ["ProductID", "Item_Purchased", "Category", "Typical_Price", "Total_Units_Sold", "Total_Revenue"]
    prod_df = prod_df[cols].rename(columns={"Item_Purchased": "ProductName", "Typical_Price": "Average_Unit_Price_USD"})

    prod_df.to_csv(out_path, index=False)
    print(f"Exported dim_product to: {out_path} ({len(prod_df)} products)")
    return prod_df


def export_dim_customer(
    cust_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    churn_df: pd.DataFrame,
    output_dir: str = "powerbi"
) -> pd.DataFrame:
    """Create and export dim_customer dimension table enriched with RFM, K-Means, and Churn."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dim_customer.csv")

    merged = cust_df.copy()

    # Merge segmentation fields
    merged["RFM_Segment"] = seg_df["RFM_Segment"]
    merged["R_Score"] = seg_df["R_Score"]
    merged["F_Score"] = seg_df["F_Score"]
    merged["M_Score"] = seg_df["M_Score"]
    merged["RFM_Score_Sum"] = seg_df["RFM_Score_Sum"]
    merged["Cluster"] = seg_df["cluster"]
    merged["Cluster_Label"] = seg_df["cluster_label"]

    # Merge churn fields
    merged["Churn_Probability"] = churn_df["Churn_Probability"].round(4)
    merged["Predicted_Churn"] = churn_df["Predicted_Churn"]
    merged["Risk_Level"] = churn_df["Risk_Level"]
    merged["Is_High_Risk"] = (churn_df["Risk_Level"] == "High Risk (Churning)").astype(int)

    # Customer Type: Returning (>=10 orders) vs New / Early-Stage (<10 orders)
    merged["Customer_Type"] = np.where(merged["total_orders"] >= 10, "Returning Customer", "New Customer")

    # Age groups
    age_bins = [17, 25, 35, 50, 65, 100]
    age_labels = ["18-25", "26-35", "36-50", "51-65", "66+"]
    merged["Age_Group"] = pd.cut(merged["Age"], bins=age_bins, labels=age_labels)

    # Renaming and column ordering
    cols_rename = {
        "total_orders": "Lifetime_Orders",
        "total_revenue": "Lifetime_Revenue",
        "average_order_value": "AOV",
        "average_days_between_orders": "Avg_Gap_Days",
        "first_purchase_date": "First_Purchase_Date",
        "last_purchase_date": "Last_Purchase_Date",
        "active_months": "Active_Months"
    }
    merged = merged.rename(columns=cols_rename)

    final_cols = [
        "CustomerID",
        "Age",
        "Age_Group",
        "Gender",
        "Location",
        "Subscription_Status",
        "Review_Rating",
        "Frequency_of_Purchases",
        "Customer_Type",
        "Loyalty_Type",
        "First_Purchase_Date",
        "Last_Purchase_Date",
        "Active_Months",
        "Avg_Gap_Days",
        "Lifetime_Orders",
        "Lifetime_Revenue",
        "AOV",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score_Sum",
        "RFM_Segment",
        "Cluster",
        "Cluster_Label",
        "Churn_Probability",
        "Predicted_Churn",
        "Risk_Level",
        "Is_High_Risk"
    ]

    # Ensure Recency, Frequency, Monetary are present
    if "Recency" not in merged.columns and "Recency" in seg_df.columns:
        merged["Recency"] = seg_df["R"]
        merged["Frequency"] = seg_df["F"]
        merged["Monetary"] = seg_df["M"]
    elif "Recency" not in merged.columns and "R" in seg_df.columns:
        merged["Recency"] = seg_df["R"]
        merged["Frequency"] = seg_df["F"]
        merged["Monetary"] = seg_df["M"]

    merged = merged[final_cols]
    merged.to_csv(out_path, index=False)
    print(f"Exported dim_customer to: {out_path} ({len(merged):,} customers)")
    return merged


def export_dim_date(
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "powerbi"
) -> pd.DataFrame:
    """Create and export a complete calendar dimension dim_date."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dim_date.csv")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"Date": dates})

    df["DateKey"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    df["Date_String"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["Quarter_Name"] = "Q" + df["Date"].dt.quarter.astype(str) + " " + df["Year"].astype(str)
    df["Month_Number"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Month_Short"] = df["Date"].dt.strftime("%b")
    df["Month_Year"] = df["Date"].dt.strftime("%b %Y")
    df["Month_Sort"] = df["Date"].dt.strftime("%Y%m").astype(int)
    df["Day_of_Month"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.strftime("%A")
    df["Day_of_Week_Short"] = df["Date"].dt.strftime("%a")
    df["Day_of_Week_Number"] = df["Date"].dt.day_of_week + 1
    df["Is_Weekend"] = np.where(df["Day_of_Week_Number"].isin([6, 7]), "Weekend", "Weekday")

    # Season mapping (Meteorological)
    def get_season(m):
        if m in [12, 1, 2]:
            return "Winter"
        elif m in [3, 4, 5]:
            return "Spring"
        elif m in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["Season"] = df["Month_Number"].apply(get_season)

    # Use Date_String as Date for CSV clean loading
    df["Date"] = df["Date_String"]
    df = df.drop(columns=["Date_String"])

    cols = [
        "DateKey", "Date", "Year", "Quarter", "Quarter_Name", "Month_Number",
        "Month_Name", "Month_Short", "Month_Year", "Month_Sort", "Day_of_Month",
        "Day_of_Week", "Day_of_Week_Short", "Day_of_Week_Number", "Is_Weekend", "Season"
    ]
    df = df[cols]
    df.to_csv(out_path, index=False)
    print(f"Exported dim_date to: {out_path} ({len(df):,} calendar days)")
    return df


def export_fact_sales(
    clean_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    output_dir: str = "powerbi"
) -> pd.DataFrame:
    """Create and export fact_sales transactional fact table."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fact_sales.csv")

    fact = clean_df.copy()

    # Map ProductID
    prod_map = dict(zip(prod_df["ProductName"], prod_df["ProductID"]))
    fact["ProductID"] = fact["Item_Purchased"].map(prod_map)

    # Map Season to OrderDate
    season_date_map = {
        "Winter": "2023-01-15",
        "Spring": "2023-04-15",
        "Summer": "2023-07-15",
        "Fall": "2023-10-15"
    }
    fact["OrderDate"] = fact["Season"].map(season_date_map)
    fact["DateKey"] = fact["OrderDate"].str.replace("-", "").astype(int)
    fact["TransactionID"] = np.arange(1, len(fact) + 1)

    cols = [
        "TransactionID",
        "OrderDate",
        "DateKey",
        "CustomerID",
        "ProductID",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Size",
        "Color",
        "Payment_Method",
        "Shipping_Type",
        "Discount_Applied",
        "Promo_Code_Used",
        "Previous_Purchases",
        "Location",
        "Country"
    ]
    fact = fact[cols]
    fact.to_csv(out_path, index=False)
    print(f"Exported fact_sales to: {out_path} ({len(fact):,} transactions)")
    return fact


def run_powerbi_export(
    clean_df_path: str = "data/processed/clean_transactions.csv",
    cust_df_path: str = "data/processed/customer_table.csv",
    seg_df_path: str = "data/processed/customer_segments.csv",
    churn_df_path: str = "data/processed/churn_predictions.csv",
    output_dir: str = "powerbi"
) -> dict:
    """Execute complete Power BI dimensional dataset export."""
    clean_df = pd.read_csv(clean_df_path)
    cust_df = pd.read_csv(cust_df_path)
    seg_df = pd.read_csv(seg_df_path)
    churn_df = pd.read_csv(churn_df_path)

    dim_prod = export_dim_product(clean_df, output_dir=output_dir)
    dim_cust = export_dim_customer(cust_df, seg_df, churn_df, output_dir=output_dir)
    dim_date = export_dim_date(output_dir=output_dir)
    fact_sales = export_fact_sales(clean_df, dim_prod, output_dir=output_dir)

    return {
        "dim_product": dim_prod,
        "dim_customer": dim_cust,
        "dim_date": dim_date,
        "fact_sales": fact_sales
    }


if __name__ == "__main__":
    run_powerbi_export()
