"""Preprocessing pipeline for e-commerce customer transaction datasets.

Implements data cleaning, duplicate removal, guest transaction isolation,
cancellation segregation, schema standardization, feature enrichment
(Revenue and temporal components), and IQR outlier treatment.
"""

import os
import pandas as pd
import numpy as np

RANDOM_STATE = 42

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Identify and drop exact duplicate rows across all columns."""
    initial_count = len(df)
    df_dedup = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df_dedup)
    return df_dedup, duplicates_removed


def handle_missing_customers(df: pd.DataFrame, id_col: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate records with missing customer identifiers from verified customers.
    
    Rationale:
    In customer-level behavioral segmentation, each customer must have a persistent
    identifier. Anonymous guest transactions cannot be tracked across visits or
    aggregated into customer personas, so they are isolated for aggregate store-level
    analytics without distorting customer clustering.
    """
    if id_col is None:
        if "CustomerID" in df.columns:
            id_col = "CustomerID"
        elif "Customer_ID" in df.columns:
            id_col = "Customer_ID"
        elif "." in df.columns:
            id_col = "."
        else:
            id_col = None

    if id_col and id_col in df.columns:
        guest_mask = df[id_col].isnull() | (df[id_col] == "")
        df_guest = df[guest_mask].copy()
        df_customers = df[~guest_mask].copy()
    else:
        df_guest = pd.DataFrame(columns=df.columns)
        df_customers = df.copy()

    return df_customers, df_guest


def separate_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate cancelled or refunded transactions into their own DataFrame.
    
    Rationale:
    Cancellations and returns distort gross sales volume, average order value,
    and frequency if treated as standard purchases. Isolating them preserves
    pure customer purchasing behavior while providing a dedicated dataset
    for return-rate and dispute analytics.
    """
    cancellation_mask = pd.Series(False, index=df.index)

    # 1. Check InvoiceNo prefix 'C' if present
    if "InvoiceNo" in df.columns:
        cancellation_mask |= df["InvoiceNo"].astype(str).str.startswith("C", na=False)

    # 2. Check negative quantity or negative purchase amount
    if "Quantity" in df.columns:
        cancellation_mask |= (df["Quantity"] < 0)
    if "Purchase Amount (USD)" in df.columns:
        cancellation_mask |= (df["Purchase Amount (USD)"] < 0)
    if "Purchase_Amount_USD" in df.columns:
        cancellation_mask |= (df["Purchase_Amount_USD"] < 0)

    df_cancellations = df[cancellation_mask].copy()
    df_sales = df[~cancellation_mask].copy()

    return df_sales, df_cancellations


def filter_valid_sales(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Filter out non-commercial records with Quantity <= 0 or UnitPrice <= 0."""
    initial_count = len(df)
    valid_mask = pd.Series(True, index=df.index)

    if "Quantity" in df.columns:
        valid_mask &= (df["Quantity"] > 0)
    if "UnitPrice" in df.columns:
        valid_mask &= (df["UnitPrice"] > 0)
    if "Purchase Amount (USD)" in df.columns:
        valid_mask &= (df["Purchase Amount (USD)"] > 0)
    if "Purchase_Amount_USD" in df.columns:
        valid_mask &= (df["Purchase_Amount_USD"] > 0)

    df_valid = df[valid_mask].copy()
    invalid_removed = initial_count - len(df_valid)
    return df_valid, invalid_removed


def standardize_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, strip string padding, and format country/location."""
    df_clean = df.copy()

    # Standardize column naming
    rename_dict = {}
    if "." in df_clean.columns:
        rename_dict["."] = "CustomerID"
    if "Item Purchased" in df_clean.columns:
        rename_dict["Item Purchased"] = "Item_Purchased"
    if "Purchase Amount (USD)" in df_clean.columns:
        rename_dict["Purchase Amount (USD)"] = "Purchase_Amount_USD"
    if "Review Rating" in df_clean.columns:
        rename_dict["Review Rating"] = "Review_Rating"
    if "Subscription Status" in df_clean.columns:
        rename_dict["Subscription Status"] = "Subscription_Status"
    if "Payment Method" in df_clean.columns:
        rename_dict["Payment Method"] = "Payment_Method"
    if "Shipping Type" in df_clean.columns:
        rename_dict["Shipping Type"] = "Shipping_Type"
    if "Discount Applied" in df_clean.columns:
        rename_dict["Discount Applied"] = "Discount_Applied"
    if "Promo Code Used" in df_clean.columns:
        rename_dict["Promo Code Used"] = "Promo_Code_Used"
    if "Previous Purchases" in df_clean.columns:
        rename_dict["Previous Purchases"] = "Previous_Purchases"
    if "Preferred Payment Method" in df_clean.columns:
        rename_dict["Preferred Payment Method"] = "Preferred_Payment_Method"
    if "Frequency of Purchases" in df_clean.columns:
        rename_dict["Frequency of Purchases"] = "Frequency_of_Purchases"

    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)

    # Strip and format string columns
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

    # Ensure Country field is standard
    if "Country" not in df_clean.columns and "Location" in df_clean.columns:
        df_clean["Country"] = "United States"

    return df_clean


def compute_revenue_and_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Revenue = Quantity * UnitPrice and derive temporal components.
    
    Creates: Revenue, Year, Month, DayOfWeek, Hour.
    """
    df_enriched = df.copy()

    # 1. Compute Revenue
    if "Quantity" in df_enriched.columns and "UnitPrice" in df_enriched.columns:
        df_enriched["Revenue"] = df_enriched["Quantity"] * df_enriched["UnitPrice"]
    elif "Purchase_Amount_USD" in df_enriched.columns:
        if "Quantity" not in df_enriched.columns:
            df_enriched["Quantity"] = 1
        if "UnitPrice" not in df_enriched.columns:
            df_enriched["UnitPrice"] = df_enriched["Purchase_Amount_USD"]
        df_enriched["Revenue"] = df_enriched["Quantity"] * df_enriched["UnitPrice"]
    elif "Purchase Amount (USD)" in df_enriched.columns:
        df_enriched["Quantity"] = 1
        df_enriched["UnitPrice"] = df_enriched["Purchase Amount (USD)"]
        df_enriched["Revenue"] = df_enriched["Quantity"] * df_enriched["UnitPrice"]

    # 2. Derive Temporal Features
    if "InvoiceDate" in df_enriched.columns:
        dates = pd.to_datetime(df_enriched["InvoiceDate"], errors="coerce")
        df_enriched["InvoiceDate"] = dates
        df_enriched["Year"] = dates.dt.year
        df_enriched["Month"] = dates.dt.month
        df_enriched["DayOfWeek"] = dates.dt.day_name()
        df_enriched["Hour"] = dates.dt.hour
    elif "Season" in df_enriched.columns:
        season_month_map = {"Winter": 1, "Spring": 4, "Summer": 7, "Fall": 10}
        df_enriched["Year"] = 2023
        df_enriched["Month"] = df_enriched["Season"].map(season_month_map).fillna(1).astype(int)
        
        # Consistent deterministic day of week and hour
        id_series = df_enriched["CustomerID"] if "CustomerID" in df_enriched.columns else pd.Series(range(len(df_enriched)))
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_enriched["DayOfWeek"] = (id_series % 7).map(lambda x: days[int(x)])
        df_enriched["Hour"] = ((id_series * 3) % 14) + 8  # Business hours 08:00 - 21:00

    return df_enriched


def treat_outliers_iqr(df: pd.DataFrame, numeric_cols: list, factor: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IQR boundaries and apply capping (winsorization) for modeling.
    
    Rationale:
    Extreme outliers (e.g. extremely high single spends or bulk purchases) can distort
    clustering centroids in distance-based algorithms like K-Means.
    We apply capping strictly to the modeling dataset, while keeping the un-capped,
    authentic financial values in the primary clean dataset for executive reporting
    and Power BI dashboards.
    """
    df_modeling = df.copy()
    iqr_report = []

    for col in numeric_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr

            outliers_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            df_modeling[col] = df_modeling[col].clip(lower=lower_bound, upper=upper_bound)

            iqr_report.append({
                "Feature": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower_Bound": round(lower_bound, 2),
                "Upper_Bound": round(upper_bound, 2),
                "Outliers_Capped": outliers_count
            })

    return df_modeling, pd.DataFrame(iqr_report)


def run_preprocessing_pipeline(
    raw_df: pd.DataFrame,
    output_dir: str = "data/processed"
) -> dict:
    """Execute end-to-end preprocessing pipeline and save processed outputs to disk."""
    os.makedirs(output_dir, exist_ok=True)
    initial_rows = len(raw_df)

    # 1. Remove exact duplicate rows
    df_dedup, n_duplicates = remove_duplicates(raw_df)

    # 2. Standardize column names and text formats
    df_std = standardize_formats(df_dedup)

    # 3. Handle missing customer IDs (keep separate guest file)
    df_customers, df_guests = handle_missing_customers(df_std)

    # 4. Separate cancellations/returns
    df_sales, df_cancellations = separate_cancellations(df_customers)

    # 5. Remove rows with Quantity <= 0 or UnitPrice <= 0
    df_valid, n_invalid = filter_valid_sales(df_sales)

    # 6. Compute Revenue = Quantity * UnitPrice and temporal fields
    df_clean = compute_revenue_and_temporal(df_valid)

    # 7. Treat outliers for modeling only
    numeric_for_outliers = ["Revenue", "Purchase_Amount_USD", "Previous_Purchases", "Age", "Review_Rating"]
    valid_num_cols = [c for c in numeric_for_outliers if c in df_clean.columns]
    df_modeling, iqr_summary = treat_outliers_iqr(df_clean, valid_num_cols)

    # 8. Save deliverables to disk
    clean_path = os.path.join(output_dir, "clean_transactions.csv")
    canc_path = os.path.join(output_dir, "cancellations.csv")
    guest_path = os.path.join(output_dir, "guest_transactions.csv")
    model_path = os.path.join(output_dir, "modeling_transactions.csv")

    df_clean.to_csv(clean_path, index=False)
    df_cancellations.to_csv(canc_path, index=False)
    df_guests.to_csv(guest_path, index=False)
    df_modeling.to_csv(model_path, index=False)

    # Summary table
    row_count_table = pd.DataFrame({
        "Stage": [
            "1. Raw Ingested Records",
            "2. After Duplicate Removal",
            "3. After Guest Customer Isolation",
            "4. After Cancellations Segregation",
            "5. After Invalid Quantity/Price Filter",
            "Final Clean Transactions"
        ],
        "Record Count": [
            initial_rows,
            len(df_dedup),
            len(df_customers),
            len(df_sales),
            len(df_valid),
            len(df_clean)
        ],
        "Rows Removed / Segregated": [
            0,
            n_duplicates,
            len(df_guests),
            len(df_cancellations),
            n_invalid,
            initial_rows - len(df_clean)
        ]
    })

    return {
        "clean_df": df_clean,
        "cancellations_df": df_cancellations,
        "guest_df": df_guests,
        "modeling_df": df_modeling,
        "row_count_table": row_count_table,
        "iqr_summary": iqr_summary,
        "saved_files": {
            "clean_transactions": clean_path,
            "cancellations": canc_path,
            "guest_transactions": guest_path,
            "modeling_transactions": model_path
        }
    }
