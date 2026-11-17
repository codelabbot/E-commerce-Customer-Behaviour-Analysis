# Data Dictionary: Customer Shopping Trends Dataset

This document details the schema of the raw customer shopping trends dataset, defines data types and value domains, identifies unavailable attributes, and provides dataset provenance.

---

## 1. Raw Dataset Schema

| Column Name (Raw) | Target Feature Name | Data Type (Raw) | Target Data Type | Description & Domain Values | Nullable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **.** | `Customer ID` | Integer | Integer / String | Unique identifier assigned to each customer record (Values: 1 to 3,900). | No (0%) |
| **Age** | `Age` | Integer | Integer | Age of customer in years (Range: 18 to 70 years; Mean: 44.1). | No (0%) |
| **Gender** | `Gender` | String / Object | Category | Customer biological gender (`Male` = 68.0%, `Female` = 32.0%). | No (0%) |
| **Item Purchased** | `Item_Purchased` | String / Object | Category | Specific retail product purchased (25 distinct items, e.g., Blouse, Jewelry, Pants, Shirt). | No (0%) |
| **Category** | `Category` | String / Object | Category | High-level merchandise department (4 categories: `Clothing`, `Accessories`, `Footwear`, `Outerwear`). | No (0%) |
| **Purchase Amount (USD)** | `Purchase_Amount_USD` | Integer | Float / Int | Transaction value in US Dollars (Range: \$20 to \$100; Mean: \$59.76). | No (0%) |
| **Location** | `Location` | String / Object | Category | US State of customer purchase (50 unique US States represented). | No (0%) |
| **Size** | `Size` | String / Object | Category | Apparel or product sizing label (`S`, `M`, `L`, `XL`). | No (0%) |
| **Color** | `Color` | String / Object | Category | Color variation of merchandise (25 unique hues, e.g., Olive, Yellow, Gray, Teal). | No (0%) |
| **Season** | `Season` | String / Object | Category | Season during which the purchase was made (`Spring`, `Summer`, `Fall`, `Winter`). | No (0%) |
| **Review Rating** | `Review_Rating` | Float | Float | Product review score submitted by customer (Range: 2.5 to 5.0; Mean: 3.75). | No (0%) |
| **Subscription Status** | `Subscription_Status`| String / Object | Category / Bool | Customer enrollment in active subscription program (`Yes` = 27.0%, `No` = 73.0%). | No (0%) |
| **Payment Method** | `Payment_Method` | String / Object | Category | Payment method used for the current transaction (6 types: Credit Card, Bank Transfer, Cash, PayPal, Venmo, Debit Card). | No (0%) |
| **Shipping Type** | `Shipping_Type` | String / Object | Category | Delivery fulfillment option selected (6 types: Express, Free Shipping, Next Day Air, Standard, 2-Day Shipping, Store Pickup). | No (0%) |
| **Discount Applied** | `Discount_Applied` | String / Object | Category / Bool | Whether a sales promotional discount was applied to the order (`Yes` = 43.0%, `No` = 57.0%). | No (0%) |
| **Promo Code Used** | `Promo_Code_Used` | String / Object | Category / Bool | Whether a promotional voucher code was redeemed (`Yes` = 43.0%, `No` = 57.0%). | No (0%) |
| **Previous Purchases** | `Previous_Purchases` | Integer | Integer | Historical number of prior purchases completed by the customer (Range: 1 to 50; Mean: 25.35). | No (0%) |
| **Preferred Payment Method** | `Preferred_Payment_Method` | String / Object | Category | Primary stated payment method preferred by customer (6 options). | No (0%) |
| **Frequency of Purchases** | `Frequency_of_Purchases` | String / Object | Category / Ordinal | General purchasing cadence (7 levels: `Weekly`, `Bi-Weekly`, `Fortnightly`, `Monthly`, `Every 3 Months`, `Quarterly`, `Annually`). | No (0%) |

---

## 2. Unavailable / Non-Existent Attributes

In accordance with project integrity guidelines, no unrecorded attributes have been manufactured or synthesized. The dataset explicitly does **not** include:

1. **Exact Timestamps / Date Records**: Specific transaction dates, times of day, day-of-week timestamps, or multi-year chronological logs are unavailable (purchases are tagged solely by meteorological `Season`).
2. **Web Analytics & Clickstream**: Digital interaction logs (such as page views, browsing session length, device type, IP geolocation, bounce rates, or cart abandonments) are absent.
3. **Socio-Economic Demographics**: Household income, educational attainment, occupation, marital status, and household size are not present.
4. **Acquisition & Marketing Attribution**: Organic search queries, paid advertising campaign IDs, affiliate referrers, and email marketing click logs are absent.
5. **Order Cancellations & Return Logs**: There are no order cancellations, negative purchase quantities, or refund dispute logs (all 3,900 records represent completed customer transactions).

---

## 3. Source Attribution

- **Dataset Name**: Consumer Behavior and Shopping Trends Dataset
- **Source**: Kaggle Public Datasets repository
- **Format**: Tabular customer transactions (3,900 records, 19 fields)
- **Domain**: Multi-category retail commerce covering US consumers across 50 states.

---

## 4. Derived & Engineered Datasets Schema

### 4.1 Churn Predictions Dataset (`data/processed/churn_predictions.csv`)
Engineered strictly pre-cutoff (cutoff date: July 16, 2023) with a 91-day post-cutoff outcome evaluation window:
- **`CustomerID`**: Unique customer identification integer (1 to 3,900).
- **`Recency`**: Days between latest pre-cutoff purchase and July 16, 2023.
- **`Frequency`**: Historical previous purchases count prior to cutoff.
- **`AOV`**: Historical average order value in USD.
- **`Monetary`**: Cumulative historical dollar spend (`Frequency * AOV`).
- **`Tenure_Days`**: Estimated active relationship span in days (`active_months * 30.4375`).
- **`Avg_Gap_Days`**: Expected inter-purchase gap mapped from stated shopping cadence.
- **`Review_Rating`**: Product review rating given by customer (1.0 to 5.0).
- **`Age`**: Customer chronological age in years.
- **`Is_Subscriber`**: Binary indicator (1 if active loyalty subscriber; 0 otherwise).
- **`Discount_Applied`**: Binary indicator (1 if discounts were utilized; 0 otherwise).
- **`Gender_Male`**: Binary indicator (1 if Male; 0 if Female).
- **`Actual_Churn`**: Ground-truth target (1 if zero purchases occurred in the 91-day post-cutoff window; 0 otherwise).
- **`Predicted_Churn`**: Model classification label from Champion Gradient Boosting pipeline (0 = Active, 1 = Churning).
- **`Churn_Probability`**: Predicted probability of churning ($0.0000$ to $1.0000$).
- **`Risk_Level`**: Operational churn risk tier: `Low Risk (Active)` ($\le 0.30$), `Medium Risk` ($0.30 - 0.70$), `High Risk (Churning)` ($> 0.70$).

### 4.2 Power BI Star-Schema Export Tables (`powerbi/`)
- **`powerbi/fact_sales.csv`**: Transactional fact table (3,900 rows) with foreign keys `CustomerID`, `ProductID`, `DateKey`, plus sales metrics (`Quantity`, `UnitPrice`, `Revenue`).
- **`powerbi/dim_customer.csv`**: Conformed customer dimension (3,900 rows) enriched with RFM scores & segments, K-Means clusters, and machine learning churn probabilities & risk tiers.
- **`powerbi/dim_product.csv`**: Product dimension (25 items) with `ProductID`, `ProductName`, `Category`, and benchmark average selling prices.
- **`powerbi/dim_date.csv`**: Comprehensive calendar dimension (730 calendar days, 2022–2023) with day, month, quarter, year, weekend, and meteorological season attributes.


