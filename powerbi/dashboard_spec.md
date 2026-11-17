# Power BI Executive Dashboard Specification & Build Guide

**Project Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI  
**Artifact:** Dashboard Visual Specification & Implementation Guide (`powerbi/dashboard_spec.md`)  
**Target Resolution:** 16:9 Widescreen (1920 x 1080 px or 1280 x 720 px)  
**Target Tool:** Power BI Desktop (Free / Pro)  

---

## 1. Executive Design System & Style Guide

### 1.1 Color Palette
A cohesive corporate color system balances visual hierarchy, executive elegance, and functional meaning (e.g., green for retained, red for churn risk):

| Role | Color Name | Hex Code | Purpose / Application |
| :--- | :--- | :---: | :--- |
| **Primary Accent** | Deep Corporate Navy | `#1F4E79` | Primary headers, active bar fills, KPI card labels, primary series |
| **Secondary Accent** | Slate Indigo | `#2B5C8F` | Secondary series, category breakdowns, table headers |
| **Success / Low Risk**| Emerald Forest | `#2E8B57` | Retained customers, low churn risk, positive growth |
| **Warning / Medium** | Marigold Amber | `#E67E22` | Medium churn risk, transitional cohorts, caution callouts |
| **Danger / High Risk**| Crimson Alert | `#C0392B` | High churn risk, lapsed accounts, critical drop-offs |
| **Neutral Dark** | Charcoal Night | `#2C3E50` | Primary visual typography, axis titles, table values |
| **Neutral Light** | Clean Mist | `#F8F9FA` | Visual card background surfaces |
| **Canvas Background** | Slate Off-White | `#F1F4F9` | Master page canvas background |
| **Borders & Dividers**| Soft Border Gray | `#E2E8F0` | Subtle 1px visual border strokes |

### 1.2 Typography & Spacing
- **Font Family:** `Segoe UI` across all titles, cards, labels, and tooltips.
- **Page Header:** 20 pt, Bold, `#1F4E79`.
- **Card KPI Callout:** 28 pt to 32 pt, Bold, `#1F4E79` / Accent Color.
- **Card Subtitle / Category Label:** 10 pt, Regular, `#7F8C8D`.
- **Visual Title:** 12 pt to 14 pt, Bold, `#2C3E50`.
- **Axis & Data Labels:** 9 pt to 10 pt, Regular, `#2C3E50`.
- **Visual Padding & Borders:** All cards have rounded corners (8 px radius), subtle border (`#E2E8F0`), and white background surface (`#FFFFFF`).

### 1.3 Global Slicer / Filter Ribbon
To maintain consistent cross-filtering across all 4 pages, place a horizontal **Filter Banner** across the top of each page below the title header:
1. **Date / Season Slicer:** `dim_date[Season]` (Buttons / Tile: `Winter`, `Spring`, `Summer`, `Fall`).
2. **State / Location Slicer:** `fact_sales[Location]` (Dropdown search list of 50 US States).
3. **Customer Segment Slicer:** `dim_customer[RFM_Segment]` (Dropdown multi-select).
4. **Churn Risk Slicer:** `dim_customer[Risk_Level]` (Tile: `Low Risk`, `Medium Risk`, `High Risk`).

---

## 2. Page Specifications (4 Dashboard Pages)

```
+---------------------------------------------------------------------------------------------------+
| [Page Title & Breadcrumb]                                              [Date / Season / Segment]  |
+---------------------------------------------------------------------------------------------------+
|  [KPI Card 1]       |  [KPI Card 2]       |  [KPI Card 3]       |  [KPI Card 4]                   |
+---------------------------------------------------------------------------------------------------+
|                                            |                                                      |
|  [Major Trend / Distribution Visual]       |  [Breakdown Visual 1]                                |
|                                            |                                                      |
+--------------------------------------------+------------------------------------------------------+
|                                            |                                                      |
|  [Breakdown Visual 2 / Heatmap]            |  [Detailed Grid / Action Table]                      |
|                                            |                                                      |
+---------------------------------------------------------------------------------------------------+
```

---

### Page 1: Sales Overview
*Strategic Goal: Provide an immediate executive pulse on revenue volume, order velocity, basket size, and geographical product demand.*

#### A. Top KPI Cards (4 Horizontal Cards)
1. **Total Revenue:**
   - **Measure:** `[Total Revenue]`
   - **Format:** `$#,##0` (Displays: `$233,081`)
   - **Sub-label:** "Total Sales Volume"
2. **Total Orders:**
   - **Measure:** `[Total Orders]`
   - **Format:** `#,##0` (Displays: `3,900`)
   - **Sub-label:** "Transactions Completed"
3. **Average Order Value (AOV):**
   - **Measure:** `[Average Order Value]`
   - **Format:** `$#,##0.00` (Displays: `$59.76`)
   - **Sub-label:** "Average Spend per Basket"
4. **Revenue per Customer:**
   - **Measure:** `[Revenue per Customer]`
   - **Format:** `$#,##0.00` (Displays: `$59.76`)
   - **Sub-label:** "Revenue Generated per Active Buyer"

#### B. Main Visuals
1. **Monthly / Seasonal Revenue & Order Volume Trend:**
   - **Visual Type:** Clustered Column & Line Chart
   - **Shared X-Axis:** `dim_date[Month_Year]` (Sort by `dim_date[Month_Sort]` ascending)
   - **Column Y-Axis (Left):** `[Total Revenue]` (Color: Corporate Navy `#1F4E79`)
   - **Line Y-Axis (Right):** `[Total Orders]` (Color: Amber `#E67E22`, Line Width: 3)
   - **Tooltips:** `[Average Order Value]`, `Quarterly Revenue Growth %`
2. **Sales Revenue & Unit Volume by Product Category:**
   - **Visual Type:** Donut Chart / Clustered Horizontal Bar Chart
   - **Category / Legend:** `dim_product[Category]` (`Clothing`, `Accessories`, `Footwear`, `Outerwear`)
   - **Values:** `[Total Revenue]`
   - **Detail Labels:** Data value & Percentage of total
   - **Tooltips:** `[Total Units Sold]`, `[Average Order Value]`
3. **Geographic Revenue Distribution by US State:**
   - **Visual Type:** Filled Map (or Horizontal Ranked Bar Chart)
   - **Location:** `fact_sales[Location]` (US State)
   - **Color Saturation / Values:** `[Total Revenue]`
   - **Tooltips:** `[Total Orders]`, `[Average Order Value]`
4. **Fulfillment & Payment Matrix:**
   - **Visual Type:** 100% Stacked Bar Chart
   - **Y-Axis:** `fact_sales[Payment_Method]` (Credit Card, PayPal, Cash, etc.)
   - **Legend:** `fact_sales[Shipping_Type]` (Express, Free Shipping, 2-Day, Standard)
   - **X-Axis:** `[Total Orders]`
   - **Insight:** Identifies if payment preference correlates with expedited delivery choices.

---

### Page 2: Customer Behaviour
*Strategic Goal: Uncover the depth of customer loyalty, order frequency cadence, customer lifetime concentration, and discount responsiveness.*

#### A. Top KPI Cards (4 Horizontal Cards)
1. **Total Customer Base:**
   - **Measure:** `[Total Customers]`
   - **Format:** `#,##0` (Displays: `3,900`)
2. **Repeat Customer %:**
   - **Measure:** `[Repeat Customer %]`
   - **Format:** `0.0%` (Displays: `83.5%` returning established buyers)
3. **Average Order Cadence:**
   - **Measure:** `[Average Days Between Orders]`
   - **Format:** `0.0` (Displays: `48.5 Days`)
4. **Customer Review Rating:**
   - **Measure:** `[Average Customer Rating]`
   - **Format:** `0.00 ★` (Displays: `3.75 ★`)

#### B. Main Visuals
1. **Customer Base Composition: Established vs. Early-Stage by Season:**
   - **Visual Type:** 100% Stacked Column Chart
   - **X-Axis:** `dim_date[Season]` (`Winter`, `Spring`, `Summer`, `Fall`)
   - **Legend:** `dim_customer[Customer_Type]` (`Returning Customer` vs. `New Customer`)
   - **Y-Axis:** `[Total Customers]`
   - **Colors:** Returning (`#1F4E79`), New (`#85C1E9`)
2. **Customer Purchasing Frequency Distribution:**
   - **Visual Type:** Horizontal Bar Chart
   - **Y-Axis:** `dim_customer[Frequency_of_Purchases]` (Weekly, Bi-Weekly, Fortnightly, Monthly, Every 3 Months, Quarterly, Annually)
   - **X-Axis:** `[Total Customers]`
   - **Data Labels:** Value & % of Customer Base
   - **Insight:** Highlights whether the active user base is high-velocity (weekly) or periodic (quarterly).
3. **Pareto Spend Concentration (Lifetime Spend vs. Historical Orders):**
   - **Visual Type:** Scatter Chart
   - **X-Axis:** `dim_customer[Lifetime_Orders]` (1 to 51)
   - **Y-Axis:** `dim_customer[Lifetime_Revenue]` ($50 to $4,500)
   - **Size Bubble:** `dim_customer[AOV]`
   - **Legend:** `dim_customer[Loyalty_Type]`
   - **Tooltips:** `CustomerID`, `Review_Rating`, `Location`
4. **Subscription Status & Promotional Sensitivity:**
   - **Visual Type:** Clustered Column Chart
   - **X-Axis:** `dim_customer[Subscription_Status]` (`Yes` vs. `No`)
   - **Legend:** `fact_sales[Discount_Applied]` (`Yes` vs. `No`)
   - **Y-Axis:** `[Average Order Value]`
   - **Insight:** Reveals whether subscriber loyalty members buy without discounts or if promotions drive basket enlargement.

---

### Page 3: Customer Segmentation (RFM & Machine Learning)
*Strategic Goal: Deliver deep visibility into heuristic RFM segments and unsupervised K-Means clusters to guide personalized marketing campaigns.*

#### A. Top KPI Cards (4 Horizontal Cards)
1. **Total Customer Lifetime Value:**
   - **Measure:** `[Total Lifetime Revenue]`
   - **Format:** `$#,##0` (Displays: `$6,152,801`)
2. **Average Customer Lifetime Value (CLV):**
   - **Measure:** `[Average Customer Lifetime Value]`
   - **Format:** `$#,##0.00` (Displays: `$1,577.64`)
3. **Champions Segment Revenue:**
   - **Measure:** `[Champions Revenue]`
   - **Format:** `$#,##0`
   - **Sub-label:** "Top Tier Heuristic RFM Value"
4. **High-Value Active Cluster Revenue:**
   - **Measure:** `[High-Value Active Revenue]`
   - **Format:** `$#,##0`
   - **Sub-label:** "Core ML High-Engaged Cluster"

#### B. Main Visuals
1. **RFM Segments Portfolio Distribution:**
   - **Visual Type:** Treemap (or Clustered Horizontal Bar)
   - **Group:** `dim_customer[RFM_Segment]` (`Champions`, `Loyal`, `Potential Loyalist`, `At Risk`, `Hibernating / Lost`)
   - **Values:** `[Total Lifetime Revenue]`
   - **Tooltips:** `[Total Customers]`, `[Average Order Value]`
2. **K-Means Cluster Performance Dual-Axis Comparison:**
   - **Visual Type:** Clustered Column Chart
   - **X-Axis:** `dim_customer[Cluster_Label]` (`High-Value Active`, `High-Value Inactive`, `Medium-Value`, `Low-Value`)
   - **Y-Axis (Left):** `[Total Customers]`
   - **Y-Axis (Right):** `[Total Lifetime Revenue]`
   - **Insight:** High-Value Inactive represents 59.8% of total revenue—highlighting re-activation potential.
3. **RFM Behavior Scatter (Recency vs. Frequency with Spend Sizing):**
   - **Visual Type:** Scatter Chart
   - **X-Axis:** `dim_customer[Recency]` (Dormancy in Days)
   - **Y-Axis:** `dim_customer[Frequency]` (Lifetime Transactions)
   - **Size:** `dim_customer[Monetary]` (Lifetime Dollars)
   - **Legend:** `dim_customer[Cluster_Label]`
   - **Play Axis (Optional):** `dim_customer[Age_Group]`
4. **Segment Cross-Tabulation Diagnostic Matrix:**
   - **Visual Type:** Matrix
   - **Rows:** `dim_customer[RFM_Segment]`
   - **Columns:** `dim_customer[Cluster_Label]`
   - **Values:** `[Total Customers]`, `[Total Revenue]`, `[Average Order Value]`
   - **Conditional Formatting:** Background color gradient on customer volume.

---

### Page 4: Business Insights & Churn Risk Prediction
*Strategic Goal: Empower retention teams to proactively rescue at-risk accounts, analyze machine learning churn signals, and maximize product retention.*

#### A. Top KPI Cards (4 Horizontal Cards)
1. **High-Risk Churn Accounts:**
   - **Measure:** `[High-Risk Churn Customers]`
   - **Format:** `#,##0` (Displays: `2,812 Customers`)
   - **Color:** Critical Alert Red (`#C0392B`)
2. **High-Risk Churn %:**
   - **Measure:** `[High-Risk Churn %]`
   - **Format:** `0.0%` (Displays: `72.1%`)
3. **Medium-Risk Opportunity Cohort:**
   - **Measure:** `[Medium-Risk Churn Customers]`
   - **Format:** `#,##0` (Displays: `237 Customers`)
   - **Color:** Warning Amber (`#E67E22`)
   - **Sub-label:** "Prime Proactive Intervention Target"
4. **Low-Risk Active Base:**
   - **Measure:** `[Low-Risk Churn Customers]`
   - **Format:** `#,##0` (Displays: `851 Customers`)
   - **Color:** Success Emerald (`#2E8B57`)

#### B. Main Visuals
1. **Churn Risk Breakdown Across RFM Segments:**
   - **Visual Type:** 100% Stacked Bar Chart
   - **Y-Axis:** `dim_customer[RFM_Segment]`
   - **Legend:** `dim_customer[Risk_Level]` (`High Risk (Churning)`, `Medium Risk`, `Low Risk (Active)`)
   - **X-Axis:** `[Total Customers]`
   - **Colors:** High (`#C0392B`), Medium (`#E67E22`), Low (`#2E8B57`)
2. **Top 10 Revenue-Generating Products & Satisfaction Score:**
   - **Visual Type:** Clustered Horizontal Bar Chart
   - **Y-Axis:** `dim_product[ProductName]` (Filtered to Top 10 by `[Total Revenue]`)
   - **X-Axis:** `[Total Revenue]`
   - **Color Gradient:** By `[Average Customer Rating]`
   - **Tooltips:** `[Total Units Sold]`, `dim_product[Category]`
3. **Churn Probability by Demographic Age Group & Subscription:**
   - **Visual Type:** Heatmap Matrix
   - **Rows:** `dim_customer[Age_Group]` (`18-25`, `26-35`, `36-50`, `51-65`, `66+`)
   - **Columns:** `dim_customer[Subscription_Status]` (`Yes`, `No`)
   - **Values:** `[Average Churn Probability]`
   - **Conditional Formatting:** Color scale from Green (0.0) to Red (1.0).
4. **Actionable Re-Activation Customer Worklist:**
   - **Visual Type:** Table / Grid Visual
   - **Columns:**
     1. `dim_customer[CustomerID]`
     2. `dim_customer[Location]`
     3. `dim_customer[RFM_Segment]`
     4. `dim_customer[Cluster_Label]`
     5. `dim_customer[Recency]`
     6. `dim_customer[Lifetime_Revenue]`
     7. `dim_customer[Churn_Probability]`
     8. `dim_customer[Risk_Level]`
   - **Pre-set Visual Level Filter:** `dim_customer[Risk_Level] = "Medium Risk"` (The 237 transitional customers who can be saved with low-cost interventions before full lapse).
   - **Export Option:** Allows marketing operators to export directly to CSV for email campaign targeting.

---

## 3. Step-by-Step Power BI Desktop Build Checklist

Follow this linear checklist to construct the `.pbix` report in Power BI Desktop from scratch:

### Phase 1: Ingestion & Power Query Transformation
- [ ] **Open Power BI Desktop** $\rightarrow$ Click **Get Data** $\rightarrow$ Select **Text/CSV**.
- [ ] Ingest the 4 files located in `powerbi/`:
  - [ ] `fact_sales.csv`
  - [ ] `dim_customer.csv`
  - [ ] `dim_product.csv`
  - [ ] `dim_date.csv`
- [ ] Click **Transform Data** to inspect Power Query data types:
  - [ ] `fact_sales`: `OrderDate` $\rightarrow$ Date; `DateKey`, `CustomerID`, `TransactionID` $\rightarrow$ Whole Number; `Revenue`, `UnitPrice` $\rightarrow$ Fixed Decimal (Currency).
  - [ ] `dim_customer`: `First_Purchase_Date`, `Last_Purchase_Date` $\rightarrow$ Date; `CustomerID`, `Age`, `Lifetime_Orders`, `Recency`, `Frequency`, `Cluster`, `Is_High_Risk` $\rightarrow$ Whole Number; `Lifetime_Revenue`, `AOV`, `Monetary`, `Churn_Probability` $\rightarrow$ Decimal Number.
  - [ ] `dim_product`: `ProductID`, `ProductName`, `Category` $\rightarrow$ Text; `Average_Unit_Price_USD`, `Total_Revenue` $\rightarrow$ Fixed Decimal.
  - [ ] `dim_date`: `Date` $\rightarrow$ Date; `DateKey`, `Year`, `Month_Number`, `Month_Sort`, `Day_of_Month`, `Day_of_Week_Number` $\rightarrow$ Whole Number.
- [ ] Click **Close & Apply**.

### Phase 2: Data Modeling & Star Schema Verification
- [ ] Navigate to the **Model View** (left pane icon).
- [ ] Verify the relationships conform exactly to the Star Schema:
  - [ ] `dim_customer[CustomerID]` $\rightarrow$ `fact_sales[CustomerID]` (1 to Many, Single)
  - [ ] `dim_product[ProductID]` $\rightarrow$ `fact_sales[ProductID]` (1 to Many, Single)
  - [ ] `dim_date[DateKey]` $\rightarrow$ `fact_sales[DateKey]` (1 to Many, Single)
- [ ] Right-click `dim_date` in the field pane $\rightarrow$ Click **Mark as Date Table** $\rightarrow$ Select `Date` column.
- [ ] Configure **Sort By Column**:
  - [ ] `dim_date[Month_Year]` $\rightarrow$ Sort by `Month_Sort`.
  - [ ] `dim_date[Day_of_Week]` $\rightarrow$ Sort by `Day_of_Week_Number`.
  - [ ] `dim_date[Month_Short]` $\rightarrow$ Sort by `Month_Number`.
  - [ ] `dim_customer[Age_Group]` $\rightarrow$ Sort by `Age_Group`.
- [ ] Set **Default Summarization** to `Do Not Summarize` for `CustomerID`, `TransactionID`, `DateKey`, `Cluster`, `R_Score`, `F_Score`, `M_Score`.
- [ ] Hide `fact_sales[CustomerID]`, `fact_sales[ProductID]`, and `fact_sales[DateKey]` from Report View.

### Phase 3: DAX Measures Table Setup
- [ ] In the **Home** tab, click **Enter Data** $\rightarrow$ Name table `_Measures` with a dummy column $\rightarrow$ Click **Load**.
- [ ] Open `powerbi/measures.dax` in a text editor.
- [ ] Copy and paste each DAX measure into Power BI (Right-click `_Measures` $\rightarrow$ **New Measure**):
  - [ ] `Total Revenue`
  - [ ] `Total Orders`
  - [ ] `Total Customers`
  - [ ] `Average Order Value`
  - [ ] `Repeat Customer %`
  - [ ] `New Customers`
  - [ ] `Returning Customers`
  - [ ] `Revenue per Customer`
  - [ ] `High-Risk Churn Customers`
  - [ ] `High-Risk Churn %`
  - [ ] `Total Lifetime Revenue`
  - [ ] `Average Customer Lifetime Value`
  - [ ] Supporting measures (Promotions, Churn tiers, Time intelligence).
- [ ] Format currency measures as `$#,##0`, percentages as `0.0%`, counts as `#,##0`.
- [ ] Delete the dummy column from `_Measures` so the table transforms into a dedicated measure group icon.

### Phase 4: Canvas Layout & Theme Formatting
- [ ] In **View** tab $\rightarrow$ **Themes** $\rightarrow$ Customize current theme:
  - [ ] Color 1: `#1F4E79` (Navy)
  - [ ] Color 2: `#2E8B57` (Emerald)
  - [ ] Color 3: `#E67E22` (Amber)
  - [ ] Color 4: `#C0392B` (Crimson)
  - [ ] Color 5: `#2B5C8F` (Slate)
  - [ ] Page background: `#F1F4F9`
  - [ ] Visual background: `#FFFFFF` with 8px corner radius.
- [ ] Create 4 tabs and rename them:
  1. `Sales Overview`
  2. `Customer Behaviour`
  3. `Customer Segmentation`
  4. `Business Insights`

### Phase 5: Building the Visuals
- [ ] **Build Page 1 (Sales Overview)**: Place top slicer banner, 4 KPI cards, Combo Monthly Trend chart, Donut Category breakdown, US Filled Map, and Payment/Shipping stacked bar.
- [ ] **Build Page 2 (Customer Behaviour)**: Place 4 loyalty KPI cards, 100% Stacked Returning/New chart, Purchase Frequency horizontal bar, Lifetime Spend vs Orders scatter plot, and Subscription/Discount clustered column.
- [ ] **Build Page 3 (Customer Segmentation)**: Place 4 CLV cards, RFM Treemap, K-Means Dual-Axis chart, Recency vs Frequency scatter plot, and Segment Cross-tab Matrix.
- [ ] **Build Page 4 (Business Insights & Churn Risk)**: Place 4 Churn KPI cards (highlighting the 237 Medium Risk customers), Churn by RFM 100% stacked bar, Top 10 Products bar chart, Age vs Subscription heatmap, and Actionable Retention Grid Table.

### Phase 6: Interactivity, Validation & Polish
- [ ] Test cross-filtering by clicking a segment in the RFM Treemap $\rightarrow$ verify all visuals on Page 3 filter dynamically.
- [ ] Verify that slicers on Date/Season, State, and Segment propagate cleanly.
- [ ] Check tooltip formatting to ensure numbers are formatted with dollar signs and commas.
- [ ] Save the completed report as `powerbi/ecommerce_customer_analytics.pbix`.
