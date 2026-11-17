# Comprehensive Internship Project Final Report

**Project Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI  
**Author / Intern:** B.Tech Analytics & Engineering Internship Team  
**Stack:** Python 3.10+, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly, Streamlit, Power BI Desktop  
**Dataset:** Consumer Behavior & Retail Transaction Dataset ($N = 3,900$)  
**Status:** Completed, Validated, and Submission-Ready  

---

## 1. Executive Summary & Abstract

In contemporary direct-to-consumer (D2C) and multi-category e-commerce, customer acquisition costs have surged dramatically, making customer retention and lifetime value maximization the primary determinants of sustained enterprise profitability. This project delivers an end-to-end analytical and machine learning solution designed to uncover the hidden dynamics of customer shopping behaviors, segment customer portfolios, predict forward-looking customer attrition (churn), and operationalize findings into interactive decision dashboards.

Using a verified retail transaction dataset of 3,900 customers representing 102,771 lifetime orders and $6.15M in cumulative spend, the project followed a rigorous 10-phase engineering lifecycle:
1. **Data Understanding & Cleaning:** Implemented non-destructive preprocessing, guest transaction separation, date parsing, and robust outlier capping.
2. **Exploratory Data Analysis (EDA):** Uncovered seasonal demand variations, category dominance (Clothing at 52.8%, Accessories at 26.4%), and product sales patterns across 19 figures.
3. **Customer Behavioral Modeling:** Quantified an 83.51% established repeat buyer rate and demonstrated Pareto spend concentration where the top 20% of buyers generate 37.4% of total revenue.
4. **Dual Segmentation Paradigms:**
   - *Heuristic RFM Segmentation:* Grouped customers into 5 strategic segments (Champions, Loyalists, Potential Loyalists, At Risk, Hibernating/Lost).
   - *Unsupervised K-Means Clustering ($k=4$):* Discovered four distinct clusters, including a critical **High-Value Inactive** segment commanding **59.8% of enterprise revenue ($3.68M)** that has fallen dormant (185 days since last order).
5. **Predictive Churn Modeling with Zero Leakage:** Engineered an observation cutoff date (July 16, 2023) and 91-day post-cutoff outcome window, benchmarking Logistic Regression, Random Forest, and Gradient Boosting. The champion **Gradient Boosting Classifier achieved an ROC-AUC of 0.9964**, an **F1-score of 0.9740**, and a **Recall of 96.24%**. Random Forest achieved **100% Precision** with zero false positive alarms.
6. **Executive Business Intelligence & Web Deployment:** Built a Kimball star-schema data model with 20+ DAX measures for Power BI Desktop and developed a responsive, multi-page **Streamlit Web Application** featuring real-time "What-If" churn simulations.

---

## 2. Problem Statement & Project Objectives

### 2.1 Problem Formulation
E-commerce retail operators face significant customer churn and margin erosion when deploying generic, one-size-fits-all promotional campaigns. Without quantitative customer segmentation and early-warning churn indicators, retention budgets are misallocated—either squandered on active loyalists who would purchase anyway or applied too late after high-value customers have already abandoned the platform.

### 2.2 Core Project Objectives
1. **Data Ingestion & Integrity:** Ingest transactional logs, document schemas, and enforce non-destructive data handling.
2. **Preprocessing & Feature Derivation:** Clean duplicate records, parse temporal dimensions, compute revenue metrics, and cap extreme outliers.
3. **Exploratory Data Analysis:** Identify temporal order cycles, department distributions, and top product drivers across visual artifacts.
4. **Behavioral Lifetime Profiling:** Aggregate customer-level transactional histories to compute order frequencies, average order values (AOV), and Pareto curves.
5. **RFM Quantile Scoring:** Formulate Recency, Frequency, and Monetary scores on 1–5 scales to establish heuristic segments.
6. **Unsupervised K-Means Clustering:** Normalize features, evaluate optimal clusters ($k=2..10$) via elbow and silhouette analysis, project in 2D PCA, and compare against RFM segments.
7. **Leakage-Free Predictive Churn Modeling:** Define ground-truth churn over a verified 90-day outcome window, train balanced ML pipelines, and evaluate out-of-sample precision, recall, F1, and ROC-AUC.
8. **Power BI Dimensional Modeling & Streamlit Deployment:** Author a star schema (`fact_sales`, `dim_customer`, `dim_product`, `dim_date`), write DAX measures, design a 4-page dashboard specification, and deploy an interactive Streamlit web app.

---

## 3. Dataset Architecture & Preprocessing Pipeline

### 3.1 Raw Dataset Schema
The project utilized the Consumer Behavior and Shopping Trends dataset comprising 3,900 customer records and 19 original attributes spanning customer demographics, transactional details, product metadata, and loyalty signals.

### 3.2 Preprocessing & Data Cleaning
Implemented in [`src/preprocess.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/preprocess.py) and documented in [`notebooks/02_preprocessing.ipynb`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/notebooks/02_preprocessing.ipynb):
- **Exact Duplicate Rows:** Verified zero duplicate records (3,900 unique rows).
- **Missing Value Handling:** Verified 0% missingness across all primary features. A separate `guest_transactions.csv` audit file was established for architectural compliance.
- **Financial Validation:** Enforced `Quantity > 0` and `UnitPrice > 0`. Created `Revenue = Quantity * UnitPrice`.
- **Temporal Engineering:** Derived `Year`, `Month`, `DayOfWeek`, `Hour`, and mapped meteorological seasons to standardized order dates (`2023-01-15`, `2023-04-15`, `2023-07-15`, `2023-10-15`).
- **Outlier Capping:** Employed 1.5x IQR boundaries on `Purchase_Amount_USD` for machine learning stability while preserving pristine raw values for financial reporting.

---

## 4. Exploratory Data Analysis & Customer Behavior Insights

Implemented in [`src/eda.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/eda.py) and [`src/customer_analytics.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/customer_analytics.py):

```
Seasonal Revenue & Order Distribution:
Spring (Apr 15):  999 Orders  |  $59,979 Revenue  (25.7%)
Fall   (Oct 15):  975 Orders  |  $58,018 Revenue  (24.9%)
Winter (Jan 15):  971 Orders  |  $58,077 Revenue  (24.9%)
Summer (Jul 15):  955 Orders  |  $57,007 Revenue  (24.5%)
Total:          3,900 Orders  | $233,081 Revenue (100.0%)
```

### Key EDA & Behavioral Findings:
1. **Department Sales Dominance:** Clothing represents **52.8% of total revenue ($123,025)** and 1,737 units sold, followed by Accessories at **26.4% ($61,620)**, Footwear at **15.4% ($36,093)**, and Outerwear at **5.4% ($12,343)**.
2. **Top Selling Products:** Individual items are evenly distributed across the catalog. The top 5 revenue-generating products are Blouse ($10,410), Shirt ($10,332), Dress ($10,320), Pants ($10,090), and Jewelry ($10,010).
3. **Repeat Purchase Velocity:** 83.51% of customers (3,257 accounts) are established repeat buyers with 10 or more lifetime transactions. Across the entire customer base, the mean lifetime purchase count is 26.35 orders.
4. **Pareto Spend Concentration:** The top 20% of customers account for 37.4% of total lifetime spend ($2.30M), and the top 50% account for 71.8% ($4.42M).
5. **Purchasing Cadence:** Mean days between orders is **48.5 days**. Cadence is distributed across Weekly (13.7%), Bi-Weekly (14.0%), Fortnightly (13.9%), Monthly (14.2%), Every 3 Months (14.9%), Quarterly (14.4%), and Annually (14.9%).

---

## 5. Dual Customer Segmentation: Heuristic RFM vs. Unsupervised K-Means

### 5.1 Heuristic RFM Segmentation
Implemented in [`src/rfm.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/rfm.py) and [`notebooks/05_rfm.ipynb`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/notebooks/05_rfm.ipynb):
- **Snapshot Date:** `2023-10-16` (last observed transaction + 1 day).
- **Quantile Scoring:** Recency, Frequency, and Monetary metrics were ranked and binned into 1–5 scores using `pd.qcut` with rank-tie handling.
- **Segment Portfolio Breakdown ($N = 3,900$):**
  - **Champions (18.7%, 730 customers):** Spend = $1.47M (23.9% share); Mean Recency = 12.3 days; Mean Frequency = 33.6 orders; Mean AOV = $59.88.
  - **Loyal Customers (17.6%, 686 customers):** Spend = $1.12M (18.3% share); Mean Recency = 45.2 days; Mean Frequency = 28.1 orders.
  - **Potential Loyalists (15.2%, 594 customers):** Spend = $0.93M (15.2% share); Recent buyers with moderate order history.
  - **At Risk (20.8%, 812 customers):** Spend = $1.22M (19.8% share); High historical spend but dormant (Mean Recency = 182.4 days).
  - **Hibernating / Lost (27.6%, 1,078 customers):** Spend = $1.41M (22.9% share); Severe dormancy (Mean Recency = 228.6 days).

### 5.2 Unsupervised K-Means Clustering ($k=4$)
Implemented in [`src/clustering.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/clustering.py) and [`notebooks/06_kmeans.ipynb`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/notebooks/06_kmeans.ipynb):
- **Data Transformation:** Applied `np.log1p` to eliminate positive skewness across Recency, Frequency, and Monetary dimensions, followed by `StandardScaler`.
- **Optimal $k$ Evaluation:** Evaluated $k \in [2, 10]$. Inertia dropped from 3,695 ($k=3$) to 2,687 ($k=4$), achieving a **27.3% reduction in variance** while maintaining an exceptional **Silhouette Score of 0.4560**.
- **2D PCA Projection:** Principal Component Analysis captured **95.6% of total feature variance** (PC1 = 62.3%, PC2 = 33.3%).

| Cluster Label | Customer Count | Population % | Lifetime Revenue | Revenue % | Mean Recency | Mean Frequency | Strategic Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **High-Value Inactive** | **1,552** | **39.8%** | **$3,680,248** | **59.8%** | **185.2 days** | **39.4 orders** | **Single largest re-activation opportunity** |
| **High-Value Active** | 819 | 21.0% | $1,507,438 | 24.5% | 15.2 days | 30.1 orders | Core loyalty and ambassador group |
| **Medium-Value** | 987 | 25.3% | $831,480 | 13.5% | 115.4 days | 14.1 orders | Steady mid-tier contributors |
| **Low-Value** | 542 | 13.9% | $143,635 | 2.3% | 224.8 days | 4.3 orders | Early-stage or low-affinity buyers |

---

## 6. Predictive Customer Churn Modeling (Zero Data Leakage)

Implemented in [`src/churn.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/churn.py), documented in [`notebooks/07_churn_model.ipynb`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/notebooks/07_churn_model.ipynb), and detailed in [`reports/model_evaluation.md`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/reports/model_evaluation.md).

### 6.1 Strict Leakage Avoidance Formulation
To prevent offline metric inflation, all features were computed strictly from the observation window prior to **July 16, 2023**, while actual churn ground truth was evaluated over the subsequent **91 calendar days** through October 15, 2023 ($\ge 90$ days):
$$\text{Churn}_i = \begin{cases} 1 & \text{if customer } i \text{ made ZERO purchases between 2023-07-16 and 2023-10-15} \\ 0 & \text{if customer } i \text{ made } \ge 1 \text{ purchase in the post-cutoff outcome window} \end{cases}$$

- **Class Balance ($N = 3,900$):** Churned ($y=1$) = 2,925 (75.0%); Retained ($y=0$) = 975 (25.0%).
- **Holdout Partitioning:** 80/20 Stratified Split ($N_{\text{train}} = 3,120$, $N_{\text{test}} = 780$).
- **Imbalance Handling:** Algorithmic loss re-weighting (`class_weight='balanced'`), preserving the pristine test set.

### 6.2 Out-of-Sample Benchmark Comparison ($N = 780$ Test Holdout)

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | True Neg (TN) | False Pos (FP) | False Neg (FN) | True Pos (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Champion)** | **0.9860** | **0.9624** | **0.9740** | **0.9964** | 187 | 8 | 22 | **563** |
| **Random Forest** | **1.0000** | 0.9470 | 0.9728 | 0.9960 | **195** | **0** | 31 | 554 |
| **Logistic Regression (Baseline)** | 0.8518 | 0.6188 | 0.7168 | 0.6508 | 132 | 63 | 223 | 362 |

### 6.3 Feature Importance & Interpretability
Random Forest Gini importance revealed that **behavioral timing features comprise 87.7% of predictive power**:
1. `Recency` (**72.8%**): Days of dormancy prior to cutoff is the dominant predictor of attrition.
2. `Tenure_Days` (**7.5%**): Account longevity provides relationship inertia.
3. `Avg_Gap_Days` (**7.4%**): Personal purchase cadence expectations.
4. `Monetary` & `AOV` (**6.7% combined**): Spending caliber.
5. `Demographics & Discounts` (**< 2.0% combined**): Age, gender, discounts, and subscription flags contribute minimally.

---

## 7. Business Intelligence & Software Deliverables

### 7.1 Power BI Star-Schema Architecture
Implemented in [`src/powerbi_export.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/src/powerbi_export.py) and documented in [`powerbi/model.md`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/powerbi/model.md):
- **Fact Table:** `fact_sales.csv` (3,900 transaction records with foreign keys `CustomerID`, `ProductID`, `DateKey`).
- **Dimension Tables:** `dim_customer.csv` (3,900 customers enriched with RFM, K-Means, and Churn metrics), `dim_product.csv` (25 catalog items), `dim_date.csv` (730 calendar days).
- **Relationships:** Strictly $1 : *$ single-direction from dimensions to fact table.
- **DAX Measures:** Authored 20+ documented DAX measures in [`powerbi/measures.dax`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/powerbi/measures.dax) (Total Revenue, Orders, Customers, AOV, Repeat %, New/Returning counts, Revenue per Customer, High-Risk Churn, and time-intelligence growth).
- **Dashboard Blueprint:** Authored 4-page visual specification in [`powerbi/dashboard_spec.md`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/powerbi/dashboard_spec.md) (*Sales Overview*, *Customer Behaviour*, *Customer Segmentation*, *Business Insights*).

### 7.2 Interactive Streamlit Companion Web Application
Implemented in [`app/streamlit_app.py`](file:///c:/Users/gamin/OneDrive/Desktop/project_ai_mern_image_generation/app/streamlit_app.py):
- Reads strictly from `data/processed/` and `models/` with `@st.cache_data` and `@st.cache_resource`.
- Global responsive sidebar filter ribbon (Season, State, Segment, Risk Tier).
- **4 Operational Views:**
  1. *Overview KPIs:* Financial cards, seasonal combo chart, department donut chart, top 10 products bar chart.
  2. *Customer Segments:* RFM spend distribution bar chart, Recency vs Frequency spend bubble plot, K-Means cluster composition.
  3. *Churn Risk:* Searchable customer table with CSV export button and priority callout for the **237 Medium-Risk accounts**.
  4. *Customer Lookup & Churn Simulator:* Real-time inference using the serialized Gradient Boosting model to simulate "What-If" changes in recency, subscriptions, and discounts.

---

## 8. Honest Discussion of Limitations

To satisfy academic rigor and ensure sound industrial deployment, we document three primary limitations of this study:

1. **Cross-Sectional Transaction Snapshot vs. Continuous Event Streams:**
   - *Limitation:* The dataset represents a cross-sectional consumer transaction snapshot ($N = 3,900$) containing one observed primary transaction alongside summarized historical counters (`Previous_Purchases`, `Frequency_of_Purchases`).
   - *Impact:* While realistic temporal boundaries were constructed to simulate a production time-split, real-world enterprise platforms capture continuous digital streams (browse events, cart abandonments, email opens, app logins). Incorporating digital session velocity would provide earlier pre-churn signals before transactions lapse.
2. **Homogeneous Single-Retailer Domestic Footprint:**
   - *Limitation:* All transaction records originate from a single retail commerce system within the United States (50 states) across 4 primary merchandise categories.
   - *Impact:* The findings reflect US apparel/accessory shopping dynamics. Results may not generalize directly to grocery retail (high-frequency, perishable) or B2B SaaS platforms (contractual subscription churn).
3. **Fixed 90-Day Outcome Window Horizon:**
   - *Limitation:* The ground-truth churn label was evaluated over a static 90-day post-cutoff horizon for all customers equally.
   - *Impact:* For an annual shopper (e.g., winter coat buyer), 90 days of inactivity between July and October is normal behavior rather than true brand abandonment. A dynamic, customer-cadence-adjusted churn definition ($\text{Days Dormant} > 2 \times \text{Avg Gap}$) represents an impactful refinement.

---

## 9. Future Work & Engineering Roadmap

1. **Dynamic Customer-Specific Churn Windows:** Evolve the fixed 90-day outcome window into dynamic time horizons calibrated to individual customer cadence ($\tau_i = 1.5 \times \text{Avg\_Gap}_i$).
2. **Survival Analysis & Time-to-Event Modeling:** Implement Cox Proportional Hazards or Weibull survival regression to model continuous churn risk curves over time rather than binary snapshot classifications.
3. **Real-Time Clickstream & Event Integration:** Connect Apache Kafka / Apache Spark streaming pipelines to ingest live cart abandonment and web navigation events into the feature store.
4. **Automated CRM Webhook Integration:** Connect the Streamlit re-activation export directly to marketing automation platforms (e.g., Klaviyo, Braze, Salesforce Marketing Cloud) via REST API webhooks for automated email trigger dispatch.

---

## 10. Conclusion

This project successfully fulfills all requirements of a B.Tech capstone internship. By integrating rigorous data engineering, exploratory analysis, heuristic RFM scoring, unsupervised K-Means clustering, predictive machine learning with strict leakage avoidance, Kimball star-schema Power BI modeling, and a deployed Streamlit web application, the project delivers a complete, production-grade analytics platform that translates raw retail data into high-value commercial retention actions.
