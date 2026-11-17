# Final Internship Presentation Slide Deck Outline & 3-Minute Demo Script

**Project Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI  
**Academic Level:** B.Tech Major Project / Internship Viva Voce  
**Document Purpose:** Presentation slide outline, speaker talking points, live demonstration script, and viva defense guide.  

---

## Part 1: 12-Slide Final Presentation Deck Outline

### Slide 1: Title & Executive Summary
- **Slide Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI
- **Subtitle:** An End-to-End Analytics, Unsupervised Clustering, Predictive Churn, and Business Intelligence System
- **Key Information:**
  - Intern Name / Roll Number / Degree (B.Tech Computer Science / Data Science)
  - Host Organization / Academic Guide
- **Speaker Talking Point (30s):**
  > "Good morning, respected panel members. In modern e-commerce, acquiring a new customer is up to 7 times more expensive than retaining an existing one. Today, I am presenting an end-to-end data science and machine learning system that transforms 3,900 customer shopping histories into actionable retention strategies, predicts customer churn with a 0.9964 ROC-AUC score, and delivers operational dashboards in Power BI and Streamlit."

---

### Slide 2: Problem Statement & Research Objectives
- **Slide Title:** The Retention Dilemma: Why E-Commerce Platforms Struggle
- **Visuals:** Visual comparison of "Blind Promotional Blasts" vs. "Targeted Cadence Marketing".
- **Key Bullet Points:**
  - *The Core Problem:* Blanket discounting causes severe margin erosion, while delayed interventions fail to prevent customer attrition.
  - *Goal 1:* Quantify purchasing habits, lifetime order velocity, and spend concentration (Pareto principle).
  - *Goal 2:* Segment customers using both heuristic RFM and unsupervised K-Means machine learning.
  - *Goal 3:* Build a leak-free predictive model to forecast customer churn over a forward-looking 90-day horizon.
  - *Goal 4:* Operationalize insights into Kimball star-schema Power BI dashboards and a deployed Streamlit web app.

---

### Slide 3: Dataset Architecture & Data Preprocessing Framework
- **Slide Title:** Data Architecture & Integrity Protocols
- **Visuals:** Diagram showing raw data $\rightarrow$ cleaning pipeline $\rightarrow$ processed datasets.
- **Key Bullet Points:**
  - *Dataset Scope:* 3,900 customer records, 102,771 lifetime orders, $6.15M cumulative spend.
  - *Global Rule Enforcement:* Strict compliance with academic integrity; zero fabrication of missing fields.
  - *Data Cleaning Highlights:*
    - Verified 0 duplicate rows and 0% null values.
    - Derived `Revenue = Quantity * UnitPrice`.
    - Handled date parsing and mapped seasonal intervals to standardized timestamps.
    - Capped 1.5x IQR outliers for modeling stability while keeping raw values for reporting.
  - *Processed Manifest:* 4 processed CSV datasets and separate audit tables for returns and guest transactions.

---

### Slide 4: Exploratory Data Analysis & Sales Demand Patterns
- **Slide Title:** Exploratory Analysis: Category & Sales Dynamics
- **Visuals:** Embedded charts of Seasonal Revenue Trend and Category Breakdown (`01_monthly_revenue_order_trend.png` & `03_top_products_revenue_quantity.png`).
- **Key Bullet Points:**
  - *Stable Seasonal Velocity:* Total observed revenue = $233,081 across 3,900 transactions.
    - Spring: $59,979 (25.7%) | Fall: $58,018 (24.9%) | Winter: $58,077 (24.9%) | Summer: $57,007 (24.5%).
  - *Category Revenue Concentration:*
    - **Clothing:** 52.8% ($123,025 | 1,737 units).
    - **Accessories:** 26.4% ($61,620 | 1,240 units).
    - **Footwear:** 15.4% ($36,093) | **Outerwear:** 5.4% ($12,343).
  - *Top Products:* Uniform distribution across 25 items led by Blouse ($10.4K), Shirt ($10.3K), and Dress ($10.3K).

---

### Slide 5: Customer Behavior & The Pareto Spend Distribution
- **Slide Title:** Customer Behavior: Repurchase Velocity & Spend Concentration
- **Visuals:** Pareto Cumulative Revenue Curve (`09_pareto_revenue_concentration.png`).
- **Key Bullet Points:**
  - *Established Repeat Buyer Rate:* **83.51% of customers (3,257)** have completed 10 or more lifetime orders; average customer lifetime transactions = 26.35.
  - *Order Cadence:* Mean inter-purchase interval is **48.5 days**.
  - *Pareto 80/20 Validation:*
    - Top 20% of customers generate **37.4% of lifetime revenue ($2.30M)**.
    - Top 50% of customers generate **71.8% of lifetime revenue ($4.42M)**.
  - *Takeaway:* Retaining the top tier is paramount; customer attrition in the upper deciles directly threatens platform solvency.

---

### Slide 6: Heuristic RFM Customer Segmentation
- **Slide Title:** Heuristic RFM Segmentation: 5 Actionable Strategic Tiers
- **Visuals:** RFM Segment Treemap and Correlation Heatmap (`11_rfm_heatmap.png` & `13_rfm_treemap.png`).
- **Key Bullet Points:**
  - *Methodology:* Snapshot date `2023-10-16`. Recency, Frequency, and Monetary scored on 1–5 quantiles with rank-tie handling.
  - *The 5 Strategic Portfolios:*
    - **Champions (18.7% | 730 cust):** $1.47M spend (23.9% share), 12.3d recency, 33.6 orders.
    - **Loyal Customers (17.6% | 686 cust):** $1.12M spend (18.3% share), 45.2d recency.
    - **Potential Loyalists (15.2% | 594 cust):** $0.93M spend, promising early trajectory.
    - **At Risk (20.8% | 812 cust):** $1.22M spend, high historical value but 182d dormant.
    - **Hibernating / Lost (27.6% | 1,078 cust):** $1.41M spend, 228d dormant.

---

### Slide 7: Unsupervised Machine Learning via K-Means Clustering
- **Slide Title:** Unsupervised Clustering ($k=4$): The $3.68M Dormancy Discovery
- **Visuals:** Elbow Curve & Silhouette Plot, and 2D PCA Cluster Scatter (`14_kmeans_elbow_silhouette.png` & `15_kmeans_pca_2d.png`).
- **Key Bullet Points:**
  - *Data Transformation:* `log1p` transformation + `StandardScaler` to remove skewness.
  - *Mathematical Justification:* Optimal $k=4$ drops inertia by 27.3% (from 3,695 to 2,687) with a high **Silhouette Score of 0.4560**. PCA captures **95.6% of total variance**.
  - *The Major Finding: High-Value Inactive Cluster:*
    - **1,552 customers (39.8% of population)** control **$3,680,248 (59.8% of revenue)**!
    - Average dormancy is **185.2 days**.
    - *Strategic Implication:* Re-activating just 15% of this cluster restores **$367,000 in annual customer equity**.

---

### Slide 8: Predictive Churn Modeling with Zero Data Leakage
- **Slide Title:** Predictive Churn Formulation: Preventing Data Leakage
- **Visuals:** Timeline diagram showing observation cutoff vs. 91-day outcome window.
- **Key Bullet Points:**
  - *The Data Leakage Trap:* Most academic churn models use future transaction features to predict future churn, causing offline test scores to collapse in production.
  - *Our Leakage-Free Architecture:*
    - **Cutoff Date:** July 16, 2023.
    - **Outcome Window:** 91 days (through October 15, 2023; satisfying $\ge 90$ days).
    - **Target Variable ($y$):** 1 if customer made 0 purchases after cutoff (75.0% churn), 0 if active (25.0%).
    - **Feature Isolation:** 11 features extracted strictly on or before July 16, 2023.
    - **Pipelines:** In-pipeline `StandardScaler` + `class_weight='balanced'` on 80/20 stratified split.

---

### Slide 9: Machine Learning Benchmark & Feature Importance
- **Slide Title:** Churn Model Evaluation: Benchmark & Behavioral Drivers
- **Visuals:** Comparative ROC Curves and Feature Importance Bar Chart (`17_churn_roc_curves.png` & `19_churn_feature_importance.png`).
- **Key Bullet Points:**
  - *Out-of-Sample Test Benchmark ($N = 780$):*
    - **Gradient Boosting (Champion):** **ROC-AUC = 0.9964**, **Recall = 96.24%**, **Precision = 98.60%**, **F1 = 0.9740**.
    - **Random Forest:** **ROC-AUC = 0.9960**, **Precision = 100.0%** (zero false alarms!), **Recall = 94.70%**.
    - **Logistic Regression (Baseline):** **ROC-AUC = 0.6508**, **Recall = 61.88%** (linear models fail).
  - *Feature Importance (Random Forest Gini):*
    - **Recency (72.8%) + Tenure (7.5%) + Avg Gap (7.4%) = 87.7% of total predictive power!**
    - Demographics (Age 1.3%, Gender 0.2%) and discounts (0.3%) contribute less than 2% combined.

---

### Slide 10: Executive Power BI Dashboard Architecture
- **Slide Title:** Business Intelligence: Kimball Star Schema & DAX Engine
- **Visuals:** Star Schema Entity Relationship Diagram (`powerbi/model.md`).
- **Key Bullet Points:**
  - *Star Schema Data Model:* Central `fact_sales` (3,900 rows) surrounded by `dim_customer`, `dim_product`, and `dim_date` (730 days).
  - *Strict Modeling Standards:* 1-to-many single-direction relationships; marked date table; hidden foreign keys.
  - *DAX Calculation Engine:* 20+ measures for Revenue, Orders, AOV, Repeat Customer %, Churn %, and Time Intelligence.
  - *4-Page Executive Dashboard Specification:*
    1. *Sales Overview* | 2. *Customer Behaviour* | 3. *Customer Segmentation* | 4. *Business Insights & Churn*.

---

### Slide 11: Interactive Streamlit Companion Web Application
- **Slide Title:** Software Deployment: Interactive Streamlit Web Suite
- **Visuals:** Screenshot / live UI view of `app/streamlit_app.py`.
- **Key Bullet Points:**
  - *Architecture:* Pure Python frontend reading strictly from `data/processed/` and `models/` with `@st.cache_data`.
  - *Global Filter Ribbon:* Season, State, Segment, and Risk Tier filters.
  - *The 4 Operational Views:*
    - **Overview KPIs:** Real-time revenue and category analytics.
    - **Segmentation Explorer:** RFM vs K-Means cross-comparison.
    - **Churn Risk Table:** Searchable table with CSV export for the **237 Medium-Risk rescue accounts**.
    - **Customer 360 Lookup & Real-Time Simulator:** Sliders to adjust recency, subscriptions, and discounts to re-calculate ML churn probability live!

---

### Slide 12: Strategic Recommendations, Limitations & Roadmap
- **Slide Title:** Commercial Action Plan & Research Limitations
- **Key Bullet Points:**
  - *Core Recommendations:*
    1. Launch $3.68M Win-Back campaign for High-Value Inactive accounts.
    2. Proactively rescue the 237 Medium-Risk accounts before 90-day lapse.
    3. Eliminate discounts for Champions (preserve $35K+ gross margin).
    4. Implement $75 free shipping threshold and product bundles.
  - *Transparent Limitations:*
    - Cross-sectional transaction snapshot vs continuous clickstream event streams.
    - Single US retailer footprint across 4 apparel categories.
    - Static 90-day time window vs individualized cadence-based horizons.
  - *Future Work:* Continuous survival analysis, dynamic churn windows, and CRM webhooks.

---

## Part 2: 3-Minute Live Demonstration Script

*Context: Delivered during the viva voce or project defense. The presenter operates the Streamlit web application on screen while narrating.*

### Minute 1: The Context, Problem & Segmentation (0:00 – 1:00)
> *"Respected examiners, welcome to the demonstration of our E-Commerce Customer Analytics and Churn AI platform.*
>
> *[Screen showing Streamlit Page: 'Overview KPIs']*  
> *Here on the Overview page, we see our commercial baseline: 3,900 customers completing $233,000 in sales with an Average Order Value of $59.76. Notice that 52.8% of revenue is generated by Apparel, led by Blouses and Shirts.*
>
> *[Clicking to Navigation: 'Customer Segments']*  
> *Moving to Customer Segmentation, our analysis revealed something extraordinary. While traditional RFM identifies our top Champions who drive 23.9% of spend, our unsupervised K-Means clustering algorithm uncovered a massive commercial vulnerability: the **High-Value Inactive cluster**. This group of 1,552 customers represents **$3.68 million—nearly 60% of our enterprise lifetime revenue**—yet they have fallen dormant, averaging 185 days since their last purchase. Re-activating just 15% of this group recovers over $360,000 in customer equity."*

### Minute 2: Predictive Machine Learning & Leakage Avoidance (1:00 – 2:00)
> *[Clicking to Navigation: 'Churn Risk (Searchable Table)']*  
> *"To prevent customers from becoming dormant in the first place, we engineered a forward-looking predictive churn model.*
>
> *Crucially, we avoided data leakage by fixing a strict cutoff date at July 16, 2023, building features solely on historical data, and evaluating ground-truth churn over the subsequent 91 days. Benchmarking three algorithms, our champion **Gradient Boosting model achieved an ROC-AUC of 0.9964 and Recall of 96.24%**, while Random Forest delivered a perfect 100% precision.*
>
> *Feature importance proved that **purchase timing—Recency, Tenure, and expected Cadence—drives 87.7% of churn risk**, whereas demographics contribute under 2%.*
>
> *Here on the Churn Risk screen, our system isolates **237 customers in the Medium-Risk tier**. These buyers have a 30% to 70% churn probability. They have not yet crossed the 90-day dormancy cliff, making them our highest-priority proactive rescue targets. Operators can filter this table and click **'Export Filtered Cohort to CSV'** to trigger automated retention emails immediately."*

### Minute 3: Customer 360 Lookup & Real-Time Simulator (2:00 – 3:00)
> *[Clicking to Navigation: 'Customer Lookup by ID']*  
> *"Finally, let’s explore the Customer 360 Diagnostic and Live Simulator.*
>
> *If we enter Customer #1, we instantly see their profile: they are in the Medium-Value cluster, dormant for 274 days, with a high churn probability of 99.6%. Our system issues an automated prescription: high dormancy detected, deploy an immediate 20% win-back code.*
>
> *Now watch our real-time 'What-If' simulator: if our retention marketing team intervenes and this customer completes a purchase today—lowering their recency to 10 days—and enrolls in our loyalty subscription program...*  
> *[Adjusting Recency slider to 10 days, selecting Subscription: Yes]*  
> *...the loaded machine learning pipeline re-evaluates their feature vector in real time, and their churn probability drops to **2.1% (Low Risk / Active)**!*
>
> *In parallel, we have exported a complete Kimball star schema and 20+ DAX measures ready for enterprise Power BI deployment.*
>
> *Thank you, and I look forward to your questions."*

---

## Part 3: Viva Voce & Technical Defense Q&A Guide

### Q1: "How did you guarantee that your churn model did not suffer from data leakage?"
- **Answer:** *"We enforced a strict temporal split. We selected July 16, 2023 as an observation cutoff date. All feature engineering—including Recency, Frequency, Monetary, AOV, and Tenure—was calculated strictly from events occurring on or before July 16. Churn was defined as a customer making zero purchases in the subsequent 91 days (through October 15, 2023). Under no circumstances did post-cutoff timestamps or transaction amounts enter the training feature matrix. Furthermore, all scaling transformations were encapsulated inside Scikit-Learn pipelines to ensure scalers were fitted only on the training folds."*

### Q2: "Why did you use both RFM and K-Means? Isn't one segmentation method enough?"
- **Answer:** *"RFM is a deterministic, heuristic rule-based framework that provides intuitive business segment labels (Champions, Loyalists, At Risk) that marketing teams readily understand. However, RFM scores metrics independently using arbitrary quantile cuts. K-Means is an unsupervised machine learning algorithm that discovers natural geometric clusters in multi-dimensional space after log-normalization. K-Means revealed the critical 'High-Value Inactive' archetype (controlling 59.8% of spend), which crosses traditional RFM boundaries and represents our greatest commercial re-activation lever."*

### Q3: "How did you handle class imbalance in churn modeling?"
- **Answer:** *"Our dataset exhibited a 75% churn to 25% retention class imbalance over the 90-day outcome window. Rather than synthetically altering or oversampling the test dataset—which would distort real-world distribution—we preserved the pristine 80/20 stratified test split and applied algorithmic cost-sensitive weighting (`class_weight='balanced'`) inside the loss functions of Logistic Regression and Random Forest. This penalized minority class misclassifications inversely proportional to their class frequency."*

### Q4: "Why did Gradient Boosting and Random Forest outperform Logistic Regression so dramatically?"
- **Answer:** *"Logistic Regression assumes a linear relationship between features and the log-odds of churning. However, customer churn is governed by sharp non-linear interactions: for example, a 60-day purchase gap is completely normal for an annual or quarterly shopper, but represents severe churn for a weekly buyer. Tree ensembles can split on non-linear interaction boundaries (e.g., `Recency > 1.5 * Avg_Gap`), capturing the true dynamics of churn where linear hyperplanes fail (AUC 0.996 vs. 0.651)."*

### Q5: "What are the limitations of your project?"
- **Answer:** *"We documented three primary limitations: First, the dataset is a cross-sectional snapshot rather than a continuous multi-year event stream, meaning we could not incorporate real-time clickstream signals like cart abandonments. Second, the data represents a single US domestic retail brand across apparel and accessories, so findings may not directly generalize to high-frequency grocery or B2B SaaS. Third, our ground truth used a static 90-day outcome window rather than a dynamic window calibrated to individual customer cadence."*
