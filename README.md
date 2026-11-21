# E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI

**Academic Level:** B.Tech Major / Internship Capstone Project  
**Author:** B.Tech Analytics Internship Team  
**Stack:** Python 3.10+, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly, Streamlit, Power BI Desktop  
**Dataset:** Consumer Shopping Trends & Transaction Dataset ($N = 3,900$)  
**Status:** Completed, Validated, and Submission-Ready  

---

## 1. Executive Summary & Problem Statement

In modern multi-category and D2C e-commerce enterprises, customer acquisition costs have escalated by over 200%, making customer retention and customer lifetime value (CLV) optimization the primary drivers of sustained platform profitability.

This project delivers an end-to-end data engineering, unsupervised clustering, predictive machine learning, and business intelligence platform:
- **Analyzes 3,900 customer shopping behaviors** representing 102,771 lifetime orders and $6.15M in cumulative spend.
- **Identifies Pareto spend concentration:** The top 20% of buyers generate 37.4% of total enterprise spend, with an 83.51% established repeat customer rate.
- **Uncovers the $3.68M Dormancy Discovery:** Unsupervised K-Means clustering ($k=4$) identified a **High-Value Inactive** segment commanding **59.8% of enterprise lifetime revenue ($3.68M)** that has fallen dormant (185 days average recency).
- **Predicts Customer Churn with Zero Leakage:** Built a leak-free machine learning benchmark with a strict observation cutoff date (July 16, 2023) and 91-day outcome window. The champion **Gradient Boosting model achieved an ROC-AUC of 0.9964, F1-score of 0.9740, and Recall of 96.24%**, while Random Forest delivered **100% Precision** (zero false-positive alarms).
- **Proactively Rescues 237 Medium-Risk Accounts:** Isolates transitional accounts (30% to 70% churn risk) for low-cost re-engagement before terminal lapse.
- **Deploys Enterprise Dashboards:** Authoring a Kimball star-schema data model with 20+ DAX measures for Power BI Desktop and a multi-page **Streamlit Web Application** featuring real-time "What-If" churn simulations.

---

## 2. Repository Architecture & Folder Map

```text
project_ai_mern_image_generation/
│
├── app/
│   └── streamlit_app.py               # 4-Page interactive Streamlit web dashboard with real-time ML simulator
│
├── data/
│   ├── raw/
│   │   ├── customer_shopping_data.csv # Raw source transaction dataset (3,900 records, 19 fields)
│   │   └── customer_shopping_data.xlsx# Source Excel workbook
│   └── processed/
│       ├── clean_transactions.csv     # Validated & cleaned transaction records (N=3,900)
│       ├── customer_table.csv         # Aggregated customer-level lifetime metrics (N=3,900)
│       ├── rfm_segments.csv           # RFM quintiles, composite scores, and heuristic segment labels
│       ├── customer_segments.csv      # Unified table with RFM segments and K-Means cluster labels
│       ├── churn_predictions.csv      # ML predictions, calibrated probabilities, and risk tiers
│       ├── cancellations.csv          # Audit table for returns/cancellations
│       ├── guest_transactions.csv     # Separate file for guest transactions without customer IDs
│       └── modeling_transactions.csv  # Outlier-capped dataset (1.5x IQR) for ML stability
│
├── docs/
│   ├── problem_statement.md           # 5-line problem statement & 8 project objectives
│   ├── data_dictionary.md             # Complete schema, attribute definitions, and unavailable fields
│   ├── business_recommendations.md    # 7 high-impact commercial recommendations anchored to data
│   ├── final_report.md                # Formal, comprehensive B.Tech final internship report
│   └── presentation_and_demo.md       # 12-slide presentation outline, speaker notes, and 3-min demo script
│
├── models/
│   ├── kmeans.pkl                     # Serialized K-Means pipeline (k=4) with StandardScaler
│   └── churn_model.pkl                # Serialized champion Gradient Boosting pipeline & scaler
│
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Phase 1: Ingestion, schema inspection, missingness audit
│   ├── 02_preprocessing.ipynb         # Phase 2: Cleaning, date parsing, outlier treatment
│   ├── 03_eda.ipynb                   # Phase 3: Exploratory data analysis & 6 diagnostic figures
│   ├── 04_customer_behaviour.ipynb    # Phase 4: Customer table, Pareto curve, repurchase rate
│   ├── 05_rfm.ipynb                   # Phase 5: RFM scoring, heatmaps, treemap segmentation
│   ├── 06_kmeans.ipynb                # Phase 6: Log transformations, optimal k evaluation, 2D PCA
│   └── 07_churn_model.ipynb           # Phase 7: Leakage-free ML churn benchmark & feature importance
│
├── powerbi/
│   ├── Dashboard_cronus_to_for_project.pbix # Official Power BI Desktop dashboard file (1.64 MB)
│   ├── fact_sales.csv                 # Transactional fact table (N=3,900)
│   ├── dim_customer.csv               # Enriched customer dimension (RFM, clusters, churn)
│   ├── dim_product.csv                # Product catalog dimension (25 items, 4 categories)
│   ├── dim_date.csv                   # Continuous calendar dimension (730 days, 2022–2023)
│   ├── model.md                       # Star schema ER diagram & relationship specification
│   ├── measures.dax                   # 20+ Documented DAX measures for Power BI Desktop
│   └── dashboard_spec.md              # 4-page dashboard visual specification & 10-step build checklist
│
├── reports/
│   ├── figures/                       # 19 High-resolution analytical charts (150 DPI)
│   │   ├── 01_monthly_revenue_order_trend.png
│   │   ├── 02_day_and_hour_patterns.png
│   │   ├── 03_top_products_revenue_quantity.png
│   │   ├── 04_revenue_by_location.png
│   │   ├── 05_order_value_distribution.png
│   │   ├── 06_category_keyword_analysis.png
│   │   ├── 07_purchase_frequency_distribution.png
│   │   ├── 08_new_vs_returning_monthly.png
│   │   ├── 09_pareto_revenue_concentration.png
│   │   ├── 10_monthly_customer_growth.png
│   │   ├── 11_rfm_heatmap.png
│   │   ├── 12_rfm_segments_treemap_bar.png
│   │   ├── 13_rfm_treemap.png
│   │   ├── 14_kmeans_elbow_silhouette.png
│   │   ├── 15_kmeans_pca_2d.png
│   │   ├── 16_kmeans_vs_rfm_crosstab.png
│   │   ├── 17_churn_roc_curves.png
│   │   ├── 18_churn_confusion_matrices.png
│   │   └── 19_churn_feature_importance.png
│   └── model_evaluation.md            # Formal ML report with benchmarks and transparent limitations
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Raw dataset ingestion and schema inspection
│   ├── preprocess.py                  # Cleaning, revenue derivation, and outlier handling
│   ├── eda.py                         # Plot generation for sales and demand patterns
│   ├── customer_analytics.py          # Customer table creation, Pareto analysis, cohorts
│   ├── rfm.py                         # Recency, Frequency, Monetary scoring engine
│   ├── clustering.py                  # K-Means clustering, elbow/silhouette, PCA projection
│   ├── churn.py                       # Leakage-free ML dataset builder & training pipeline
│   └── powerbi_export.py              # Star-schema dimensional table exporter
│
├── requirements.txt                   # Pinned project dependencies
└── README.md                          # Master documentation and reproduction guide
```

---

## 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git (optional)
- Modern web browser (Chrome, Edge, Firefox)
- Power BI Desktop (Optional, for `.pbix` dashboard build)

### 3.2 Virtual Environment Setup
```bash
# 1. Clone or navigate to the repository root
cd project_ai_mern_image_generation

# 2. Create an isolated virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate

# 4. Install all pinned dependencies
pip install -r requirements.txt
```

---

## 4. How to Reproduce All Results From Scratch

The entire pipeline is modular, deterministic, and fully reproducible (`random_state=42` is pinned across all scripts). You can re-execute the entire pipeline command-by-command:

```bash
# Step 1: Preprocessing & Data Cleaning
# Generates data/processed/clean_transactions.csv, cancellations.csv, guest_transactions.csv
python src/preprocess.py

# Step 2: Exploratory Data Analysis & Visualizations
# Generates figures 01 to 06 in reports/figures/
python src/eda.py

# Step 3: Customer Behavioral Lifetime Table & Pareto Analysis
# Generates data/processed/customer_table.csv and figures 07 to 10
python src/customer_analytics.py

# Step 4: RFM Quantile Scoring & Heuristic Segmentation
# Generates data/processed/rfm_segments.csv and figures 11 to 13
python src/rfm.py

# Step 5: Unsupervised K-Means Clustering & 2D PCA Projection
# Generates models/kmeans.pkl, data/processed/customer_segments.csv, and figures 14 to 16
python src/clustering.py

# Step 6: Leakage-Free Predictive Churn Modeling Benchmark
# Generates models/churn_model.pkl, data/processed/churn_predictions.csv, and figures 17 to 19
python -c "from src.churn import run_churn_pipeline; run_churn_pipeline()"

# Step 7: Export Power BI Star-Schema Dimensional Tables
# Generates fact_sales.csv, dim_customer.csv, dim_product.csv, dim_date.csv in powerbi/
python src/powerbi_export.py
```

---

## 5. Running the Streamlit Companion Application

The web dashboard is fully standalone, reading strictly from `data/processed/` and `models/`:

```bash
# Launch Streamlit server
streamlit run app/streamlit_app.py
```

The application will launch your browser at `http://localhost:8501`.

### Application Features:
1. **📊 Overview KPIs:** Top financial metrics, dual-axis seasonal trends, department revenue shares, top 10 products, and payment/shipping matrix.
2. **🎯 Customer Segments (RFM + K-Means):** RFM spend distributions, Recency vs Frequency bubble plot, and K-Means cluster composition ($k=4$).
3. **⚠️ Churn Risk (Searchable Customer Table):** Multi-filter searchable grid with priority callout for the **237 Medium-Risk accounts** and a **1-click CSV Export button**.
4. **🔍 Customer Lookup & Real-Time Churn Simulator:** Individual Customer 360 profile card and an interactive "What-If" simulator adjusting recency, subscriptions, and discounts to re-compute machine learning churn probabilities on the fly.

---

## 6. Running the Jupyter Notebooks

All 7 analytical phases are documented in standalone Jupyter Notebooks complete with markdown explanations, code cells, pre-computed outputs, and closing key findings:

```bash
jupyter notebook
```
Navigate to `notebooks/` and execute sequentially:
- `01_data_understanding.ipynb`
- `02_preprocessing.ipynb`
- `03_eda.ipynb`
- `04_customer_behaviour.ipynb`
- `05_rfm.ipynb`
- `06_kmeans.ipynb`
- `07_churn_model.ipynb`

---

## 7. Power BI Executive Dashboard Setup

The project includes the completed, pre-built Power BI dashboard file along with all underlying star-schema tables and documentation in `powerbi/`:

1. **Directly Open Completed Dashboard:** Double-click [`powerbi/Dashboard_cronus_to_for_project.pbix`](powerbi/Dashboard_cronus_to_for_project.pbix) to launch the completed interactive report in **Power BI Desktop**.
2. **Re-build / Re-link from Scratch (Optional):**
   - Ingest `fact_sales.csv`, `dim_customer.csv`, `dim_product.csv`, `dim_date.csv` into Power BI Desktop via **Get Data -> Text/CSV**.
   - Verify relationships in Model View according to [`powerbi/model.md`](powerbi/model.md).
   - Create a `_Measures` table and copy the pre-built DAX calculations from [`powerbi/measures.dax`](powerbi/measures.dax).
   - Construct the 4-page dashboard following the exact visual layout in [`powerbi/dashboard_spec.md`](powerbi/dashboard_spec.md).

---

## 8. Summary of Machine Learning Benchmark Results

Evaluated on held-out test cohort ($N = 780$, 80/20 Stratified Split) with balanced class weights:

| Model Architecture | Precision | Recall | F1-Score | ROC-AUC | True Neg | False Pos | False Neg | True Pos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Champion)** | **0.9860** | **0.9624** | **0.9740** | **0.9964** | 187 | 8 | 22 | **563** |
| **Random Forest** | **1.0000** | 0.9470 | 0.9728 | 0.9960 | **195** | **0** | 31 | 554 |
| **Logistic Regression (Baseline)** | 0.8518 | 0.6188 | 0.7168 | 0.6508 | 132 | 63 | 223 | 362 |

### Key Feature Drivers (Random Forest Gini Share):
- **Behavioral Timing Accounts for 87.7% of Predictive Signal:** `Recency` (72.8%), `Tenure_Days` (7.5%), and `Avg_Gap_Days` (7.4%).
- **Demographics are Secondary (<2.0% Combined):** Age, gender, discounts, and subscription status contribute minimally compared to customer purchase timing.

---

## 9. Key Project Deliverables Manifest

| Document / Artifact | Location | Purpose |
| :--- | :--- | :--- |
| **Final Comprehensive Report** | [`docs/final_report.md`](docs/final_report.md) | Complete formal B.Tech internship capstone report. |
| **Strategic Business Plan** | [`docs/business_recommendations.md`](docs/business_recommendations.md) | 7 Data-anchored commercial actions ($3.68M re-activation, VIP loyalty, bundles). |
| **Presentation & Demo Guide** | [`docs/presentation_and_demo.md`](docs/presentation_and_demo.md) | 12-Slide deck outline, speaker notes, 3-min demo script, and viva defense Q&A. |
| **Streamlit Companion App** | [`app/streamlit_app.py`](app/streamlit_app.py) | Standalone interactive web dashboard with live ML simulator. |
| **Official Power BI Dashboard** | [`powerbi/Dashboard_cronus_to_for_project.pbix`](powerbi/Dashboard_cronus_to_for_project.pbix) | Complete interactive Power BI Desktop report with futuristic wireframe theme. |
| **Power BI Specification** | [`powerbi/dashboard_spec.md`](powerbi/dashboard_spec.md) | 4-Page dashboard visual specification and 10-step build checklist. |
| **DAX Measures File** | [`powerbi/measures.dax`](powerbi/measures.dax) | 20+ Production DAX measures for Power BI Desktop. |
| **Model Evaluation Report** | [`reports/model_evaluation.md`](reports/model_evaluation.md) | Detailed ML performance benchmark and honest limitations. |
| **Data Dictionary** | [`docs/data_dictionary.md`](docs/data_dictionary.md) | Complete schema, domain values, and unavailable attributes. |

---

## 10. License & Academic Attribution

- **Project:** B.Tech Major / Internship Capstone Project
- **Source Dataset:** Consumer Behavior and Shopping Trends Dataset (Public Domain / Kaggle)
- **Author:** B.Tech Analytics Internship Team
- **Academic Notice:** All work follows scientific integrity; no artificial customer attributes were fabricated.
