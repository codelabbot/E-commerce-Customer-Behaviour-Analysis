# Machine Learning Model Evaluation Report: Customer Churn Prediction

**Project Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI  
**Phase:** Phase 7 — Predictive Machine Learning Modeling  
**Author:** B.Tech Internship Analytics Team  
**Dataset:** Consumer Behavior / Retail Transaction Dataset ($N = 3,900$)  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Customer churn prediction is a critical capability for e-commerce platforms seeking to preserve customer lifetime value and optimize retention marketing spend. In Phase 7 of this project, we designed and deployed a machine learning pipeline to forecast the probability that an active customer will churn—defined mathematically as making **zero purchases in a 90-day forward-looking outcome window**.

To uphold academic and industrial standards, this experiment strictly enforces **data leakage avoidance** by isolating all feature engineering to an observation period prior to a defined cutoff date (**July 16, 2023**), while evaluating actual churn over the subsequent 91 calendar days (**July 16, 2023 to October 15, 2023**).

Three distinct machine learning paradigms were trained, scaled inside encapsulated pipelines, and benchmarked on a stratified 20% test holdout ($N = 780$):
1. **Logistic Regression (Baseline)**: An interpretable linear classifier with inverse class weighting.
2. **Random Forest Classifier**: A non-linear bagging ensemble of 100 balanced decision trees.
3. **Gradient Boosting Classifier**: A sequential boosting ensemble optimizing log-loss gradients.

### Core Findings:
- **Champion Model**: **Gradient Boosting Classifier** achieved the highest overall performance with an **ROC-AUC of 0.9964**, an **F1-score of 0.9740**, a **Recall of 96.24%**, and a **Precision of 98.60%**.
- **High-Precision Alternative**: **Random Forest Classifier** achieved an **ROC-AUC of 0.9960** with **100.0% Precision** (zero false positive alarms) and a **Recall of 94.70%** (F1 = 0.9728).
- **Linear Baseline Limitation**: Logistic Regression achieved an **ROC-AUC of 0.6508** and Recall of **61.88%**, demonstrating that customer attrition is governed by complex non-linear interactions between recency, cadence, and customer tenure that cannot be captured by linear hyperplanes.
- **Primary Driver**: Behavioral **Recency** represents **72.8%** of the model's total predictive power, followed by **Customer Tenure** (7.5%) and **Average Gap Between Orders** (7.4%). Demographic attributes and discounts contribute less than 2% combined.

---

## 2. Problem Formulation & Leakage Avoidance Architecture

Data leakage is the most frequent point of failure in real-world churn forecasting. Leakage occurs when signals from the future outcome window inadvertently bleed into feature engineering, creating artificially inflated offline test metrics that collapse in production.

```
Observation / Feature Extraction Window           Outcome Window (Target Definition)
==============================================|=========================================
Start: Jan 1, 2023                            | Cutoff Date: July 16, 2023
Historical features extracted:                | Ground Truth:
- Recency (days prior to cutoff)              | Did customer make ANY purchase
- Historical Frequency & Monetary             | between July 16 and Oct 15, 2023 (91 days)?
- AOV, Tenure, Average Order Gap              | Yes -> y = 0 (Retained / Active)
- Rating, Demographics                        | No  -> y = 1 (Churned)
==============================================|=========================================
```

### Strict Leakage Avoidance Rules Enforced:
1. **Cutoff Date**: Fixed at **July 16, 2023**.
2. **Evaluation Duration**: Spans **91 calendar days** ($\ge 90$ days) through **October 15, 2023**.
3. **Pre-Cutoff Feature Independence**: All feature attributes are derived strictly on or before July 16, 2023. Under no circumstances are post-cutoff purchase dates, order values, or activity timestamps utilized in feature generation.
4. **Target Formulation**:
   $$\text{Churn}_i = \begin{cases} 1 & \text{if customer } i \text{ completed 0 transactions between 2023-07-16 and 2023-10-15} \\ 0 & \text{if customer } i \text{ completed } \ge 1 \text{ transaction between 2023-07-16 and 2023-10-15} \end{cases}$$

---

## 3. Dataset Profile & Feature Engineering

### 3.1 Class Balance
Across the 3,900 customer population:
- **Churned ($y = 1$)**: 2,925 customers (**75.00%**)
- **Retained ($y = 0$)**: 975 customers (**25.00%**)
- **Total Population ($N$)**: 3,900 customers

The 75% churn rate naturally reflects the retail reality of low-frequency apparel and consumer product purchasing, where an individual whose cadence is quarterly or annual does not necessarily purchase within every rolling 90-day window.

### 3.2 Feature Matrix (11 Input Dimensions)
All features are computed strictly from the observation window:
| Feature | Type | Source & Derivation | Business Meaning |
| :--- | :--- | :--- | :--- |
| `Recency` | Integer | Days between latest pre-cutoff purchase and July 16, 2023 | Days of inactivity prior to the decision point |
| `Frequency` | Integer | Total historical previous transactions completed | Lifetime customer transaction depth |
| `AOV` | Float | Average order value per transaction (USD) | Customer spending caliber per visit |
| `Monetary` | Float | Cumulative historical spending (`Frequency * AOV`) | Total historical financial contribution |
| `Tenure_Days` | Integer | Estimated relationship span (`active_months * 30.4375`) | Longevity and maturity of account relationship |
| `Avg_Gap_Days` | Integer | Expected purchase cadence mapped from shopping frequency | Expected natural inter-purchase gap |
| `Review_Rating` | Float | Product review score submitted by customer (1.0 - 5.0) | Sentiment and satisfaction proxy |
| `Age` | Integer | Customer chronological age | Demographic age bracket |
| `Is_Subscriber` | Binary | 1 if active loyalty subscriber; 0 otherwise | Formal loyalty commitment |
| `Discount_Applied` | Binary | 1 if discounts were utilized; 0 otherwise | Price sensitivity and deal-seeking behavior |
| `Gender_Male` | Binary | 1 if Male; 0 if Female | Demographic gender attribute |

---

## 4. Model Training & Pipeline Architecture

### 4.1 Stratified Train/Test Partitioning
To preserve the 75% : 25% class balance without distortion, we executed an 80/20 stratified split:
- **Training Cohort ($N = 3,120$)**: 2,340 Churned (75.0%), 780 Retained (25.0%)
- **Test Holdout ($N = 780$)**: 585 Churned (75.0%), 195 Retained (25.0%)
- All random state seeds pinned to `random_state = 42`.

### 4.2 In-Pipeline Scaling & Imbalance Handling
To prevent test data leakage during normalization, all feature preprocessing is encapsulated inside `sklearn.pipeline.Pipeline` objects:
- `StandardScaler()` is fitted strictly on the training partition and applied blindly to the test holdout.
- Rather than synthetically oversampling or undersampling the pristine test set, class imbalance was handled natively inside model loss functions via `class_weight='balanced'`.

```
Pipeline Flow:
Raw Feature Vector -> [StandardScaler (Fit on Train Only)] -> [Classifier (Balanced Loss)] -> Prediction
```

---

## 5. Model Evaluation & Benchmark Results

### 5.1 Out-of-Sample Performance Comparison ($N = 780$ Test Set)

| Model Name | Precision | Recall | F1-Score | ROC-AUC | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9860** | **0.9624** | **0.9740** | **0.9964** | 187 | 8 | 22 | **563** |
| **Random Forest** | **1.0000** | 0.9470 | 0.9728 | 0.9960 | **195** | **0** | 31 | 554 |
| **Logistic Regression (Baseline)** | 0.8518 | 0.6188 | 0.7168 | 0.6508 | 132 | 63 | 223 | 362 |

### 5.2 Receiver Operating Characteristic (ROC) & AUC
The comparative ROC analysis confirms that both non-linear ensemble architectures achieve stellar discrimination power:
- **Gradient Boosting**: $\text{AUC} = 0.9964$
- **Random Forest**: $\text{AUC} = 0.9960$
- **Logistic Regression**: $\text{AUC} = 0.6508$

![Comparative ROC Curves](figures/17_churn_roc_curves.png)

### 5.3 Confusion Matrix Diagnostics

![Side-by-Side Confusion Matrices](figures/18_churn_confusion_matrices.png)

#### Operational Trade-Off Analysis:
1. **Random Forest (Precision Champion)**:
   - $\text{FP} = 0$: Zero false alarms. Not a single active customer was mistakenly flagged as churning.
   - $\text{FN} = 31$: Missed 31 churners (captured 554 of 585).
   - **Business Use Case**: Ideal when the retention action incurs significant marginal expense (e.g., offering a \$25 cash discount or gift card), where offering discounts to customers who were going to buy anyway causes severe margin erosion.
2. **Gradient Boosting (Recall Champion)**:
   - $\text{FP} = 8$: Flagged only 8 active customers unnecessarily.
   - $\text{FN} = 22$: Successfully captured 563 of 585 churners (recovering 9 additional high-risk customers compared to Random Forest).
   - **Business Use Case**: Ideal when retention intervention is low-cost (e.g., automated email re-engagement or push notification), where capturing every possible at-risk customer outweighs minimal message duplication.

---

## 6. Feature Importance & Interpretability

![Random Forest Feature Importance](figures/19_churn_feature_importance.png)

### Quantitative Ranking (Random Forest Gini Importance)
| Feature Name | Gini Importance | Relative Contribution | Plain-English Business Meaning |
| :--- | :---: | :---: | :--- |
| `Recency` | **0.7280** | **72.8%** | Days since last purchase prior to cutoff. Long dormancy is the single greatest precursor to complete churn. |
| `Tenure_Days` | **0.0749** | **7.5%** | Account longevity. Established customers have deeper behavioral inertia and lower baseline churn risk. |
| `Avg_Gap_Days` | **0.0738** | **7.4%** | Baseline cadence. A 45-day lull is normal for a quarterly buyer, but represents severe churn for a bi-weekly buyer. |
| `Monetary` | **0.0340** | **3.4%** | Total cumulative spend. Higher-value accounts demonstrate higher relationship resilience. |
| `AOV` | **0.0330** | **3.3%** | Average basket size. Reflects purchase capacity and category preference. |
| `Frequency` | **0.0223** | **2.2%** | Number of historical transactions completed. |
| `Review_Rating` | **0.0154** | **1.5%** | Customer satisfaction score. Dissatisfied customers exhibit elevated attrition propensity. |
| `Age` | **0.0126** | **1.3%** | Customer age bracket. |
| `Discount_Applied` | 0.0028 | 0.3% | Coupon usage behavior. |
| `Gender_Male` | 0.0018 | 0.2% | Gender indicator. |
| `Is_Subscriber` | 0.0013 | 0.1% | Active subscription flag. |

### Key Business Insights:
1. **The Primacy of Timing**: Churn is overwhelmingly a function of **time since last action relative to expected cadence** (`Recency` + `Avg_Gap_Days` + `Tenure_Days` comprise **87.7%** of total model importance).
2. **Demographics Are Secondary**: Age, gender, discounts, and subscriptions account for less than **2%** combined. Marketing teams often spend substantial effort segmenting by demographics, but our empirical findings prove that **behavioral recency signals are 36x more predictive than demographic attributes**.

---

## 7. Customer Churn Risk Segmentation

Using the production Gradient Boosting pipeline, we scored all 3,900 customers and mapped their calibrated churn probabilities into three operational risk tiers (saved to `data/processed/churn_predictions.csv`):

| Risk Tier | Probability Range | Customer Count | Population Share | Recommended Operational Retention Playbook |
| :--- | :---: | :---: | :---: | :--- |
| **Low Risk (Active)** | $0.00 \le P < 0.30$ | **851** | **21.8%** | Do not offer margin-eroding discounts. Deliver standard loyalty points, new product releases, and cross-sell recommendations. |
| **Medium Risk** | $0.30 \le P \le 0.70$ | **237** | **6.1%** | **Highest Priority Intervention Group.** These customers are transitioning toward dormancy. Send timely cadence-based reminders, "We miss you" personalized nudges, or small incentives. |
| **High Risk (Churning)** | $0.70 < P \le 1.00$ | **2,812** | **72.1%** | Substantial dormancy has already occurred. Deploy aggressive win-back campaigns, major promotional discounts (e.g., 25% off), or customer service survey inquiries to diagnose dissatisfaction. |

---

## 8. Honest Discussion of Limitations

To satisfy academic rigor and ensure sound industrial deployment, we document the following known limitations of this modeling phase:

### 8.1 Dataset Snapshot Architecture vs. Continuous Event Streams
- **Limitation**: The underlying dataset represents a cross-sectional consumer transaction snapshot ($N = 3,900$) containing one observed primary transaction alongside summarized historical counters (`Previous_Purchases`, `Frequency_of_Purchases`).
- **Impact**: While our observation cutoff and outcome window successfully model a production-style time-split evaluation, real-world e-commerce platforms have access to continuous event streams (e.g., session browse events, abandoned carts, email clicks, site search queries). Incorporating web session frequency would further enrich the feature space.

### 8.2 The "Recency Dominance" Effect
- **Limitation**: `Recency` accounts for 72.8% of feature importance. Because churn is defined temporally over a 90-day window, pre-cutoff recency naturally correlates strongly with post-cutoff inactivity.
- **Impact**: In operational production systems, waiting until a customer has been dormant for 90 days before triggering an alert means the customer has already practically churned. Future model iterations should define "pre-churn early warning" targets (e.g., probability of exceeding $1.5\times$ typical gap days) to enable earlier proactive intervention.

### 8.3 Homogeneous Geographic and Product Diversity
- **Limitation**: In the provided consumer behavior dataset, all transactions are domestic (United States) and each customer record is linked to an individual product category purchase.
- **Impact**: In global multi-category e-commerce (such as the UCI Online Retail dataset), customers purchase across dozens of distinct categories and countries. In such environments, features like `Number_of_Distinct_Categories` and `International_Flag` provide additional predictive variance that was not present in this single-country dataset.

### 8.4 Fixed 90-Day Outcome Window Horizon
- **Limitation**: The 90-day outcome window treats all customers with the same time threshold, regardless of their intrinsic purchase cycle.
- **Impact**: An annual shopper (e.g., winter coat buyer) who does not purchase between July and October is technically labeled as "churned" ($y = 1$) under a fixed 90-day window, even though their natural repurchase cycle has not yet elapsed. A dynamic, customer-cadence-adjusted churn definition ($\text{Days Since Last Purchase} > 2 \times \text{Avg Gap}$) represents an impactful avenue for future research.

---

## 9. Serialization & Artifact Manifest

All Phase 7 code, models, data tables, and figures have been generated and serialized to disk:

| Artifact Path | Description | Format / Size |
| :--- | :--- | :--- |
| `src/churn.py` | Complete reusable churn training, evaluation, plotting, and inference module | Python 3 Module (342 lines) |
| `notebooks/07_churn_model.ipynb` | Comprehensive interactive Jupyter Notebook with step explanations and outputs | Jupyter Notebook (23 cells) |
| `models/churn_model.pkl` | Serialized production Gradient Boosting pipeline, scaler, feature names, and metadata | Joblib Binary (394 KB) |
| `data/processed/churn_predictions.csv` | Full customer scoring dataset with Actual, Predicted, Probability, and Risk Tiers | CSV Dataset (3,900 rows, 16 cols) |
| `reports/figures/17_churn_roc_curves.png` | High-resolution comparative ROC curves across all evaluated models | PNG Image (150 DPI) |
| `reports/figures/18_churn_confusion_matrices.png` | High-resolution side-by-side confusion matrices with classification counts | PNG Image (150 DPI) |
| `reports/figures/19_churn_feature_importance.png` | Horizontal bar chart of Random Forest feature importances with percentage labels | PNG Image (150 DPI) |
| `reports/model_evaluation.md` | Formal, honest model evaluation report detailing setup, benchmarks, and limitations | Markdown Document |

---

## 10. Conclusion & Readiness for Next Phases

Phase 7 successfully satisfies all syllabus and technical requirements:
1. Formulated a leak-free customer churn definition using a verified cutoff date with 91 days of subsequent evaluation.
2. Engineered 11 pre-cutoff features and handled class imbalance through balanced loss weighting.
3. Evaluated three models: Gradient Boosting achieved **0.9964 ROC-AUC**, Random Forest achieved **1.0000 Precision**, and Logistic Regression provided the baseline.
4. Serialized the champion model pipeline to `models/churn_model.pkl` and exported risk predictions for all 3,900 customers.
5. Documented results, operational recommendations, and limitations in this comprehensive report.

With customer segmentation (Phase 5 & 6) and customer churn prediction (Phase 7) fully completed, the project is primed for:
- **Phase 8**: Interactive Web Application (Streamlit) for real-time customer lookup, segmentation exploration, and churn risk scoring.
- **Phase 9**: Executive Power BI Dashboard and reporting artifacts.
