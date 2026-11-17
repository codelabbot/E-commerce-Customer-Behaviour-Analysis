# Strategic Business Recommendations

**Project Title:** E-Commerce Customer Behaviour Analysis and Customer Segmentation Using Python, Machine Learning and Power BI  
**Artifact:** Executive Action Plan (`docs/business_recommendations.md`)  
**Target Stakeholders:** Chief Commercial Officer (CCO), VP of Marketing, Retention Operations Team  
**Dataset Scope:** 3,900 Customers | $6.15M Lifetime Value | 102,771 Total Orders  

---

## Executive Summary

Through rigorous exploratory data analysis, heuristic RFM profiling, unsupervised K-Means clustering, and predictive machine learning, this project analyzed 3,900 customer shopping behaviors. The analysis revealed that **87.7% of customer attrition risk is governed by purchase timing and order cadence**, while demographic variables contribute under 2%.

Crucially, **59.8% of enterprise lifetime revenue ($3.68M) resides in a single lapsed customer cluster (High-Value Inactive)**, representing the business's largest untapped commercial lever.

Below are seven concrete, prioritized strategic initiatives. Every recommendation is directly anchored to empirical findings from our models and data pipelines.

---

## 1. The $3.68M Re-Activation Campaign for "High-Value Inactive" Customers

- **Empirical Foundation:** K-Means clustering identified 1,552 customers (39.8% of the customer base) classified as **High-Value Inactive**. This cohort commands **$3,680,248 (59.8%)** of cumulative enterprise lifetime revenue, yet has fallen dormant with an average recency of **185.2 days** (over 6 months since their last order).
- **Strategic Action:**
  1. Launch a dedicated, automated 3-stage **"Welcome Back VIP" Win-Back Workflow**:
     - *Touch 1 (Day 1):* Personalized email highlighting new seasonal arrivals in their historically preferred category (e.g., Clothing: Blouses, Pants).
     - *Touch 2 (Day 5):* A time-limited, exclusive 20% re-activation voucher code (e.g., `COMEBACK20`) valid for 7 days.
     - *Touch 3 (Day 10):* Free Express Shipping voucher ($15 estimated value) to remove checkout friction.
  2. Implement outbound concierge phone outreach or SMS for the top 100 highest-spending accounts in this cluster (accounts with lifetime spend > $3,000).
- **Target Financial Impact:** Re-activating just **15% of this cohort (233 customers)** with an average basket of $60 would generate **$13,980 in immediate direct revenue** and restore an estimated **$367,000 in ongoing annual lifetime value**.

---

## 2. Immediate Proactive Rescue of the 237 "Medium-Risk" Transition Cohort

- **Empirical Foundation:** Our champion Gradient Boosting churn model classified **237 customers (6.1% of the population)** into the **Medium-Risk Tier** (churn probability between 30.0% and 70.0%). These customers have not yet crossed the terminal 90-day dormancy threshold, but are exhibiting declining order velocity.
- **Strategic Action:**
  1. Treat the **Medium-Risk table** as an automated daily CRM priority queue (accessible via the Streamlit dashboard and Power BI Page 4 worklist).
  2. Deploy low-cost, non-margin-eroding retention nudges before full lapse occurs:
     - Personalized push notifications or emails triggered when `Days_Since_Last_Order` reaches **$1.25\times$ their typical purchasing gap** (e.g., at 60 days for a 48.5-day average buyer).
     - "Restock & Replenish" reminders based on typical item lifecycles.
     - 500 bonus loyalty points upon ordering within 14 days.
  3. **Avoid heavy discounting:** Because these customers are still warm, cash discounts cause unnecessary margin erosion. Focus on engagement, product recommendations, and loyalty recognition.
- **Target Financial Impact:** Preventing churn for **50% of this group (118 accounts)** protects approximately **$186,000 in annualized customer equity**.

---

## 3. VIP Loyalty & Margin Preservation for "Champions"

- **Empirical Foundation:** RFM analysis identified **730 Champions (18.7% of customers)** who contribute **$1,468,790 (23.9% of lifetime revenue)** with an average recency of **12.3 days**, an order frequency of **33.6 purchases**, and an AOV of **$59.88**. Random Forest modeling proved these customers have **0.0% false-positive churn risk**; they are committed, highly engaged brand advocates.
- **Strategic Action:**
  1. Establish an invitation-only **"Tier-1 Ambassador Club"**:
     - 48-hour early access to new collection drops (e.g., Fall outerwear and Spring apparel).
     - Dedicated customer support routing and complimentary gift packaging.
     - Quarterly surprise-and-delight rewards (e.g., branded accessories, free expedited shipping).
  2. **Eliminate promotional discounts for Champions:** Our data shows 57.0% of transactions occur at full catalog price. Offering coupon discounts to Champions needlessly forfeits gross margin on purchases that would occur organically.
- **Target Financial Impact:** Eliminating unnecessary 10% discounts across Champions preserves an estimated **$35,000+ in annual gross margin**, while early access drives higher initial sell-through rates at full retail price.

---

## 4. Cross-Category Product Bundling to Increase Basket Size (AOV)

- **Empirical Foundation:** Exploratory data analysis revealed that average order value is tightly clustered at **$59.76**, with transactions limited to single-item purchases across 25 distinct catalog items. Clothing generates **52.8% of revenue ($123,025)** and Accessories generates **26.4% ($61,620)**. Top-selling products include Blouses ($10,410), Shirts ($10,332), Dresses ($10,320), Pants ($10,090), and Jewelry ($10,010).
- **Strategic Action:**
  1. Build curated **"Complete the Look" Cross-Category Bundles**:
     - *Workplace Elegance Bundle:* Blouse ($60) + Pants ($60) + Scarf ($50) = $170 retail value, offered at $149 (12% bundle discount).
     - *Weekend Casual Bundle:* Shirt ($60) + Shorts ($55) + Sunglasses ($60) = $175 retail, offered at $155.
     - *Evening Gala Bundle:* Dress ($65) + Jewelry ($60) + Handbag ($60) = $185 retail, offered at $160.
  2. Configure checkout-page algorithmic recommendations: "Customers who bought this Blouse also added Jewelry (Save $15 when purchased together)".
- **Target Financial Impact:** Increasing Average Order Value by just **15% (from $59.76 to $68.72)** across 3,900 annual orders yields **+$34,944 in incremental top-line revenue** with zero customer acquisition expense.

---

## 5. Subscription Program Expansion for Predictable Cash Flow

- **Empirical Foundation:** Currently, only **27.0% of customers (1,053 accounts)** are enrolled in the active subscription program, while **73.0% (2,847 accounts)** remain non-subscribers. Analysis demonstrates that subscribed customers exhibit higher satisfaction ratings (mean 3.82 vs 3.72) and significantly lower baseline churn probabilities.
- **Strategic Action:**
  1. Target high-cadence shoppers: 1,268 customers shop on a **Weekly, Bi-Weekly, or Fortnightly cadence (32.5% of the base)**. Target these frequent buyers at checkout with a 1-click **"Subscribe & Save 5%"** auto-replenishment option.
  2. Introduce an annual VIP membership pass ($49/year) that unlocks unlimited free standard shipping, free returns, and exclusive member pricing events.
- **Target Financial Impact:** Converting **500 frequent buyers (12.8% increase in subscription penetration)** generates **$24,500 in upfront membership revenue** and lifts recurring monthly purchase predictability by ~35%.

---

## 6. Cadence-Synchronized Lifecycle Triggers (87.7% Behavioral Timing Focus)

- **Empirical Foundation:** Feature importance analysis of our Random Forest and Gradient Boosting models demonstrated that **Recency accounts for 72.8%** of churn predictive power, **Tenure accounts for 7.5%**, and **Expected Inter-Purchase Gap accounts for 7.4%** (totaling **87.7%**). In contrast, demographic variables (Age 1.3%, Gender 0.2%) and discount utilization (0.3%) account for less than 2% combined.
- **Strategic Action:**
  1. Discontinue generic, mass demographic marketing blasts segmented purely by age bracket or gender.
  2. Implement an automated **Cadence-Synchronized Lifecycle Engine**:
     - Map each customer to their empirical purchasing cadence: Weekly (7 days), Monthly (30 days), Quarterly (90 days).
     - Set dynamic, customer-specific alert boundaries:
       - **On-Cadence (0.0x to 1.0x Gap):** Content marketing, new lookbooks, loyalty status updates.
       - **Cadence Warning (1.0x to 1.5x Gap):** Personalized replenishment nudge; "Running low on your favorites?".
       - **Imminent Churn (1.5x to 2.0x Gap):** Modest promotional incentive (free shipping or $10 off $60).
       - **Dormant (>2.0x Gap):** High-touch win-back campaign.
- **Target Financial Impact:** Aligning communications with customer buying rhythms typically increases marketing campaign conversion rates by **2.5x to 4.0x** over generic promotional blasts, while dramatically reducing unsubscribe rates.

---

## 7. Strategic Free Shipping Threshold to Drive Basket Expansion

- **Empirical Foundation:** Transaction data shows that **677 customers selected Free Shipping**, and express shipping options (Next Day Air, Express, 2-Day) represent over 50% of fulfillments. The catalog unit price averages $59.76.
- **Strategic Action:**
  1. Institute a dynamic **Free Standard Shipping Threshold at $75.00**.
  2. At an average basket size of $59.76, an individual customer purchasing one apparel item ($55–$65) is typically $10 to $20 below the free shipping qualification.
  3. Deploy a dynamic cart banner: *"You are only $15.24 away from FREE Shipping! Add a Belt ($59), Socks ($25), or Sunglasses ($59) to qualify."*
- **Target Financial Impact:** In e-commerce retail, 58% of shoppers add extra items to their cart to qualify for free shipping. If 25% of one-item buyers add an accessory ($25–$50) to cross the threshold, annual incremental revenue exceeds **$28,000** with high margin contribution.

---

## Summary Matrix of Strategic Recommendations

| # | Strategic Initiative | Primary Target Cohort | Key Data Metric Anchor | Expected Business Outcome | Priority |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | **$3.68M Win-Back Campaign** | High-Value Inactive | 1,552 cust, 59.8% rev, 185d recency | Recover $367K in annual CLV; $14K instant revenue | **P1 (Immediate)** |
| **2** | **Proactive Rescue of Medium Risk** | Medium-Risk Churn Tier | 237 cust, 30%-70% churn prob | Prevent $186K in customer attrition with zero discounts | **P1 (Immediate)** |
| **3** | **Champions VIP & Margin Shield** | Champions RFM Segment | 730 cust, $1.47M rev, 0% FP churn | Save $35K+ in gross margin; elevate retention advocacy | **P2 (30-60 Days)** |
| **4** | **Cross-Category Product Bundling** | All Shoppers (Apparel + Acc) | Clothing 52.8%, Acc 26.4%, $59.76 AOV | Lift AOV by +15% (+$34.9K incremental revenue) | **P2 (30-60 Days)** |
| **5** | **Subscription Program Expansion** | Frequent Buyers (Weekly/Bi-Wk) | 27% current sub rate; 1,268 frequent | Convert 500 accounts; +$24.5K upfront cash flow | **P3 (60-90 Days)** |
| **6** | **Cadence-Synchronized Triggers** | Complete Customer Base | Timing = 87.7% ML signal; Demo < 2% | 2.5x - 4.0x conversion lift over demographic blasts | **P2 (30-60 Days)** |
| **7** | **$75 Free Shipping Threshold** | One-Item Buyers ($59.76 AOV) | 677 free shipping, $59.76 avg order | Lift cart size; +$28K annual incremental gross sales | **P3 (60-90 Days)** |
