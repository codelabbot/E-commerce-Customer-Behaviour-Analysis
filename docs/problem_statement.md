# Project Problem Statement & Objectives

## Problem Statement
Modern multi-channel retailers collect extensive customer purchasing records but often fail to translate raw transaction profiles into differentiated commercial strategies.
Without granular customer segmentation, marketing campaigns treat diverse shopper demographics uniformly, squandering promotional budget and lowering customer loyalty.
This project addresses the challenge of discovering distinct customer behavioral personas across purchase frequencies, spend amounts, product categories, and subscription preferences.
By applying rigorous data preprocessing, statistical behavioral analysis, and unsupervised machine learning clustering, we group customers into actionable cohorts.
The resulting segmentation models and commercial insights are deployed through interactive Streamlit and Power BI applications to guide targeted marketing and revenue growth.

---

## Project Objectives

1. **Data Ingestion & Quality Audit**: Ingest the raw customer shopping behavior dataset into `data/raw/`, audit dimensional schemas, evaluate null values, verify numeric distributions, and confirm absence of corrupt entries.
2. **Data Cleaning & Standardization Pipeline**: Design a reusable preprocessing module in `src/` to normalize feature naming (e.g. rename `.` to `Customer_ID`), encode categorical and ordinal features (e.g. purchase frequencies), and handle feature scaling.
3. **Exploratory Behavioral Analysis (EDA)**: Perform deep exploratory analysis into customer age distributions, spending by category, seasonal buying patterns, payment channel preferences, and review score trends across states.
4. **Behavioral Feature Engineering**: Construct high-signal segmentation attributes including customer spending tier, historical loyalty index (combining `Previous Purchases` with `Frequency of Purchases`), and discount sensitivity metrics.
5. **Machine Learning Customer Clustering**: Implement unsupervised clustering algorithms (K-Means and Hierarchical Clustering) with hyperparameter optimization (Elbow Method, Silhouette Score Analysis), maintaining reproducible seeds (`random_state=42`).
6. **Customer Persona Profiling**: Quantify cluster centroids to characterize distinct customer segments (such as High-Value Subscribers, Budget Frequent Buyers, Discount-Driven Shoppers, and Occasional High-Spenders) with strategic marketing recommendations.
7. **Interactive Streamlit Web Dashboard**: Build an interactive web dashboard enabling business users to filter customer segments, analyze demographic distributions, and predict the segment of new customer profiles in real time.
8. **Executive Power BI Dashboard**: Develop an enterprise Power BI report featuring executive KPI cards, geographic sales distribution across 50 states, category revenue breakdowns, and customer segment cohort comparisons.
