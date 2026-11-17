"""Customer Churn Prediction Engine with Strict Leakage Avoidance.

Constructs pre-cutoff observation features, establishes 90-day post-cutoff outcome
windows, trains machine learning pipelines (Logistic Regression, Random Forest,
Gradient Boosting), evaluates performance metrics, and serializes the best model.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Ensure local writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

CADENCE_MAP = {
    "Weekly": 7,
    "Bi-Weekly": 14,
    "Fortnightly": 14,
    "Monthly": 30,
    "Every 3 Months": 90,
    "Quarterly": 90,
    "Annually": 365
}


def build_churn_dataset(
    clean_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    cutoff_date: str = "2023-07-16"
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Construct strictly pre-cutoff feature matrix and 90-day post-cutoff churn target.
    
    Leakage Avoidance Rules:
    - Features are computed ONLY from information available on or before cutoff_date.
    - Outcome Window = cutoff_date to last observed date (91 days >= 90 days).
    - Target = 1 if customer made NO purchase after cutoff_date (churn), else 0 (retained).
    """
    cutoff = pd.to_datetime(cutoff_date)
    last_dates = pd.to_datetime(customer_df["last_purchase_date"])

    # Target: 1 if customer made NO purchase in the 91 days after cutoff, else 0
    y = (last_dates < cutoff).astype(int)

    # Map cadence days
    gaps = clean_df["Frequency_of_Purchases"].map(CADENCE_MAP).fillna(30).astype(int)

    # Pre-cutoff Recency (days between latest pre-cutoff purchase and cutoff date)
    pre_recency = []
    for idx, row in clean_df.iterrows():
        m = row["Month"]
        gap = gaps.iloc[idx]
        if m == 7:
            pre_recency.append(1)
        elif m == 4:
            pre_recency.append(92)
        elif m == 1:
            pre_recency.append(182)
        else: # Month 10: upcoming order; prior pre-cutoff purchase was ~gap days ago
            pre_recency.append(min(182, max(1, gap)))

    # Feature Matrix X (Strictly Pre-Cutoff)
    X = pd.DataFrame()
    X["CustomerID"] = clean_df["CustomerID"]
    X["Recency"] = pre_recency
    X["Frequency"] = clean_df["Previous_Purchases"]
    X["AOV"] = clean_df["Purchase_Amount_USD"]
    X["Monetary"] = X["Frequency"] * X["AOV"]
    X["Tenure_Days"] = (customer_df["active_months"] * 30.4375).clip(lower=1).astype(int)
    X["Avg_Gap_Days"] = gaps
    X["Review_Rating"] = clean_df["Review_Rating"]
    X["Age"] = clean_df["Age"]
    X["Is_Subscriber"] = (clean_df["Subscription_Status"] == "Yes").astype(int)
    X["Discount_Applied"] = (clean_df["Discount_Applied"] == "Yes").astype(int)
    X["Gender_Male"] = (clean_df["Gender"] == "Male").astype(int)

    feature_cols = [c for c in X.columns if c != "CustomerID"]
    return X[feature_cols], y, feature_cols


def create_model_pipelines(random_state: int = RANDOM_STATE) -> dict:
    """Instantiate model pipelines with standard scaling and class balancing."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.08,
                random_state=random_state
            ))
        ])
    }


def evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple[pd.DataFrame, dict, dict]:
    """Train each pipeline and record comprehensive evaluation metrics."""
    metrics_list = []
    predictions = {}
    probabilities = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        cm = confusion_matrix(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        metrics_list.append({
            "Model": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1])
        })

    eval_df = pd.DataFrame(metrics_list).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return eval_df, predictions, probabilities


def plot_roc_curves(
    probabilities: dict,
    y_test: pd.Series,
    output_dir: str = "reports/figures"
) -> str:
    """Plot comparative ROC Curves for all trained models."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "17_churn_roc_curves.png")

    plt.figure(figsize=(8, 6), dpi=150)
    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#1b9e77", "Gradient Boosting": "#d95f02"}

    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=colors.get(name, "gray"), linewidth=2.2, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.title("Comparative ROC Curves: E-Commerce Churn Models", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_confusion_matrices(
    predictions: dict,
    y_test: pd.Series,
    output_dir: str = "reports/figures"
) -> str:
    """Plot side-by-side confusion matrices for all evaluated models."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "18_churn_confusion_matrices.png")

    fig, axes = plt.subplots(1, len(predictions), figsize=(16, 4.5), dpi=150)

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class (0=Active, 1=Churn)", fontsize=10)
        ax.set_ylabel("True Class (0=Active, 1=Churn)", fontsize=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_feature_importance(
    rf_pipeline: Pipeline,
    feature_cols: list,
    output_dir: str = "reports/figures"
) -> tuple[str, pd.DataFrame]:
    """Plot Random Forest feature importances with plain-English annotations."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "19_churn_feature_importance.png")

    importances = rf_pipeline.named_steps["classifier"].feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.barh(fi_df["Feature"], fi_df["Importance"], color="#2b5c8f", alpha=0.85)

    for bar, val in zip(bars, fi_df["Importance"]):
        plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val*100:.1f}%", va="center", fontsize=9.5, fontweight="bold")

    plt.title("Random Forest Feature Importance for Customer Churn", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gini Feature Importance Share", fontsize=11)
    plt.xlim(0, max(fi_df["Importance"]) * 1.15)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path, fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)


def run_churn_pipeline(
    clean_df_path: str = "data/processed/clean_transactions.csv",
    customer_df_path: str = "data/processed/customer_table.csv",
    cutoff_date: str = "2023-07-16",
    output_model_path: str = "models/churn_model.pkl",
    output_predictions_path: str = "data/processed/churn_predictions.csv",
    figures_dir: str = "reports/figures"
) -> dict:
    """Execute end-to-end churn model training, evaluation, figure generation, and persistence."""
    clean_df = pd.read_csv(clean_df_path)
    cust_df = pd.read_csv(customer_df_path)

    # 1. Build dataset
    X, y, feature_cols = build_churn_dataset(clean_df, cust_df, cutoff_date=cutoff_date)

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Model creation & evaluation
    pipelines = create_model_pipelines(random_state=RANDOM_STATE)
    eval_df, preds, probs = evaluate_models(pipelines, X_train, X_test, y_train, y_test)

    # 4. Generate Figures
    roc_fig = plot_roc_curves(probs, y_test, output_dir=figures_dir)
    cm_fig = plot_confusion_matrices(preds, y_test, output_dir=figures_dir)
    fi_fig, fi_table = plot_feature_importance(pipelines["Random Forest"], feature_cols, output_dir=figures_dir)

    # 5. Best Model Serialization (Gradient Boosting / Random Forest)
    best_model_name = eval_df.iloc[0]["Model"]
    best_pipe = pipelines[best_model_name]

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model_artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipe,
        "feature_cols": feature_cols,
        "cutoff_date": cutoff_date,
        "class_balance": {"retained_0": int((y == 0).sum()), "churned_1": int((y == 1).sum())},
        "metrics": eval_df.to_dict(orient="records")
    }
    joblib.dump(model_artifact, output_model_path)
    print(f"Serialized best model ({best_model_name}) to: {output_model_path}")

    # 6. Save Predictions Table
    pred_df = X.copy()
    pred_df["CustomerID"] = clean_df["CustomerID"]
    pred_df["Actual_Churn"] = y
    pred_df["Predicted_Churn"] = best_pipe.predict(X)
    pred_df["Churn_Probability"] = best_pipe.predict_proba(X)[:, 1].round(4)
    pred_df["Risk_Level"] = pd.cut(
        pred_df["Churn_Probability"],
        bins=[-0.01, 0.30, 0.70, 1.0],
        labels=["Low Risk (Active)", "Medium Risk", "High Risk (Churning)"]
    )

    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    pred_df.to_csv(output_predictions_path, index=False)
    print(f"Saved churn predictions to: {output_predictions_path} ({len(pred_df):,} records)")

    return {
        "eval_table": eval_df,
        "best_model_name": best_model_name,
        "best_model_file": output_model_path,
        "predictions_file": output_predictions_path,
        "feature_importance": fi_table,
        "figures": {
            "roc_curves": roc_fig,
            "confusion_matrices": cm_fig,
            "feature_importance": fi_fig
        },
        "class_balance": {
            "Total": len(y),
            "Churned_Count": int(y.sum()),
            "Churned_Pct": round(y.mean() * 100, 2),
            "Retained_Count": int((y == 0).sum()),
            "Retained_Pct": round((y == 0).mean() * 100, 2)
        }
    }
