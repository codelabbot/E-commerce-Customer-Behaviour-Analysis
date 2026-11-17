"""K-Means clustering and customer segmentation pipeline.

Implements feature log-transformation, standard scaling, k-selection evaluation
(Elbow curve & Silhouette analysis), model training, 2D PCA projection,
cluster profiling, and cross-tabulation against rule-based RFM segments.
Saves model artifacts to models/kmeans.pkl and data/processed/customer_segments.csv.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Ensure local writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/figures/.mpl")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

CLUSTER_NAME_MAP = {
    2: "High-Value Active",
    1: "High-Value Inactive",
    0: "Medium-Value",
    3: "Low-Value"
}


def prepare_rfm_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list]:
    """Apply log1p transformation and StandardScaler to Recency, Frequency, Monetary."""
    feature_cols = ["Recency", "Frequency", "Monetary"]
    X_raw = df[feature_cols].copy()
    X_log = np.log1p(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    return X_scaled, scaler, feature_cols


def evaluate_kmeans_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE
) -> tuple[list, list]:
    """Evaluate K-Means across a range of k values using Inertia and Silhouette scores."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)

    return inertias, silhouettes


def plot_elbow_and_silhouette(
    k_range: range,
    inertias: list,
    silhouettes: list,
    chosen_k: int = 4,
    output_dir: str = "reports/figures"
) -> str:
    """Plot Elbow (Inertia) curve and Silhouette scores side-by-side."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "14_kmeans_elbow_silhouette.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    k_list = list(k_range)

    # 1. Elbow Curve (Inertia / WCSS)
    axes[0].plot(k_list, inertias, marker="o", color="#2b5c8f", linewidth=2.2, markersize=7)
    axes[0].axvline(chosen_k, color="red", linestyle="--", linewidth=1.8, label=f"Chosen k={chosen_k}")
    axes[0].set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=11)
    axes[0].set_ylabel("Inertia (WCSS)", fontsize=11)
    axes[0].set_xticks(k_list)
    axes[0].legend(fontsize=10)

    # 2. Silhouette Score Curve
    axes[1].plot(k_list, silhouettes, marker="s", color="#1b9e77", linewidth=2.2, markersize=7)
    axes[1].axvline(chosen_k, color="red", linestyle="--", linewidth=1.8, label=f"Chosen k={chosen_k} (Score: {silhouettes[k_list.index(chosen_k)]:.3f})")
    axes[1].set_title("Silhouette Coefficient vs. Cluster Count (k)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=11)
    axes[1].set_ylabel("Mean Silhouette Coefficient", fontsize=11)
    axes[1].set_xticks(k_list)
    axes[1].legend(fontsize=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def fit_kmeans(
    X_scaled: np.ndarray,
    k: int = 4,
    random_state: int = RANDOM_STATE
) -> tuple[KMeans, np.ndarray]:
    """Fit final KMeans clustering model on scaled features."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def get_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    label_col: str = "Cluster_Label"
) -> pd.DataFrame:
    """Compute aggregate profile statistics for each discovered K-Means cluster."""
    profile = df.groupby([cluster_col, label_col]).agg(
        Customer_Count=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    total_customers = len(df)
    total_revenue = profile["Total_Revenue"].sum()

    profile["Customer_Share_%"] = (profile["Customer_Count"] / total_customers * 100).round(2)
    profile["Revenue_Share_%"] = (profile["Total_Revenue"] / total_revenue * 100).round(2)
    profile["Avg_Recency"] = profile["Avg_Recency"].round(1)
    profile["Avg_Frequency"] = profile["Avg_Frequency"].round(1)
    profile["Avg_Monetary"] = profile["Avg_Monetary"].round(2)
    profile["Total_Revenue"] = profile["Total_Revenue"].round(2)

    return profile.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label_map: dict,
    output_dir: str = "reports/figures",
    random_state: int = RANDOM_STATE
) -> tuple[str, PCA]:
    """Project scaled 3D RFM features into 2D PCA space and plot cluster scatter."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "15_kmeans_pca_2d.png")

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    var_1, var_2 = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels
    pca_df["Cluster_Label"] = pca_df["Cluster"].map(label_map)

    # Compute centroids in PCA space
    centroids_pca = pca_df.groupby("Cluster_Label")[["PC1", "PC2"]].mean().reset_index()

    plt.figure(figsize=(10, 7), dpi=150)
    palette = {"High-Value Active": "#2ca02c", "High-Value Inactive": "#d62728",
               "Medium-Value": "#1f77b4", "Low-Value": "#9467bd"}

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster_Label",
        palette=palette,
        alpha=0.6,
        s=35,
        edgecolor="none"
    )

    # Plot Centroids
    plt.scatter(
        centroids_pca["PC1"],
        centroids_pca["PC2"],
        c="black",
        s=160,
        marker="X",
        edgecolor="white",
        linewidth=2,
        label="Cluster Centroids",
        zorder=10
    )

    for _, row in centroids_pca.iterrows():
        plt.annotate(
            row["Cluster_Label"],
            (row["PC1"], row["PC2"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85)
        )

    plt.title(f"2D PCA Projection of Customer Segments\n(Explained Variance: PC1={var_1:.1f}%, PC2={var_2:.1f}% | Total: {var_1+var_2:.1f}%)",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel(f"Principal Component 1 ({var_1:.1f}% Variance - Frequency & Spend)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_2:.1f}% Variance - Recency)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Discovered Cluster")

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path, pca


def plot_crosstab_heatmap(
    df: pd.DataFrame,
    rfm_col: str = "Segment",
    cluster_col: str = "Cluster_Label",
    output_dir: str = "reports/figures"
) -> tuple[str, pd.DataFrame]:
    """Compute and visualize cross-tabulation matrix of K-Means vs. RFM segments."""
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "16_kmeans_vs_rfm_crosstab.png")

    ct = pd.crosstab(df[rfm_col], df[cluster_col])

    plt.figure(figsize=(10, 6), dpi=150)
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Customer Count"},
        linewidths=1.2,
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.title("Cross-Tabulation Matrix: Rule-Based RFM vs. Unsupervised K-Means Clusters",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("K-Means Cluster Persona", fontsize=11, labelpad=10)
    plt.ylabel("Rule-Based RFM Segment", fontsize=11, labelpad=10)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path, ct


def run_clustering_pipeline(
    df_rfm: pd.DataFrame,
    k: int = 4,
    output_csv: str = "data/processed/customer_segments.csv",
    model_pkl: str = "models/kmeans.pkl",
    figures_dir: str = "reports/figures"
) -> dict:
    """Run full unsupervised clustering pipeline, serialize model, and save outputs."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(model_pkl), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Feature Prep
    X_scaled, scaler, feature_cols = prepare_rfm_features(df_rfm)

    # 2. Evaluate k=2..10
    k_range = range(2, 11)
    inertias, silhouettes = evaluate_kmeans_k(X_scaled, k_range)
    elbow_fig = plot_elbow_and_silhouette(k_range, inertias, silhouettes, chosen_k=k, output_dir=figures_dir)

    # 3. Fit Final Model
    km, labels = fit_kmeans(X_scaled, k=k)

    # 4. Annotate DataFrame
    df_out = df_rfm.copy()
    df_out["Cluster"] = labels
    df_out["Cluster_Label"] = df_out["Cluster"].map(CLUSTER_NAME_MAP)

    # 5. Profiles & PCA
    profiles = get_cluster_profiles(df_out)
    pca_fig, pca = plot_pca_clusters(X_scaled, labels, CLUSTER_NAME_MAP, output_dir=figures_dir)

    # 6. Cross-tabulation
    rfm_col = "Segment" if "Segment" in df_out.columns else "RFM_Segment"
    crosstab_fig, ct_matrix = plot_crosstab_heatmap(df_out, rfm_col=rfm_col, output_dir=figures_dir)

    # 7. Save Dataset Deliverable
    # Format requested: CustomerID, R, F, M, RFM segment, cluster, cluster label
    export_cols = ["CustomerID", "Recency", "Frequency", "Monetary", rfm_col, "Cluster", "Cluster_Label"]
    other_cols = [c for c in df_out.columns if c not in export_cols]
    df_export = df_out[export_cols + other_cols].copy()
    df_export = df_export.rename(columns={
        "Recency": "R",
        "Frequency": "F",
        "Monetary": "M",
        rfm_col: "RFM_Segment",
        "Cluster": "cluster",
        "Cluster_Label": "cluster_label"
    })
    df_export.to_csv(output_csv, index=False)
    print(f"Saved customer segments dataset to: {output_csv} ({len(df_export):,} rows)")

    # 8. Save Serialized Model Package
    model_artifact = {
        "kmeans": km,
        "scaler": scaler,
        "pca": pca,
        "k": k,
        "feature_cols": feature_cols,
        "cluster_name_map": CLUSTER_NAME_MAP
    }
    joblib.dump(model_artifact, model_pkl)
    print(f"Serialized KMeans model and scaler to: {model_pkl}")

    return {
        "dataset": df_export,
        "profile_table": profiles,
        "crosstab_matrix": ct_matrix,
        "k_eval": {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes},
        "elbow_fig": elbow_fig,
        "pca_fig": pca_fig,
        "crosstab_fig": crosstab_fig,
        "model_file": model_pkl,
        "csv_file": output_csv
    }
