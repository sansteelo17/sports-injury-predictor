"""
Player Archetype Clustering Module
===================================

This module implements a hybrid clustering approach (HDBSCAN + KMeans) to classify
players into meaningful injury risk archetypes based on their injury patterns,
workload responses, and physiological characteristics.

Archetype System Overview
-------------------------
The system identifies 5 primary archetypes:

1. **High-Risk Frequent Responder**
   - High total injuries, short gaps between injuries
   - Sensitive to repeated loading, accumulates micro-trauma
   - Key indicators: high total_injuries, low avg_days_between_injuries, high reinjury_rate

2. **Catastrophic Vulnerability Profile**
   - Severe injuries when they occur, high re-aggravation risk
   - Low tissue tolerance, vulnerable to relapse
   - Key indicators: high max_severity, high avg_severity, high pct_type_tear

3. **Load-Sensitive Variable Responder**
   - Large fluctuations in injury severity and patterns
   - Inconsistent adaptive response to workload
   - Key indicators: high severity_cv, high std_severity, variable body areas

4. **Recurrent Pattern Profile**
   - Recurring injuries to same body areas
   - Requires continuity and controlled overload
   - Key indicators: high reinjury_rate, low body_area_entropy, increasing severity_trend

5. **Stable Resilient Profile**
   - Consistent, low-severity injuries
   - Tolerates load well, predictable responder
   - Key indicators: low avg_severity, long gaps between injuries, low reinjury_rate

Usage
-----
>>> from src.feature_engineering.archetype import build_player_archetype_features
>>> from src.models.archetype import cluster_players, assign_archetype_names
>>>
>>> player_features = build_player_archetype_features(injury_df)
>>> results = cluster_players(player_features)
>>> results = assign_archetype_names(results)
>>>
>>> # Access archetypes
>>> df_with_archetypes = results["df"]
>>> print(df_with_archetypes[["name", "archetype", "archetype_confidence"]])
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import hdbscan
import matplotlib.pyplot as plt


# =====================================================================
# ARCHETYPE DEFINITIONS
# =====================================================================

ARCHETYPE_DEFINITIONS = {
    "High-Risk Frequent Responder": {
        "short_name": "High-Risk Frequent",
        "description": (
            "Accumulates micro-trauma quickly; poor recovery-to-load ratio; "
            "sensitive to repeated high-intensity sessions. These players "
            "experience frequent injuries but typically of lower severity."
        ),
        "key_characteristics": [
            "High total injury count",
            "Short intervals between injuries",
            "Often strain-type injuries",
            "Responds poorly to fixture congestion"
        ],
        "training_focus": (
            "Reduce high-intensity exposure; introduce controlled workloads; "
            "avoid consecutive heavy days; emphasize recovery modalities."
        ),
        "minutes_strategy": "Limit extended match minutes; avoid late-game fatigue exposure.",
        "risk_level": "high",
        # Feature thresholds for classification
        "indicators": {
            "total_injuries": ("high", 0.7),  # Above 70th percentile
            "avg_days_between_injuries": ("low", 0.3),  # Below 30th percentile
            "avg_severity": ("low_to_medium", 0.5),  # Below 50th percentile
        }
    },

    "Catastrophic Vulnerability Profile": {
        "short_name": "Catastrophic + Re-aggravation",
        "description": (
            "High vulnerability to severe injuries and relapse. Tissue tolerance "
            "threshold is low, with a pattern of major injuries when they occur."
        ),
        "key_characteristics": [
            "High severity when injured",
            "Prone to tears and ligament damage",
            "Long recovery periods required",
            "History of re-aggravation"
        ],
        "training_focus": (
            "Avoid overload; apply controlled, rehab-influenced microcycles; "
            "monitor RPE closely; extended warm-up protocols."
        ),
        "minutes_strategy": "Avoid full match exposure; minutes must be tightly controlled.",
        "risk_level": "critical",
        "indicators": {
            "max_severity": ("high", 0.8),
            "high_severity_rate": ("high", 0.7),
            "avg_severity": ("high", 0.7),
        }
    },

    "Load-Sensitive Variable Responder": {
        "short_name": "Moderate-Load High-Variance",
        "description": (
            "Large fluctuations in weekly load tolerance; inconsistent adaptive response. "
            "Injury patterns are unpredictable, making management challenging."
        ),
        "key_characteristics": [
            "High variability in injury severity",
            "Diverse injury types and body areas",
            "Unpredictable response to workload",
            "Requires careful monitoring"
        ],
        "training_focus": (
            "Stabilize weekly structure; reduce peaks and troughs; maintain "
            "predictable loading for several weeks; consistent recovery protocols."
        ),
        "minutes_strategy": "Manage playing time to avoid sudden increases.",
        "risk_level": "moderate",
        "indicators": {
            "severity_cv": ("high", 0.7),
            "std_severity": ("high", 0.6),
            "body_area_entropy": ("high", 0.6),
        }
    },

    "Recurrent Pattern Profile": {
        "short_name": "Moderate-Risk Recurrent",
        "description": (
            "History of recurring issues to specific body areas; sensitive to sudden "
            "reductions in chronic load; requires training continuity."
        ),
        "key_characteristics": [
            "Repeated injuries to same areas",
            "Pattern of re-injury within 60 days",
            "Low diversity in injury sites",
            "Worsening severity over time"
        ],
        "training_focus": (
            "Maintain progressive overload; reinforce previous injury sites; "
            "avoid long gaps between sessions; targeted prehab work."
        ),
        "minutes_strategy": "Keep match minutes steady without abrupt changes.",
        "risk_level": "moderate",
        "indicators": {
            "reinjury_rate": ("high", 0.7),
            "body_area_entropy": ("low", 0.4),
            "severity_trend": ("positive", 0),  # Positive slope
        }
    },

    "Stable Resilient Profile": {
        "short_name": "Low-Severity Stable",
        "description": (
            "Consistent responder with good load tolerance. When injuries occur, "
            "they are typically minor with quick recovery."
        ),
        "key_characteristics": [
            "Low injury severity",
            "Good recovery between injuries",
            "Predictable response to training",
            "High tissue tolerance"
        ],
        "training_focus": (
            "Maintain current training structure; adjust only if risk increases; "
            "can handle progressive overload well."
        ),
        "minutes_strategy": "Full availability under standard rotation.",
        "risk_level": "low",
        "indicators": {
            "avg_severity": ("low", 0.3),
            "avg_days_between_injuries": ("high", 0.7),
            "reinjury_rate": ("low", 0.3),
        }
    }
}


# =====================================================================
# 1. PREPROCESS PLAYER FEATURES
# =====================================================================

def prepare_archetype_features(df: pd.DataFrame, feature_subset: list = None):
    """
    Prepare features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Player-level feature matrix from build_player_archetype_features()
    feature_subset : list, optional
        Specific features to use for clustering. If None, uses all numeric features.
        Recommended subset for injury archetypes:
        - total_injuries, avg_severity, max_severity, std_severity
        - avg_days_between_injuries, reinjury_rate, high_severity_rate
        - severity_cv, severity_trend, body_area_entropy

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix
    scaler : StandardScaler
        Fitted scaler for later use
    feature_cols : list
        Names of features used
    """
    df = df.copy()

    if "name" in df.columns:
        df_numeric = df.drop(columns=["name"])
    else:
        df_numeric = df.copy()

    # Use subset if specified
    if feature_subset is not None:
        available_features = [f for f in feature_subset if f in df_numeric.columns]
        df_numeric = df_numeric[available_features]

    # Replace problematic values
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(df_numeric.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    return X_scaled, scaler, df_numeric.columns.tolist()


def get_recommended_clustering_features():
    """
    Return the recommended feature subset for archetype clustering.

    These features capture the key dimensions of injury risk profiles:
    - Frequency: total_injuries, avg_days_between_injuries
    - Severity: avg_severity, max_severity, high_severity_rate
    - Patterns: reinjury_rate, body_area_entropy, severity_trend
    - Variability: std_severity, severity_cv
    """
    return [
        # Core frequency metrics
        "total_injuries",
        "avg_days_between_injuries",

        # Severity metrics
        "avg_severity",
        "max_severity",
        "high_severity_rate",

        # Pattern metrics
        "reinjury_rate",
        "body_area_entropy",
        "severity_trend",

        # Variability metrics
        "std_severity",
        "severity_cv",
    ]



# =====================================================================
# 2. PRIMARY CLUSTERING → HDBSCAN
# =====================================================================

def run_hdbscan(X, min_cluster_size=5, min_samples=None):
    """
    HDBSCAN automatically finds natural density-based clusters.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    min_cluster_size : int
        Minimum cluster size (default: 5)
    min_samples : int, optional
        Minimum samples in neighborhood (default: None = min_cluster_size)

    Returns
    -------
    dict
        - model: fitted HDBSCAN model
        - labels: cluster assignments (-1 means noise)
        - silhouette: silhouette score (excluding noise)
        - n_clusters: number of clusters found
        - noise_ratio: proportion of points classified as noise
    """
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True,
        prediction_data=True  # Enable soft clustering
    )

    labels = model.fit_predict(X)

    # Compute silhouette excluding noise
    mask = labels != -1
    n_valid = mask.sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    score = silhouette_score(X[mask], labels[mask]) if n_valid > 1 and n_clusters > 1 else -1
    noise_ratio = 1 - (n_valid / len(labels)) if len(labels) > 0 else 0

    return {
        "model": model,
        "labels": labels,
        "silhouette": score,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio
    }



# =====================================================================
# 3. FALLBACK CLUSTERING → KMEANS
# =====================================================================

def run_kmeans(X, k=5):
    """
    K-Means clustering as fallback for noise points.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix
    k : int
        Number of clusters (default: 5 for 5 archetypes)

    Returns
    -------
    dict
        - model: fitted KMeans model
        - labels: cluster assignments
        - silhouette: silhouette score
        - centroids: cluster centers
    """
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1

    return {
        "model": km,
        "labels": labels,
        "silhouette": score,
        "centroids": km.cluster_centers_
    }



# =====================================================================
# 4. HYBRID ASSIGNMENT LOGIC
# =====================================================================

def apply_hybrid_cluster(hdbscan_labels, kmeans_labels):
    """
    Combine HDBSCAN and KMeans labels.

    If HDBSCAN assigned -1 (noise), use KMeans cluster instead.
    This ensures all players get an archetype assignment.

    Parameters
    ----------
    hdbscan_labels : np.ndarray
        Labels from HDBSCAN (-1 for noise)
    kmeans_labels : np.ndarray
        Labels from KMeans (all assigned)

    Returns
    -------
    final_labels : np.ndarray
        Combined labels
    assignment_source : np.ndarray
        'hdbscan' or 'kmeans' for each point (for confidence scoring)
    """
    final = []
    source = []

    for h_label, k_label in zip(hdbscan_labels, kmeans_labels):
        if h_label == -1:
            final.append(k_label)
            source.append("kmeans")
        else:
            final.append(h_label)
            source.append("hdbscan")

    return np.array(final), np.array(source)



# =====================================================================
# 5. MASTER PIPELINE: HDBSCAN + KMEANS (HYBRID)
# =====================================================================

def cluster_players(df_features: pd.DataFrame, k_fallback=5, use_recommended_features=True):
    """
    Full hybrid clustering pipeline for player archetype assignment.

    Pipeline Steps:
      1. Prepare and scale features
      2. Run HDBSCAN (primary - finds natural clusters)
      3. Run KMeans (fallback - ensures all players assigned)
      4. Combine labels using hybrid logic
      5. Compute cluster profiles for archetype naming

    Parameters
    ----------
    df_features : pd.DataFrame
        Player-level feature matrix from build_player_archetype_features()
        Must contain 'name' column and numeric features.
    k_fallback : int
        Number of clusters for KMeans fallback (default: 5)
    use_recommended_features : bool
        If True, uses curated feature subset for better archetype separation

    Returns
    -------
    dict
        - scaler: fitted StandardScaler
        - features: list of feature names used
        - hdbscan_model: fitted HDBSCAN
        - kmeans_model: fitted KMeans
        - hdbscan_silhouette: HDBSCAN silhouette score
        - kmeans_silhouette: KMeans silhouette score
        - df: DataFrame with cluster assignments
        - labels: final cluster labels
        - assignment_source: 'hdbscan' or 'kmeans' for each player
        - cluster_profiles: statistics for each cluster
        - X_scaled: scaled feature matrix (for visualization)
    """
    # Select features
    feature_subset = get_recommended_clustering_features() if use_recommended_features else None

    X_scaled, scaler, feat_cols = prepare_archetype_features(df_features, feature_subset)

    # Primary clustering
    hdb = run_hdbscan(X_scaled)

    # Fallback clustering
    kmeans = run_kmeans(X_scaled, k=k_fallback)

    # Combine labels
    final_labels, assignment_source = apply_hybrid_cluster(hdb["labels"], kmeans["labels"])

    # Add to dataframe
    df_out = df_features.copy()
    df_out["cluster"] = final_labels
    df_out["assignment_source"] = assignment_source

    # Compute cluster profiles
    cluster_profiles = compute_cluster_profiles(df_out, feat_cols)

    return {
        "scaler": scaler,
        "features": feat_cols,
        "hdbscan_model": hdb["model"],
        "kmeans_model": kmeans["model"],
        "hdbscan_silhouette": hdb["silhouette"],
        "kmeans_silhouette": kmeans["silhouette"],
        "hdbscan_n_clusters": hdb["n_clusters"],
        "hdbscan_noise_ratio": hdb["noise_ratio"],
        "df": df_out,
        "labels": final_labels,
        "assignment_source": assignment_source,
        "cluster_profiles": cluster_profiles,
        "X_scaled": X_scaled
    }


def compute_cluster_profiles(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute statistical profiles for each cluster.

    For each cluster, computes mean, std, and percentile rank of key features.
    This enables automatic archetype naming based on cluster characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments
    feature_cols : list
        Feature columns to profile

    Returns
    -------
    dict
        Cluster ID -> profile statistics
    """
    profiles = {}

    # Compute global percentiles for each feature
    global_percentiles = {}
    for col in feature_cols:
        if col in df.columns:
            global_percentiles[col] = df[col].rank(pct=True)

    for cluster_id in sorted(df["cluster"].unique()):
        mask = df["cluster"] == cluster_id
        cluster_df = df[mask]

        profile = {
            "n_players": len(cluster_df),
            "features": {}
        }

        for col in feature_cols:
            if col in df.columns:
                cluster_vals = cluster_df[col]

                profile["features"][col] = {
                    "mean": float(cluster_vals.mean()),
                    "std": float(cluster_vals.std()),
                    "median": float(cluster_vals.median()),
                    # How does this cluster compare to all players?
                    "percentile_rank": float(global_percentiles[col][mask].mean()),
                    "is_high": float(global_percentiles[col][mask].mean()) > 0.7,
                    "is_low": float(global_percentiles[col][mask].mean()) < 0.3,
                }

        profiles[cluster_id] = profile

    return profiles



# =====================================================================
# 6. ARCHETYPE NAMING AND ASSIGNMENT
# =====================================================================

def assign_archetype_names(results: dict) -> dict:
    """
    Automatically assign meaningful archetype names to clusters based on their profiles.

    This function analyzes cluster statistics and matches them to predefined
    archetype definitions based on their characteristic features.

    Parameters
    ----------
    results : dict
        Output from cluster_players()

    Returns
    -------
    dict
        Updated results with:
        - df: DataFrame with 'archetype' and 'archetype_confidence' columns
        - cluster_to_archetype: mapping of cluster IDs to archetype names
        - archetype_profiles: detailed profiles for each archetype
    """
    df = results["df"].copy()
    cluster_profiles = results["cluster_profiles"]

    # Score each cluster against each archetype definition
    cluster_to_archetype = {}
    archetype_scores = {}

    for cluster_id, profile in cluster_profiles.items():
        scores = score_cluster_against_archetypes(profile)
        archetype_scores[cluster_id] = scores

        # Get best matching archetype
        best_archetype = max(scores, key=scores.get)
        cluster_to_archetype[cluster_id] = best_archetype

    # Handle duplicates - ensure each archetype is assigned to at most one cluster
    cluster_to_archetype = resolve_archetype_conflicts(cluster_to_archetype, archetype_scores)

    # Add archetype names to dataframe
    df["archetype"] = df["cluster"].map(
        lambda c: ARCHETYPE_DEFINITIONS[cluster_to_archetype[c]]["short_name"]
    )
    df["archetype_full"] = df["cluster"].map(cluster_to_archetype)

    # Compute confidence based on assignment source and cluster membership strength
    df["archetype_confidence"] = df.apply(
        lambda row: compute_archetype_confidence(
            row, archetype_scores.get(row["cluster"], {})
        ),
        axis=1
    )

    # Build archetype profiles for output
    archetype_profiles = {}
    for cluster_id, archetype_name in cluster_to_archetype.items():
        archetype_profiles[archetype_name] = {
            **ARCHETYPE_DEFINITIONS[archetype_name],
            "cluster_stats": cluster_profiles[cluster_id],
            "n_players": cluster_profiles[cluster_id]["n_players"],
        }

    results["df"] = df
    results["cluster_to_archetype"] = cluster_to_archetype
    results["archetype_profiles"] = archetype_profiles

    return results


def score_cluster_against_archetypes(cluster_profile: dict) -> dict:
    """
    Score how well a cluster matches each archetype definition.

    Uses the indicator features defined in ARCHETYPE_DEFINITIONS to compute
    a match score based on whether the cluster's percentile ranks align
    with the expected direction (high/low) for each indicator.

    Parameters
    ----------
    cluster_profile : dict
        Profile for a single cluster from compute_cluster_profiles()

    Returns
    -------
    dict
        Archetype name -> match score
    """
    scores = {}

    for archetype_name, definition in ARCHETYPE_DEFINITIONS.items():
        score = 0
        max_score = 0
        indicators = definition.get("indicators", {})

        for feature, (direction, threshold) in indicators.items():
            max_score += 1
            feature_stats = cluster_profile["features"].get(feature, {})
            percentile = feature_stats.get("percentile_rank", 0.5)

            if direction == "high":
                if percentile >= threshold:
                    score += 1
                elif percentile >= threshold - 0.2:
                    score += 0.5
            elif direction == "low":
                if percentile <= threshold:
                    score += 1
                elif percentile <= threshold + 0.2:
                    score += 0.5
            elif direction == "low_to_medium":
                if percentile <= threshold:
                    score += 1
            elif direction == "positive":
                # For trend features, check if mean is positive
                mean_val = feature_stats.get("mean", 0)
                if mean_val > threshold:
                    score += 1

        # Normalize score
        scores[archetype_name] = score / max_score if max_score > 0 else 0

    return scores


def resolve_archetype_conflicts(initial_mapping: dict, archetype_scores: dict) -> dict:
    """
    Resolve conflicts when multiple clusters want the same archetype.

    Assigns archetypes greedily based on best match scores, ensuring
    each archetype is only used once.

    Parameters
    ----------
    initial_mapping : dict
        Initial cluster -> archetype mapping (may have duplicates)
    archetype_scores : dict
        Cluster -> {archetype -> score} mapping

    Returns
    -------
    dict
        Resolved cluster -> archetype mapping with no duplicates
    """
    _ = initial_mapping  # Used for reference, actual resolution uses scores
    used_archetypes = set()
    final_mapping = {}

    # Sort clusters by their best score (descending) to prioritize best matches
    sorted_clusters = sorted(
        archetype_scores.keys(),
        key=lambda c: max(archetype_scores[c].values()),
        reverse=True
    )

    for cluster_id in sorted_clusters:
        scores = archetype_scores[cluster_id]

        # Sort archetypes by score for this cluster
        sorted_archetypes = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)

        # Find best available archetype
        for archetype in sorted_archetypes:
            if archetype not in used_archetypes:
                final_mapping[cluster_id] = archetype
                used_archetypes.add(archetype)
                break

        # Fallback if all archetypes used (shouldn't happen with 5 archetypes and k=5)
        if cluster_id not in final_mapping:
            final_mapping[cluster_id] = sorted_archetypes[0]

    return final_mapping


def compute_archetype_confidence(row: pd.Series, archetype_scores: dict) -> str:
    """
    Compute confidence level for archetype assignment.

    Based on:
    1. Assignment source (hdbscan = higher confidence)
    2. Score difference between best and second-best archetype match

    Parameters
    ----------
    row : pd.Series
        Row from the clustered DataFrame
    archetype_scores : dict
        Archetype -> score mapping for this cluster

    Returns
    -------
    str
        Confidence level: 'very-high', 'high', 'medium', or 'low'
    """
    # Base confidence from assignment source
    source = row.get("assignment_source", "kmeans")
    base_confidence = 0.7 if source == "hdbscan" else 0.5

    # Boost based on score margin
    if archetype_scores:
        sorted_scores = sorted(archetype_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            base_confidence += margin * 0.3

    # Convert to category
    if base_confidence >= 0.85:
        return "very-high"
    elif base_confidence >= 0.7:
        return "high"
    elif base_confidence >= 0.55:
        return "medium"
    else:
        return "low"


def get_archetype_profile(archetype_name: str) -> dict:
    """
    Get the full profile for an archetype by name.

    Parameters
    ----------
    archetype_name : str
        Either short name (e.g., "High-Risk Frequent") or full name

    Returns
    -------
    dict
        Archetype profile with description, training focus, etc.
    """
    # Try direct lookup
    if archetype_name in ARCHETYPE_DEFINITIONS:
        return ARCHETYPE_DEFINITIONS[archetype_name]

    # Try matching by short name
    for full_name, profile in ARCHETYPE_DEFINITIONS.items():
        if profile["short_name"] == archetype_name:
            return profile

    # Return empty profile if not found
    return {
        "short_name": archetype_name,
        "description": "Unknown archetype profile",
        "training_focus": "Consult with sports science staff",
        "minutes_strategy": "Standard management",
        "risk_level": "unknown",
        "key_characteristics": [],
    }


def summarize_archetypes(results: dict) -> pd.DataFrame:
    """
    Create a summary table of all archetypes with their characteristics.

    Parameters
    ----------
    results : dict
        Output from assign_archetype_names()

    Returns
    -------
    pd.DataFrame
        Summary table with archetype statistics
    """
    cluster_profiles = results["cluster_profiles"]
    cluster_to_archetype = results["cluster_to_archetype"]

    summary_rows = []

    for cluster_id, archetype_name in cluster_to_archetype.items():
        profile = cluster_profiles[cluster_id]
        definition = ARCHETYPE_DEFINITIONS[archetype_name]

        row = {
            "archetype": definition["short_name"],
            "archetype_full_name": archetype_name,
            "n_players": profile["n_players"],
            "risk_level": definition["risk_level"],
        }

        # Add key feature statistics
        for feature in ["total_injuries", "avg_severity", "reinjury_rate", "avg_days_between_injuries"]:
            if feature in profile["features"]:
                row[f"{feature}_mean"] = round(profile["features"][feature]["mean"], 2)
                row[f"{feature}_pct"] = round(profile["features"][feature]["percentile_rank"], 2)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("risk_level", ascending=False)


# =====================================================================
# 7. VISUALIZATION FUNCTIONS
# =====================================================================

def plot_pca_clusters(X_scaled, labels, title="Archetype Clusters (Hybrid)",
                      archetype_names=None, figsize=(10, 8)):
    """
    Visualize clusters in 2D PCA space.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    archetype_names : dict, optional
        Mapping of cluster IDs to archetype names for legend
    figsize : tuple
        Figure size
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=figsize)

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = f"Cluster {label}"
        if archetype_names and label in archetype_names:
            # Use short name for legend
            archetype_full = archetype_names[label]
            name = ARCHETYPE_DEFINITIONS.get(archetype_full, {}).get("short_name", name)

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[colors[idx]],
            label=f"{name} (n={mask.sum()})",
            alpha=0.7,
            s=60
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

    return pca


def plot_archetype_radar(results: dict, figsize=(12, 10)):
    """
    Create radar/spider charts comparing archetype profiles.

    Parameters
    ----------
    results : dict
        Output from assign_archetype_names()
    figsize : tuple
        Figure size
    """
    cluster_profiles = results["cluster_profiles"]
    cluster_to_archetype = results["cluster_to_archetype"]

    # Features to plot
    features = [
        "total_injuries", "avg_severity", "max_severity",
        "reinjury_rate", "avg_days_between_injuries", "severity_cv"
    ]

    # Filter to available features
    available_features = []
    for f in features:
        for profile in cluster_profiles.values():
            if f in profile["features"]:
                available_features.append(f)
                break

    if len(available_features) < 3:
        print("Not enough features for radar plot")
        return

    features = available_features
    n_features = len(features)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    _, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_to_archetype)))

    for idx, (cluster_id, archetype_name) in enumerate(cluster_to_archetype.items()):
        profile = cluster_profiles[cluster_id]
        short_name = ARCHETYPE_DEFINITIONS[archetype_name]["short_name"]

        # Get percentile ranks for each feature
        values = []
        for f in features:
            if f in profile["features"]:
                values.append(profile["features"][f]["percentile_rank"])
            else:
                values.append(0.5)

        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=short_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in features], size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Archetype Feature Profiles (Percentile Ranks)", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.show()


def plot_cluster_distribution(results: dict, figsize=(10, 6)):
    """
    Plot the distribution of players across archetypes.

    Parameters
    ----------
    results : dict
        Output from assign_archetype_names()
    figsize : tuple
        Figure size
    """
    df = results["df"]

    archetype_counts = df["archetype"].value_counts()

    # Define color based on risk level
    risk_colors = {
        "critical": "#d32f2f",
        "high": "#f57c00",
        "moderate": "#fbc02d",
        "low": "#388e3c",
        "unknown": "#757575"
    }

    colors = []
    for archetype in archetype_counts.index:
        for definition in ARCHETYPE_DEFINITIONS.values():
            if definition["short_name"] == archetype:
                colors.append(risk_colors.get(definition["risk_level"], "#757575"))
                break
        else:
            colors.append("#757575")

    plt.figure(figsize=figsize)
    bars = plt.barh(archetype_counts.index, archetype_counts.values, color=colors)

    # Add count labels
    for bar, count in zip(bars, archetype_counts.values):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=10)

    plt.xlabel("Number of Players")
    plt.ylabel("Archetype")
    plt.title("Player Distribution by Archetype")
    plt.tight_layout()
    plt.show()