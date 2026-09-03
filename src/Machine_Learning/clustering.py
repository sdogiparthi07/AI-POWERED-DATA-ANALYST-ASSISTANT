import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def get_clustering_columns(df):
    """Get useful numeric columns for clustering."""

    columns = []

    for column in df.select_dtypes(
        include="number"
    ).columns:

        name = str(column).lower().strip()

# Exclude identifier like columns
        if (
            name == "id"
            or name.endswith("_id")
            or name.endswith(" id")
            or "row_id" in name
            or "postal" in name
            or "zip" in name
            or "pincode" in name
            or "pin_code" in name
        ):
            continue

# Exclude constant columns
        if df[column].nunique() <= 1:
            continue

# Exclude almost unique numeric identifier columns
        unique_ratio = (
            df[column].nunique() / len(df)
        )

        if unique_ratio > 0.95:
            continue

        columns.append(column)

    return columns


def prepare_clustering_data(df):
    """Prepare numeric data for clustering."""

    columns = get_clustering_columns(df)

    if len(columns) < 2:
        raise ValueError(
            "At least two useful numeric columns "
            "are required for clustering."
        )

    data = df[columns].copy()

    for column in columns:

        if data[column].isna().sum() > 0:

            median = data[column].median()

            if pd.notna(median):
                data[column] = data[column].fillna(
                    median
                )

    return data


def train_kmeans(df, n_clusters=3):
    """Train a K-Means clustering model."""

    numeric_data = prepare_clustering_data(df)

    if len(numeric_data) < n_clusters:
        raise ValueError(
            "Number of clusters cannot be greater "
            "than the number of rows."
        )

    if n_clusters < 2:
        raise ValueError(
            "Number of clusters must be at least 2."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_data
    )

    # Use a sample for training on very large datasets
    if len(scaled_data) > 5000:
        training_data = scaled_data[:5000]
    else:
        training_data = scaled_data

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=5
    )

    model.fit(
        training_data
    )

    clusters = model.predict(
        scaled_data
    )

    # Create result dataframe
    result = numeric_data.copy()
    result["cluster"] = clusters

    # Use a sample for silhouette score on large datasets
    score_sample_size = min(
        3000,
        len(scaled_data)
    )

    score_data = scaled_data[
        :score_sample_size
    ]

    score_clusters = clusters[
        :score_sample_size
    ]

    if len(set(score_clusters)) >= 2:

        score = silhouette_score(
            score_data,
            score_clusters
        )

    else:
        score = 0.0

    return (
        model,
        scaler,
        result,
        score
    )

def calculate_elbow(df):
    """Calculate K-Means inertia for different cluster counts."""

    numeric_data = prepare_clustering_data(df)

    if len(numeric_data) < 3:
        raise ValueError(
            "At least three rows are required "
            "for clustering."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_data
    )
    if len(scaled_data) > 5000:
      scaled_data = scaled_data[:5000].copy()
    else:
       training_data = scaled_data

# Sample large datasets to make elbow calculation faster
    if len(scaled_data) > 5000:

        scaled_data = scaled_data[:5000]

    max_clusters = min(
        7,
        len(scaled_data) - 1
    )

    results = []

    for k in range(
        2,
        max_clusters + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=5
        )

        model.fit(
            scaled_data
        )

        results.append({
            "clusters": k,
            "inertia": model.inertia_
        })

    return results