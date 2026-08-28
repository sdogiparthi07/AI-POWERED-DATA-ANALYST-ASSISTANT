from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def train_kmeans(df, n_clusters=3):
    """Train a K-Means clustering model."""

    numeric_data = df.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    numeric_data = numeric_data.dropna()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_data
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(
        scaled_data
    )

    score = silhouette_score(
        scaled_data,
        clusters
    )

    result = numeric_data.copy()

    result["cluster"] = clusters

    return model, scaler, result, score


def calculate_elbow(df):
    """Calculate inertia for different cluster values."""

    numeric_data = df.select_dtypes(
        include="number"
    )

    numeric_data = numeric_data.dropna()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_data
    )

    results = []

    for k in range(2, 8):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(scaled_data)

        results.append({
            "clusters": k,
            "inertia": model.inertia_
        })

    return results