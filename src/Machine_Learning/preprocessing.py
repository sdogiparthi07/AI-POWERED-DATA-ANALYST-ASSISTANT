from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def split_features_target(df, target_column):
    """
    Split the dataset into features (X) and target (y).
    """

    data = df.dropna(subset=[target_column]).copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y


def get_numeric_columns(X):
    """
    Return all numeric columns.
    """

    return X.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(X):
    """
    Return all categorical columns.
    """

    return X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


def create_preprocessor(X):
    """
    Create a preprocessing pipeline for numeric and categorical data.
    """

    numeric_columns = get_numeric_columns(X)
    categorical_columns = get_categorical_columns(X)

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                "passthrough",
                numeric_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return preprocessor


def split_train_test(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )
