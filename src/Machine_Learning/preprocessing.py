import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def split_features_target(df, target_column):
    """Separate features and target."""

    if target_column not in df.columns:
        raise ValueError(
            "Target column does not exist."
        )

    data = df.dropna(
        subset=[target_column]
    ).copy()

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y


def get_numeric_columns(X):
    """Get numeric columns."""

    return X.select_dtypes(
        include="number"
    ).columns.tolist()


def get_categorical_columns(X):
    """Get categorical columns."""

    return X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def get_datetime_columns(X):
    """Get datetime columns."""

    return X.select_dtypes(
        include="datetime"
    ).columns.tolist()


def remove_id_columns(X):
    """Remove columns that look like IDs."""

    X = X.copy()

    columns_to_remove = []

    for column in X.columns:

        column_name = str(
            column
        ).lower().strip()

        if (
            column_name == "id"
            or column_name.endswith("_id")
            or column_name.endswith(" id")
            or "row_id" in column_name
        ):
            columns_to_remove.append(
                column
            )

    if columns_to_remove:

        X = X.drop(
            columns=columns_to_remove
        )

    return X


def convert_datetime_columns(X):
    """Convert datetime columns into useful numeric features."""

    X = X.copy()

    datetime_columns = get_datetime_columns(X)

    for column in datetime_columns:

        X[column + "_year"] = X[column].dt.year
        X[column + "_month"] = X[column].dt.month
        X[column + "_day"] = X[column].dt.day
        X[column + "_dayofweek"] = (
            X[column].dt.dayofweek
        )

        X = X.drop(
            columns=[column]
        )

    return X


def create_preprocessor(X):
    """Create preprocessing steps for numeric and categorical data."""

    numeric_columns = get_numeric_columns(X)

    categorical_columns = get_categorical_columns(X)

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


def split_train_test(
    X,
    y,
    test_size=0.2
):
    """Split data into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )


def validate_regression_target(
    df,
    target_column
):
    """Check whether a column is suitable for regression."""

    if target_column not in df.columns:

        return (
            False,
            "Target column does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_column]
    ):

        return (
            False,
            "Regression target must be numeric."
        )

    unique_values = df[target_column].nunique()

    if unique_values < 2:

        return (
            False,
            "Target must contain at least two different values."
        )

    return (
        True,
        "Valid regression target."
    )


def validate_classification_target(
    df,
    target_column
):
    """Check whether a column is suitable for classification."""

    if target_column not in df.columns:

        return (
            False,
            "Target column does not exist."
        )

    unique_values = df[target_column].nunique()

    if unique_values < 2:

        return (
            False,
            "Target must contain at least two classes."
        )

    if unique_values > 20:

        return (
            False,
            "This column has too many unique values "
            "for a simple classification problem."
        )

    return (
        True,
        "Valid classification target."
    )