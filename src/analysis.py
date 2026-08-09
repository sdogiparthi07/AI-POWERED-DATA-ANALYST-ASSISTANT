import pandas as pd


def get_dataset_profile(df):
    """
    Generate a basic profile of the dataset.
    """

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "datetime_columns": df.select_dtypes(
            include=["datetime"]
        ).columns.tolist(),
    }

    return profile


def get_numeric_summary(df):
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    return summary


def get_categorical_summary(df):
    """
    Generate summary information for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    results = []

    for column in categorical_columns:

        results.append({
            "column": column,
            "unique_values": int(df[column].nunique()),
            "missing_values": int(df[column].isna().sum()),
            "most_frequent": (
                df[column].mode().iloc[0]
                if not df[column].mode().empty
                else None
            )
        })

    return pd.DataFrame(results)