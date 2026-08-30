import pandas as pd
def get_dataset_profile(df):
    """Generate a basic profile of the dataset."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include="datetime"
    ).columns.tolist()

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns
    }

    return profile


def get_numeric_summary(df):
    """Generate descriptive statistics for numeric columns."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["missing_values"] = (
        df[numeric_df.columns]
        .isna()
        .sum()
    )

    summary["unique_values"] = (
        df[numeric_df.columns]
        .nunique()
    )

    return summary


def get_categorical_summary(df):
    """Generate summary information for categorical columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    results = []

    for column in categorical_columns:

        mode = df[column].mode()

        most_frequent = (
            mode.iloc[0]
            if not mode.empty
            else None
        )

        results.append({
            "column": column,
            "unique_values": int(
                df[column].nunique()
            ),
            "missing_values": int(
                df[column].isna().sum()
            ),
            "most_frequent": most_frequent
        })

    return pd.DataFrame(results)