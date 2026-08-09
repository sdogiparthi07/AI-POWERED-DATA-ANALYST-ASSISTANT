import pandas as pd


def clean_column_names(df):
    """
    Standardize column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )

    return df


def remove_duplicate_rows(df):
    """
    Remove duplicate rows.
    """

    df = df.copy()

    duplicates_removed = df.duplicated().sum()

    df = df.drop_duplicates()

    return df, int(duplicates_removed)


def handle_missing_values(df):
    """
    Handle missing values based on column data type.
    """

    df = df.copy()

    missing_before = int(df.isna().sum().sum())

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    # Numeric columns → median
    for column in numeric_columns:
        if df[column].isna().sum() > 0:
            df[column] = df[column].fillna(
                df[column].median()
            )

    # Categorical columns → mode
    for column in categorical_columns:
        if df[column].isna().sum() > 0:

            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(
                    mode.iloc[0]
                )
            else:
                df[column] = df[column].fillna(
                    "Unknown"
                )

    missing_after = int(df.isna().sum().sum())

    return df, missing_before, missing_after

def detect_and_convert_dates(df):
    """
    Detect columns that are likely to contain dates
    and convert them to datetime format.
    """

    df = df.copy()

    date_columns = []

    for column in df.columns:

        if df[column].dtype == "object":

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.80:
                df[column] = converted
                date_columns.append(column)

    return df, date_columns

def clean_dataset(df):

    df = df.copy()

    original_rows = len(df)

    # 1. Standardize column names
    df = clean_column_names(df)

    # 2. Remove duplicate rows
    df, duplicates_removed = remove_duplicate_rows(df)

    # 3. Handle missing values
    (
        df,
        missing_before,
        missing_after
    ) = handle_missing_values(df)

    # 4. Detect and convert date columns
    df, date_columns = detect_and_convert_dates(df)

    cleaned_rows = len(df)

    cleaning_report = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "date_columns": date_columns,
        "columns": len(df.columns)
    }

    return df, cleaning_report