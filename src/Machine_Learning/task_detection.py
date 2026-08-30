import pandas as pd


def get_identifier_columns(df):
    """Find columns that are likely identifiers."""

    identifier_columns = []

    for column in df.columns:

        name = str(column).lower().strip()

        if (
            name == "id"
            or name.endswith("_id")
            or name.endswith(" id")
            or "row_id" in name
        ):
            identifier_columns.append(column)

    return identifier_columns


def get_feature_columns(df, target_column):
    """Get columns that can be used as features."""

    identifier_columns = get_identifier_columns(df)

    feature_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        if column in identifier_columns:
            continue

        feature_columns.append(column)

    return feature_columns


def detect_target_type(df, target_column):
    """Decide whether the target is for regression or classification."""

    if target_column not in df.columns:
        raise ValueError(
            "Target column does not exist."
        )

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError(
            "Target column does not contain any values."
        )

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    unique_values = target.nunique()

    if unique_values <= 2:
        return "classification"

    column_name = str(
        target_column
    ).lower().strip()

    classification_words = [
        "class",
        "category",
        "type",
        "status",
        "label",
        "grade",
        "rating",
        "level",
        "flag"
    ]

    for word in classification_words:

        if word in column_name:
            return "classification"

    return "regression"