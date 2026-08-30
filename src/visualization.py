import pandas as pd
import plotly.express as px


def get_numeric_columns(df):
    """Get numeric columns."""

    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def get_categorical_columns(df):
    """Get categorical columns."""

    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def get_date_columns(df):
    """Get datetime columns."""

    return [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]


def create_histogram(df, column):
    """Create a histogram for a numeric column."""

    if column not in df.columns:
        raise ValueError(
            "Selected column does not exist."
        )

    if column not in get_numeric_columns(df):
        raise ValueError(
            "Histogram requires a numeric column."
        )

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    return fig


def create_bar_chart(df, column):
    """Create a bar chart for a categorical column."""

    if column not in df.columns:
        raise ValueError(
            "Selected column does not exist."
        )

    if column not in get_categorical_columns(df):
        raise ValueError(
            "Bar chart requires a categorical column."
        )

    data = (
        df[column]
        .value_counts()
        .head(15)
        .reset_index()
    )

    data.columns = [
        column,
        "count"
    ]

    fig = px.bar(
        data,
        x=column,
        y="count",
        title=f"Top Values in {column}"
    )

    return fig


def create_scatter_plot(
    df,
    x_column,
    y_column
):
    """Create a scatter plot for two numeric columns."""

    numeric_columns = get_numeric_columns(df)

    if x_column not in numeric_columns:
        raise ValueError(
            "X-axis must be a numeric column."
        )

    if y_column not in numeric_columns:
        raise ValueError(
            "Y-axis must be a numeric column."
        )

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}"
    )

    return fig


def create_line_chart(
    df,
    date_column,
    value_column
):
    """Create a line chart over time."""

    if date_column not in get_date_columns(df):
        raise ValueError(
            "Line chart requires a date column."
        )

    if value_column not in get_numeric_columns(df):
        raise ValueError(
            "Line chart requires a numeric value column."
        )

    data = (
        df.groupby(date_column)[value_column]
        .sum()
        .reset_index()
        .sort_values(date_column)
    )

    fig = px.line(
        data,
        x=date_column,
        y=value_column,
        title=f"{value_column} Over Time"
    )

    return fig


def create_correlation_heatmap(df):
    """Create a correlation heatmap."""

    numeric_data = df.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    correlation = numeric_data.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        title="Correlation Heatmap",
        aspect="auto"
    )

    return fig


def create_elbow_plot(results):
    """Create an elbow plot for K-Means clustering."""

    if not results:
        raise ValueError(
            "No clustering results available."
        )

    data = pd.DataFrame(results)

    fig = px.line(
        data,
        x="clusters",
        y="inertia",
        markers=True,
        title="K-Means Elbow Method"
    )

    return fig