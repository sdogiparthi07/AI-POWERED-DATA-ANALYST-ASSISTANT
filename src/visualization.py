import pandas as pd
import plotly.express as px


def get_numeric_columns(df):
    """Get all numeric columns from the dataframe."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    """Get all categorical columns from the dataframe."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_date_columns(df):
    """Get columns that contain datetime values."""
    return df.select_dtypes(include="datetime").columns.tolist()


def create_histogram(df, column):
    """Create a histogram for a numeric column."""

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    return fig


def create_bar_chart(df, column):
    """Create a bar chart showing the most common categories."""

    data = df[column].value_counts().head(15).reset_index()

    data.columns = [column, "count"]

    fig = px.bar(
        data,
        x=column,
        y="count",
        title=f"Top Values in {column}"
    )

    return fig


def create_scatter_plot(df, x_column, y_column):
    """Create a scatter plot using two numeric columns."""

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}"
    )

    return fig


def create_line_chart(df, date_column, value_column):
    """Create a line chart showing how a value changes over time."""

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
    """Create a heatmap showing correlations between numeric columns."""

    numeric_data = df.select_dtypes(include="number")

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

    data = pd.DataFrame(results)

    fig = px.line(
        data,
        x="clusters",
        y="inertia",
        markers=True,
        title="K-Means Elbow Method"
    )

    return fig