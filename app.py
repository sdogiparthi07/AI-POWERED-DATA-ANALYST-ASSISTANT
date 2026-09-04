import streamlit as st
import pandas as pd
import plotly.express as px


from src.data_loader import load_csv
from src.analysis import (
    get_dataset_profile,
    get_numeric_summary,
    get_categorical_summary
)
from src.data_cleaner import clean_dataset
from src.visualization import (
    get_numeric_columns,
    get_categorical_columns,
    get_date_columns,
    create_histogram,
    create_bar_chart,
    create_scatter_plot,
    create_line_chart,
    create_correlation_heatmap,
    create_elbow_plot
)
from src.Machine_Learning.clustering import (
    train_kmeans,
    calculate_elbow,
    get_clustering_columns
)
from src.Machine_Learning.model_comparison import (
    compare_regression_models,
    compare_classification_models
        
)

from src.nlp.question_answering import answer_question
from src.GEN_AI.gemini_analyzer import analyze_dataset

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI POWERED DATA ANALYST ASSISTANT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# SESSION STATE

if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None

if "cleaning_report" not in st.session_state:
    st.session_state["cleaning_report"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None


# HELPER FUNCTIONS

def load_uploaded_dataset(uploaded_file):
    """
    Load CSV or Excel dataset based on the uploaded file type.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return load_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file type. Please upload a CSV or Excel file."
        )


def format_number(value):
    """
    Format numeric values for dashboard display.
    """

    if pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.2f}"
# SIDEBAR

with st.sidebar:

    st.markdown("## 🤖 AI POWERED DATA ANALYST ASSISTANT")

    st.caption(
        "your intelligent data analysis workspace"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📂 Data Quality",
            "📊 Dashboard",
            "📈 Visual Analytics",
            "🤖 Machine Learning",
            "💬 Ask Your Data",
            "🧠 AI Insights"
        ]
    )

    st.divider()

    if st.session_state["cleaned_df"] is not None:
        st.success(
            f"Dataset loaded: "
            f"{st.session_state['uploaded_file_name']}"
        )
    else:
        st.info(
            "📊 Your dashboard is waiting for a dataset."
        )

    st.divider()

    st.caption("AI POWERED DATA ANALYST ASSISTANT")
    st.caption("Streamlit • ML • NLP • GenAI")


# GET DATASET FROM SESSION STATE

cleaned_df = st.session_state["cleaned_df"]


# OVERVIEW PAGE

if page == "🏠 Overview":

    st.title("🤖 AI POWERED DATA ANALYST ASSISTANT")

    st.subheader(
        "Ask questions. Discover insights."
    )

    st.divider()

    st.markdown("### 👋 Get Started")

    st.write(
        "Upload a CSV or Excel dataset to begin the analysis workflow. "
        "The application will clean and prepare the dataset before making "
        "it available across the different analysis sections."
    )

    # DATASET UPLOAD

    st.markdown("### 📂 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload any tabular CSV or Excel dataset."
    )

    if uploaded_file is not None:

        # Avoid unnecessary reprocessing of the same file
        if (
            st.session_state["uploaded_file_name"]
            != uploaded_file.name
        ):

            try:

                with st.spinner("Loading dataset..."):
                    df = load_uploaded_dataset(
                        uploaded_file
                    )

                with st.spinner(
                    "Cleaning and preparing your dataset..."
                ):
                    cleaned_df, cleaning_report = clean_dataset(
                        df
                    )

                # Store processed dataset in session state
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_report"] = cleaning_report
                st.session_state["uploaded_file_name"] = uploaded_file.name

                st.success(
                    "Dataset uploaded and cleaned successfully!"
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to process the dataset: {error}"
                )

    # DATASET STATUS

    if cleaned_df is not None:

        st.divider()

        st.markdown("### ✅ Dataset Ready")

        st.success(
            f"{st.session_state['uploaded_file_name']} "
            "has been successfully loaded, cleaned, "
            "and prepared for analysis."
        )

        st.markdown("### 🚀 Analysis Workflow")

        st.write(
            "Use the navigation menu to review data quality, "
            "explore key metrics and visualizations, "
            "run machine-learning models, ask questions using "
            "natural language, and generate AI-powered insights."
        )

    else:

        st.divider()

        st.info(
            "Upload a dataset above to begin your analysis."
        )


# DATA QUALITY PAGE

elif page == "📂 Data Quality":

    st.title("📂 Data Quality")

    st.caption(
        "Understand the quality and structure of your dataset."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        cleaning_report = (
            st.session_state["cleaning_report"]
        )

# CLEANING REPORT

        st.markdown("### 🧹 Data Cleaning Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Original Rows",
                f"{cleaning_report['original_rows']:,}"
            )

        with col2:
            st.metric(
                "Cleaned Rows",
                f"{cleaning_report['cleaned_rows']:,}"
            )

        with col3:
            st.metric(
                "Duplicates Removed",
                f"{cleaning_report['duplicates_removed']:,}"
            )

        with col4:
            st.metric(
                "Missing Values Remaining",
                f"{cleaning_report['missing_values_after']:,}"
            )
# DATE COLUMNS

        date_columns = cleaning_report.get(
            "date_columns",
            []
        )

        if date_columns:

            st.divider()

            st.markdown("### 📅 Detected Date Columns")

            for column in date_columns:
                st.write(f"✓ {column}")

# DATASET PROFILE

        st.divider()

        st.markdown("### 📊 Dataset Structure")

        profile = get_dataset_profile(
            cleaned_df
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Numeric Columns",
                len(profile["numeric_columns"])
            )

        with col2:
            st.metric(
                "Categorical Columns",
                len(profile["categorical_columns"])
            )

        with col3:
            st.metric(
                "Date Columns",
                len(profile["datetime_columns"])
            )
# DATA PREVIEW

        st.divider()

        st.markdown("### 📋 Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(20),
            width="stretch"
        )

# USER-FRIENDLY NUMERICAL SUMMARY

        st.divider()
        st.markdown("### 🔢 Numerical Summary")

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns.tolist()

        excluded_summary_columns = {
            "id", "row_id", "order_id", "customer_id",
            "product_id", "postal_code", "zip_code", "zipcode"
        }

        numeric_columns = [
            column
            for column in numeric_columns
            if (
                column.lower() not in excluded_summary_columns
                and not column.lower().endswith("_id")
                and not column.lower().startswith("id_")
            )
        ]

        if numeric_columns:

            numeric_summary_rows = []

            for column in numeric_columns:
                numeric_summary_rows.append({
                    "Column": column.replace("_", " ").title(),
                    "Average": format_number(cleaned_df[column].mean()),
                    "Median": format_number(cleaned_df[column].median()),
                    "Minimum": format_number(cleaned_df[column].min()),
                    "Maximum": format_number(cleaned_df[column].max())
                })

            st.dataframe(
                pd.DataFrame(numeric_summary_rows),
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Average shows the typical value, Median shows the middle value, "
                "and Minimum and Maximum show the observed range."
            )

        else:
            st.info("No suitable numerical columns found.")

# CATEGORICAL SUMMARY

        st.markdown("### 🔤 Categorical Summary")

        categorical_columns = cleaned_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if categorical_columns:

            categorical_summary_rows = []

            for column in categorical_columns:

                value_counts = cleaned_df[column].value_counts(
                    dropna=True
                )

                if not value_counts.empty:
                    most_common = str(value_counts.index[0])
                    frequency = int(value_counts.iloc[0])
                else:
                    most_common = "N/A"
                    frequency = 0

                categorical_summary_rows.append({
                    "Column": column.replace("_", " ").title(),
                    "Unique Values": int(
                        cleaned_df[column].nunique(dropna=True)
                    ),
                    "Most Common Value": most_common,
                    "Frequency": frequency
                })

            st.dataframe(
                pd.DataFrame(categorical_summary_rows),
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Unique Values shows how many different values exist. "
                "Most Common Value shows the value that appears most often, "
                "and Frequency shows how many times it appears."
            )

        else:
            st.info("No categorical columns found.")


# DASHBOARD PAGE

elif page == "📊 Dashboard":

    st.title("📊 Data Dashboard")

    st.caption(
        "Get a quick view of important numerical measures "
        "across the complete cleaned dataset."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        df = cleaned_df
        numeric_columns = get_numeric_columns(df)

        identifier_names = {
            "id", "row_id", "order_id", "customer_id",
            "product_id", "postal_code", "zip_code", "zipcode"
        }

        def is_identifier_column(column_name):
            name = str(column_name).strip().lower()

            return (
                name in identifier_names
                or name.endswith("_id")
                or name.startswith("id_")
            )

        dashboard_numeric_columns = [
            column
            for column in numeric_columns
            if not is_identifier_column(column)
        ]

# KEY NUMERIC METRICS
        st.markdown("### 📈 Key Numeric Metrics")

        if dashboard_numeric_columns:

            metric_columns = dashboard_numeric_columns[:4]
            metric_cards = st.columns(len(metric_columns))

            for card, column_name in zip(
                metric_cards,
                metric_columns
            ):

                average_value = df[column_name].mean()

                with card:
                    st.metric(
                        label=(
                            f"Average "
                            f"{column_name.replace('_', ' ').title()}"
                        ),
                        value=format_number(average_value)
                    )

            st.caption(
                "These averages are calculated across the complete "
                "cleaned dataset."
            )

        else:

            st.info(
                "No suitable numerical fields were detected "
                "for dashboard metrics."
            )

# TIME OVERVIEW
        date_columns = get_date_columns(df)

        if date_columns:

            st.divider()
            st.markdown("### 📅 Time Overview")

            if len(date_columns) > 1:

                selected_date_column = st.selectbox(
                    "Select a date field",
                    date_columns,
                    key="dashboard_date_column"
                )

            else:

                selected_date_column = date_columns[0]

            dashboard_dates = pd.to_datetime(
                df[selected_date_column],
                errors="coerce"
            ).dropna()

            if not dashboard_dates.empty:

                start_date = dashboard_dates.min()
                end_date = dashboard_dates.max()

                total_days = max(
                    (end_date - start_date).days,
                    0
                )

                if total_days >= 730:
                    time_span = f"{total_days / 365.25:.1f} years"
                elif total_days >= 60:
                    time_span = f"{total_days / 30.44:.1f} months"
                else:
                    time_span = f"{total_days:,} days"

                time_col1, time_col2 = st.columns(2)

                with time_col1:
                    st.metric(
                        "Data Period",
                        (
                            f"{start_date.strftime('%b %Y')} "
                            f"– {end_date.strftime('%b %Y')}"
                        )
                    )

                with time_col2:
                    st.metric(
                        "Time Span",
                        time_span
                    )

                st.markdown("#### Records Over Time")

                time_data = pd.DataFrame({
                    "date": pd.to_datetime(
                        df[selected_date_column],
                        errors="coerce"
                    )
                }).dropna()

                # Choose the graph period automatically based on
                # the actual range of the selected date column.
                if total_days >= 730:
                    # Long-range datasets: show one point per year.
                    time_data["period"] = (
                        time_data["date"]
                        .dt.year
                        .astype(str)
                    )
                    x_title = "Year"

                elif total_days >= 90:
                    # Medium-range datasets: show one point per month.
                    time_data["period"] = (
                        time_data["date"]
                        .dt.to_period("M")
                        .astype(str)
                    )
                    x_title = "Month"

                else:
                    # Short-range datasets: show one point per day.
                    time_data["period"] = (
                        time_data["date"]
                        .dt.strftime("%Y-%m-%d")
                    )
                    x_title = "Date"

                records_over_time = (
                    time_data
                    .groupby("period", sort=False)
                    .size()
                    .reset_index(name="records")
                )

                # Sort the periods chronologically.
                if x_title == "Year":
                    records_over_time["sort_key"] = (
                        records_over_time["period"].astype(int)
                    )

                elif x_title == "Month":
                    records_over_time["sort_key"] = pd.to_datetime(
                        records_over_time["period"] + "-01",
                        errors="coerce"
                    )

                else:
                    records_over_time["sort_key"] = pd.to_datetime(
                        records_over_time["period"],
                        errors="coerce"
                    )

                records_over_time = (
                    records_over_time
                    .sort_values("sort_key")
                    .drop(columns="sort_key")
                )

                time_chart = px.line(
                    records_over_time,
                    x="period",
                    y="records",
                    markers=True,
                    labels={
                        "period": x_title,
                        "records": "Number of Records"
                    }
                )

                # Keep every actual period from the selected date field
                # visible in chronological order.
                time_chart.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=records_over_time["period"].tolist(),
                    tickmode="array",
                    tickvals=records_over_time["period"].tolist(),
                    ticktext=records_over_time["period"].tolist()
                )

                time_chart.update_layout(
                    xaxis_title=x_title,
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(
                    time_chart,
                    width="stretch"
                )

                st.caption(
                    "This chart shows how the dataset records are "
                    "distributed across the available time period."
                )

            else:

                st.info(
                    "A date field was detected, but it does not contain "
                    "usable date values for the Time Overview."
                )

        st.divider()

        st.info(
            "For distributions, category comparisons, relationships, "
            "trends, and correlations, use the Visual Analytics page."
        )


# VISUAL ANALYTICS PAGE

elif page == "📈 Visual Analytics":

    st.title("📈 Visual Analytics")

    st.caption(
        "Create interactive visualizations and explore "
        "relationships within your dataset."
    )

    if cleaned_df is None:
        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:
        df = cleaned_df

        numeric_columns = get_numeric_columns(df)

        # Remove Row ID and Postal Code from Visual Analytics
        excluded_numeric_columns = ["row_id", "postal_code"]

        numeric_columns = [
            col for col in numeric_columns
            if col.lower() not in excluded_numeric_columns
        ]

        categorical_columns = get_categorical_columns(df)
        date_columns = get_date_columns(df)

        # HISTOGRAM
        if numeric_columns:
            st.markdown("### 📊 Distribution Analysis")

            selected_numeric = st.selectbox(
                "Select a numeric column",
                numeric_columns,
                key="histogram_column"
            )

            histogram = create_histogram(df, selected_numeric)

            st.plotly_chart(
                histogram,
                width="stretch"
            )
        else:
            st.info(
                "No numeric columns are available "
                "for distribution analysis."
            )

        # BAR CHART
        if categorical_columns:
            st.divider()
            st.markdown("### 📊 Category Analysis")

            selected_category = st.selectbox(
                "Select a categorical column",
                categorical_columns,
                key="bar_column"
            )

            bar_chart = create_bar_chart(df, selected_category)

            st.plotly_chart(
                bar_chart,
                width="stretch"
            )
        else:
            st.info(
                "No categorical columns are available "
                "for category analysis."
            )

        # SCATTER PLOT
        if len(numeric_columns) >= 2:
            st.divider()
            st.markdown("### 🔵 Relationship Between Variables")

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="scatter_x"
                )

            with col2:
                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=1,
                    key="scatter_y"
                )

            scatter_chart = create_scatter_plot(
                df, x_column, y_column
            )

            st.plotly_chart(
                scatter_chart,
                width="stretch"
            )

# LINE CHART
        if date_columns and numeric_columns:
            st.divider()
            st.markdown("### 📈 Trend Over Time")

            col1, col2 = st.columns(2)

            with col1:
                selected_date = st.selectbox(
                    "Date column",
                    date_columns,
                    key="line_date"
                )

            with col2:
                selected_value = st.selectbox(
                    "Value column",
                    numeric_columns,
                    key="line_value"
                )

            line_chart = create_line_chart(
                df, selected_date, selected_value
            )

            st.plotly_chart(
                line_chart,
                width="stretch"
            )

# CORRELATION HEATMAP
        if len(numeric_columns) >= 2:
            st.divider()
            st.markdown("### 🔥 Correlation Analysis")

            heatmap = create_correlation_heatmap(
                df[numeric_columns]
            )

            st.plotly_chart(
                heatmap,
                width="stretch"
            )


# MACHINE LEARNING PAGE

elif page == "🤖 Machine Learning":

    st.title("🤖 Machine Learning")

    st.caption(
        "Train machine learning models using your dataset."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        df = cleaned_df

        ml_type = st.selectbox(
            "Select machine learning type",
            [
                "Regression",
                "Classification",
                "Clustering"
            ],
            key="ml_type"
        )

# REGRESSION
        if ml_type == "Regression":

            st.markdown("### 📈 Regression")

            st.write(
                "Train and compare regression models "
                "to predict a numerical target."
            )

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            if not numeric_columns:

                st.warning(
                    "No numerical columns are available "
                    "for regression."
                )

            else:

                target_column = st.selectbox(
                    "Select target column",
                    numeric_columns,
                    key="regression_target"
                )

                if st.button(
                    "🚀 Compare Regression Models",
                    key="compare_regression"
                ):

                    try:

                        results_df, best_model = (
                            compare_regression_models(
                                df,
                                target_column
                            )
                        )

                        st.success(
                            "All regression models trained successfully!"
                        )

                        st.markdown(
                            "### 📊 Model Comparison"
                        )

                        st.dataframe(
                            results_df,
                            width="stretch",
                            hide_index=True
                        )

                        st.markdown(
                            "### 🏆 Best Regression Model"
                        )

                        st.success(
                            f"Best model based on RMSE: "
                            f"{best_model}"
                        )

                        st.caption(
                            "RMSE is used as the primary selection metric because "
                            "it measures prediction error in the same units as the "
                            "target variable and penalizes larger errors. "
                            "Lower RMSE indicates better predictive performance."
                        )

                    except Exception as error:

                        import traceback

                        st.error(
                            f"Unable to train regression models: {error}"
                        )

                        st.code(
                            traceback.format_exc()
                        )


# CLASSIFICATION

        elif ml_type == "Classification":

            st.markdown("### 🎯 Classification")

            st.write(
                "Train and compare classification models "
                "to predict a categorical target."
            )

            classification_columns = []

            for column in df.columns:

                unique_values = df[column].nunique()

                if 2 <= unique_values <= 20:
                    classification_columns.append(column)

            if not classification_columns:

                st.warning(
                    "No suitable categorical target columns were found."
                )

            else:

                target_column = st.selectbox(
                    "Select target column",
                    classification_columns,
                    key="classification_target"
                )

                st.info(
                    f"Selected target has "
                    f"{df[target_column].nunique()} unique classes."
                )

                if st.button(
                    "🚀 Compare Classification Models",
                    key="compare_classification"
                ):

                    try:

                        results_df, best_model = (
                            compare_classification_models(
                                df,
                                target_column
                            )
                        )

                        st.success(
                            "All classification models "
                            "trained successfully!"
                        )

                        st.markdown(
                            "### 📊 Model Comparison"
                        )

                        st.dataframe(
                            results_df,
                            width="stretch"
                        )

                        best_f1 = results_df["F1 Score"].max()

                        best_models = results_df.loc[
                            results_df["F1 Score"] == best_f1,
                            "Model"
                        ].tolist()

                        if len(best_models) == 1:
                            st.markdown(
                                "### 🏆 Best Classification Model"
                            )

                            st.success(
                                f"Best model based on F1 Score: "
                                f"{best_models[0]}"
                            )

                        else:
                            st.markdown(
                                "### 🏆 Best Classification Models"
                            )

                            st.success(
                                f"Best models based on F1 Score: "
                                f"{', '.join(best_models)}"
                            )

                        st.caption(
                            "F1 Score is used as the primary selection metric because "
                            "it balances precision and recall. Higher F1 Score indicates "
                            "better classification performance. If multiple models achieve "
                            "the same highest F1 Score, all tied models are reported."
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to train classification "
                            f"models: {error}"
                        )


#CLUSTERING

        elif ml_type == "Clustering":

            st.markdown("### 🔵 K-Means Clustering")

            clustering_columns = get_clustering_columns(
                df
            )

            if len(clustering_columns) < 2:

                st.warning(
                    "At least two useful numeric columns "
                    "are required for clustering."
                )

            else:

                st.write(
                    "Columns used for clustering:"
                )

                st.write(
                    ", ".join(clustering_columns)
                )

# ELBOW METHOD

                st.markdown(
                    "### 📉 Elbow Method"
                )

                try:

                    elbow_results = calculate_elbow(
                        df
                    )

                    elbow_df = pd.DataFrame(
                        elbow_results
                    )

                    elbow_chart = create_elbow_plot(
                    elbow_results
                    )

                    st.plotly_chart(
                        elbow_chart,
                        width="stretch"
                    )

                    st.caption(
                        "The elbow point can help choose "
                        "a suitable number of clusters."
                    )

                except ValueError as error:

                    st.warning(
                        str(error)
                    )

# K-MEANS CLUSTERING

                max_clusters = min(7,len(df) - 1)
                default_clusters = min(
                    3,
                    max_clusters
                )

                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=max_clusters,
                    value=default_clusters,
                    key="clustering_clusters"
                )

                if st.button(
                    "Run Clustering",
                    key="run_clustering"
                ):

                    try:

                        model, scaler, result, score = (
                            train_kmeans(
                                df,
                                n_clusters
                            )
                        )
                        st.markdown("### 🎯 Clustering Quality")

                        st.metric(
                            "Silhouette Score",
                            f"{score:.3f}"
                        )
                        st.success(
                            "Clustering completed successfully."
                        )

                        st.markdown("### 📊 Cluster Summary")

                        cluster_summary = (
                                result
                                .groupby("cluster")
                                .agg(
                                    Rows=("cluster", "size")
                                )
                                .reset_index()
                            )

                        cluster_summary["Percentage"] = (
                                cluster_summary["Rows"]
                                / len(result)
                                * 100
                            ).round(2)

                        st.dataframe(
                                cluster_summary,
                                width="stretch",
                                hide_index=True
                            )
                        st.markdown("### 🔎 Cluster Profiles")

                        cluster_profiles = (
                            result
                            .groupby("cluster")
                            .mean(numeric_only=True)
                            .reset_index()
                        )

                        st.dataframe(
                            cluster_profiles,
                            width="stretch",
                            hide_index=True
                        )
                        st.markdown("### 📈 Cluster Distribution")

                        cluster_counts = (
                            result["cluster"]
                            .value_counts()
                            .sort_index()
                            .reset_index()
                        )

                        cluster_counts.columns = [
                            "Cluster",
                            "Rows"
                        ]

                        fig = px.bar(
                            cluster_counts,
                            x="Cluster",
                            y="Rows",
                            title="Number of Records in Each Cluster"
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch"
                        )
                        st.markdown("### 📋 Clustered Data")

                        st.dataframe(
                            result.head(100),
                            width="stretch",
                            hide_index=True
                        )
                        st.markdown(
                            "### 🔵 Cluster Visualization"
                        )

                        clustering_columns = (
                            get_clustering_columns(df)
                        )

                        if len(clustering_columns) >= 2:

                            col1, col2 = st.columns(2)

                            with col1:

                                x_column = st.selectbox(
                                    "X-axis",
                                    clustering_columns,
                                    key="cluster_x"
                                )

                            with col2:
                                y_column = [
                                    column
                                    for column in clustering_columns
                                    if column != x_column
                                ]

                                y_column = st.selectbox(
                                    "Y-axis",
                                    y_column,
                                    key="cluster_y_axis"
                                )

                            cluster_chart = px.scatter(
                                result,
                                x=x_column,
                                y=y_column,
                                color="cluster",
                                title=(
                                    f"Clusters: "
                                    f"{x_column} vs {y_column}"
                                )
                            )

                            st.plotly_chart(
                                cluster_chart,
                                width="stretch"
                            )

                    except ValueError as error:

                        st.error(
                            str(error)
                        )
# ASK YOUR DATA PAGE

elif page == "💬 Ask Your Data":

    st.title("💬 Ask Your Data")

    st.caption(
        "Ask questions about your dataset using natural language."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        question = st.text_input(
            "Ask a question about your dataset",
            placeholder="Enter your question..."
        )

        if st.button(
            "🔍 Ask",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Analyzing your question..."
                ):

                    answer = answer_question(
                        cleaned_df,
                        question
                    )

                st.divider()

                st.markdown("### 💡 Answer")

# GROUPED / COMPARISON RESULT
                if isinstance(answer, tuple):

                    results, grouped = answer

                    if len(grouped) > 0:

                        top_group = grouped.index[0]
                        top_value = grouped.iloc[0]

                        st.success(
                            f"{top_group} has the highest value "
                            f"at {top_value:,.2f}.\n\n "
                            "The complete comparison is shown below."
                        )

                    result_df = grouped.reset_index()

                    result_df.columns = [
                        "Category",
                        "Value"
                    ]

                    st.dataframe(
                        result_df,
                        width="stretch",
                        hide_index=True
                    )

                    st.markdown("### 📊 Comparison")

                    if (
                        "time" in question.lower()
                        or "trend" in question.lower()
                    ):

                        st.line_chart(
                            grouped
                        )

                    else:

                        st.bar_chart(
                            grouped
                        )

# LIST RESULT
                elif isinstance(answer, list):

                    for item in answer:
                        st.write(item)

# SINGLE NATURAL LANGUAGE ANSWER
                else:

                    st.success(answer)


# AI INSIGHTS PAGE

elif page == "🧠 AI Insights":

    st.title("🧠 AI Generated Insights")

    st.caption(
        "Use GenAI to automatically identify important "
        "patterns and insights from your dataset."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        st.write(
            "The AI analyst will examine the dataset and "
            "generate meaningful observations based on "
            "the available data."
        )

        if st.button(
            "✨ Generate AI Insights"
        ):

            with st.spinner(
                "AI is analyzing your dataset..."
            ):

                ai_insights = analyze_dataset(
                    cleaned_df
                )

            st.divider()

            st.markdown(
                "### 💡 AI Analysis"
            )

            st.write(
                ai_insights
            )