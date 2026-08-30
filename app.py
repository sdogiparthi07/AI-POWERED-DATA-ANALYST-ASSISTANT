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
        "Upload your dataset to begin exploring its structure, "
        "patterns, and insights."
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
                    cleaned_df, cleaning_report = clean_dataset(df)

# Store dataset in session state
                st.session_state["cleaned_df"] = cleaned_df

                st.session_state["cleaning_report"] = (
                    cleaning_report
                )

                st.session_state["uploaded_file_name"] = (
                    uploaded_file.name
                )

                st.success(
                    "Dataset uploaded and cleaned successfully!"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to process the dataset: {e}"
                )

# SHOW DATASET STATUS

    if cleaned_df is not None:

        st.divider()

        st.markdown("### 📋 Current Dataset")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                f"{cleaned_df.shape[0]:,}"
            )

        with col2:
            st.metric(
                "Columns",
                f"{cleaned_df.shape[1]:,}"
            )

        with col3:

            numeric_count = len(
                cleaned_df.select_dtypes(
                    include="number"
                ).columns
            )

            st.metric(
                "Numeric Columns",
                numeric_count
            )

        with col4:

            categorical_count = len(
                cleaned_df.select_dtypes(
                    include=["object", "str"]
                ).columns
            )

            st.metric(
                "Categorical Columns",
                categorical_count
            )

        st.divider()

        st.markdown("### 🚀 Explore Your Dataset")

        st.write(
            "Use the navigation menu on the left to explore "
            "data quality, dashboards, visual analytics, "
            "machine learning, natural-language questions, "
            "and AI-generated insights."
        )

    else:

        st.divider()


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

# NUMERICAL ANALYSIS

        st.divider()

        st.markdown("### 🔢 Numerical Analysis")

        numeric_summary = get_numeric_summary(
            cleaned_df
        )

        if not numeric_summary.empty:

            st.dataframe(
                numeric_summary,
                width="stretch"
            )

        else:

            st.info(
                "No numerical columns found."
            )

# CATEGORICAL ANALYSIS

        st.markdown("### 🔤 Categorical Analysis")

        categorical_summary = get_categorical_summary(
            cleaned_df
        )

        if not categorical_summary.empty:

            st.dataframe(
                categorical_summary,
                width="stretch"
            )

        else:

            st.info(
                "No categorical columns found."
            )


# DASHBOARD PAGE

elif page == "📊 Dashboard":

    st.title("📊 Data Dashboard")

    st.caption(
        "Explore key metrics, patterns, and trends "
        "automatically identified from your dataset."
    )

    if cleaned_df is None:

        st.warning(
            "Please upload a dataset from the Overview page first."
        )

    else:

        df = cleaned_df

# DATASET OVERVIEW

        st.markdown("### 📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with col2:
            st.metric(
                "Columns",
                f"{df.shape[1]:,}"
            )

        with col3:

            numeric_columns = get_numeric_columns(df)

            st.metric(
                "Numeric Fields",
                len(numeric_columns)
            )

        with col4:

            categorical_columns = (
                get_categorical_columns(df)
            )

            st.metric(
                "Categorical Fields",
                len(categorical_columns)
            )

# DATA QUALITY SNAPSHOT

        st.divider()

        st.markdown("### 🔎 Data Quality Snapshot")

        missing_values = int(
            df.isna().sum().sum()
        )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        complete_rows = int(
            df.dropna().shape[0]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Missing Values",
                f"{missing_values:,}"
            )

        with col2:
            st.metric(
                "Duplicate Rows",
                f"{duplicate_rows:,}"
            )

        with col3:
            st.metric(
                "Complete Rows",
                f"{complete_rows:,}"
            )

# KEY NUMERIC METRICS

        st.divider()

        st.markdown("### 📈 Key Numeric Metrics")

        if numeric_columns:

            metric_columns = numeric_columns[:4]

            metric_cards = st.columns(
                len(metric_columns)
            )

            for card, column_name in zip(
                metric_cards,
                metric_columns
            ):

                average_value = df[
                    column_name
                ].mean()

                with card:
                    st.metric(
                        label=(
                            f"Average "
                            f"{column_name.replace('_', ' ').title()}"
                        ),
                        value=format_number(
                            average_value
                        )
                    )

        else:

            st.info(
                "No numeric columns were detected."
            )

# AUTOMATIC DASHBOARD VISUALS

        st.divider()

        st.markdown("### 📊 Dataset Highlights")

# First numeric distribution

        if numeric_columns:

            dashboard_col1, dashboard_col2 = st.columns(2)

            with dashboard_col1:

                st.markdown(
                    f"**Distribution of {numeric_columns[0]}**"
                )

                histogram = create_histogram(
                    df,
                    numeric_columns[0]
                )

                st.plotly_chart(
                    histogram,
                    width="stretch"
                )

# First categorical distribution

            if categorical_columns:

                with dashboard_col2:

                    st.markdown(
                        f"**Category Distribution by "
                        f"{categorical_columns[0]}**"
                    )

                    bar_chart = create_bar_chart(
                        df,
                        categorical_columns[0]
                    )

                    st.plotly_chart(
                        bar_chart,
                        width="stretch"
                    )

 # TIME TREND

        date_columns = get_date_columns(df)

        if date_columns and numeric_columns:

            st.divider()

            st.markdown("### 📈 Trend Over Time")

            selected_date = st.selectbox(
                "Date column",
                date_columns,
                key="dashboard_date"
            )

            selected_value = st.selectbox(
                "Numeric column",
                numeric_columns,
                key="dashboard_value"
            )

            line_chart = create_line_chart(
                df,
                selected_date,
                selected_value
            )

            st.plotly_chart(
                line_chart,
                width="stretch"
            )

# DASHBOARD NOTE

        st.divider()

        st.info(
            "💡 For detailed variable selection and advanced "
            "visual analysis, open the Visual Analytics page."
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

            histogram = create_histogram(
                df,
                selected_numeric
            )

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

            bar_chart = create_bar_chart(
                df,
                selected_category
            )

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

            st.markdown(
                "### 🔵 Relationship Between Variables"
            )

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
                df,
                x_column,
                y_column
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
                df,
                selected_date,
                selected_value
            )

            st.plotly_chart(
                line_chart,
                width="stretch"
            )

# CORRELATION HEATMAP

        if len(numeric_columns) >= 2:

            st.divider()

            st.markdown(
                "### 🔥 Correlation Analysis"
            )

            heatmap = create_correlation_heatmap(
                df
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
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown(
                            "### 🏆 Best Regression Model"
                        )

                        st.success(
                            f"Best model based on RMSE: "
                            f"{best_model}"
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

                        st.markdown(
                            "### 🏆 Best Classification Model"
                        )

                        st.success(
                            f"Best model based on F1 Score: "
                            f"{best_model}"
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
                        use_container_width=True
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

                        st.metric(
                            "Silhouette Score",
                            round(score, 3)
                        )

                        st.markdown("### 📊 Cluster Summary")

                        cluster_summary = (
                            result["cluster"]
                            .value_counts()
                            .sort_index()
                            .reset_index()
                        )

                        cluster_summary.columns = [
                            "Cluster",
                            "Records"
                        ]

                        st.dataframe(
                            cluster_summary,
                            use_container_width=True,
                            hide_index=True
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
                                use_container_width=True,
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
                            use_container_width=True,
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
                            use_container_width=True
                        )
                        st.markdown("### 🎯 Cluster Visualization")

                        cluster_columns = [
                            column
                            for column in result.columns
                            if column != "cluster"
                        ]

                        if len(cluster_columns) >= 2:

                            x_column = cluster_columns[0]
                            y_column = cluster_columns[1]

                            fig = px.scatter(
                                result,
                                x=x_column,
                                y=y_column,
                                color="cluster",
                                title=f"{x_column} vs {y_column} by Cluster"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "At least two numeric columns are required "
                                "for cluster visualization."
                            )
                        st.markdown("### 📋 Clustered Data")

                        st.dataframe(
                            result.head(100),
                            use_container_width=True,
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
                                    for column in cluster_columns
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
                                use_container_width=True
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
            "Ask a question about your dataset...",
            placeholder=(
                "Example: Which category has the highest value?"
            )
        )

        if st.button("🔍 Ask"):

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

# GROUPED RESULT WITH CHART

                if isinstance(answer, tuple):

                    results, grouped = answer

                    for item in results:
                        st.write(item)

                    st.markdown(
                        "### 📊 Visualization"
                    )

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

# TEXT RESULT

                else:

                    st.write(answer)


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

