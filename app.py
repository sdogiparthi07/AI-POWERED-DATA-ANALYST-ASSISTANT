import streamlit as st
import pandas as pd

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
    create_correlation_heatmap
)
from src.Machine_Learning.regression import (
    train_random_forest,
    train_model_for_prediction
)
from src.Machine_Learning.evaluation import evaluate_regression


# PAGE CONFIGURATION


st.set_page_config(
    page_title="AI Powered Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# SIDEBAR

with st.sidebar:

    st.title("📊 AI Powered Data Analyst Assistant")

    st.divider()

    st.subheader("Workspace")

    st.button("📂 Upload Dataset", width="stretch")
    st.button("🧹 Data Cleaning", width="stretch")
    st.button("📊 Data Analysis", width="stretch")
    st.button("📈 Visualizations", width="stretch")
    st.button("🤖 Machine Learning", width="stretch")
    st.button("🔮 Predictions", width="stretch")
    st.button("💬 AI Assistant", width="stretch")

    st.divider()

    st.caption("AI Powered Data Analyst Assistant v1.0")


# MAIN PAGE


st.title("AI Powered Data Analyst Assistant")

st.markdown(
    """
    ### Turn your data into insights.

    Upload a CSV or Excel dataset and let the system automatically:

    - 🧹 Clean your data
    - 📊 Analyze patterns
    - 📈 Create visualizations
    - 🤖 Build machine-learning models
    - 🔮 Generate predictions
    - 💬 Answer questions using natural language
    """
)

st.divider()


# DATASET UPLOAD


st.subheader("📂 Upload your dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


# PROCESS UPLOADED FILE


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

# CSV PROCESSING

    if uploaded_file.name.lower().endswith(".csv"):

# LOAD DATASET

        st.write("CSV file detected.")

        df = load_csv(uploaded_file)

# CLEAN DATASET
        

        cleaned_df, cleaning_report = clean_dataset(df)

        st.success("Dataset cleaned successfully!")
# CLEANING REPORT

        st.subheader("🧹 Data Cleaning Report")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Original Rows",
                cleaning_report["original_rows"]
            )

        with col2:
            st.metric(
                "Cleaned Rows",
                cleaning_report["cleaned_rows"]
            )

        with col3:
            st.metric(
                "Duplicates Removed",
                cleaning_report["duplicates_removed"]
            )

        with col4:
            st.metric(
                "Missing Values Remaining",
                cleaning_report["missing_values_after"]
            )

# DETECTED DATE COLUMNS

        if cleaning_report["date_columns"]:

            st.subheader("📅 Detected Date Columns")

            for column in cleaning_report["date_columns"]:

                st.write(f"✓ {column}")

# CLEANED DATASET PREVIEW

        st.subheader("📋 Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(10),
            width="stretch"
        )

# DATA ANALYSIS

        st.divider()

        st.subheader("📊 Dataset Analysis")

        profile = get_dataset_profile(cleaned_df)

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
# NUMERICAL ANALYSIS

        st.subheader("🔢 Numerical Analysis")

        numeric_summary = get_numeric_summary(cleaned_df)

        if not numeric_summary.empty:

            st.dataframe(
                numeric_summary,
                width="stretch"
            )

        else:

            st.info("No numerical columns found.")

# CATEGORICAL ANALYSIS
    

        st.subheader("🔤 Categorical Analysis")

        categorical_summary = get_categorical_summary(cleaned_df)

        if not categorical_summary.empty:

            st.dataframe(
                categorical_summary,
                width="stretch"
            )

        else:

            st.info("No categorical columns found.")
# VISUALIZATIONS

        st.divider()

        st.subheader("📈 Visualizations")

        numeric_columns = get_numeric_columns(cleaned_df)
        categorical_columns = get_categorical_columns(cleaned_df)
        date_columns = get_date_columns(cleaned_df)
# HISTOGRAM

        if numeric_columns:

            st.subheader("📊 Distribution")

            selected_numeric = st.selectbox(
                "Select a numeric column",
                numeric_columns,
                key="histogram_column"
            )

            histogram = create_histogram(
                cleaned_df,
                selected_numeric
            )

            st.plotly_chart(
                histogram,
                width="stretch"
            )
# BAR CHART

        if categorical_columns:

            st.subheader("📊 Category Analysis")

            selected_category = st.selectbox(
                "Select a categorical column",
                categorical_columns,
                key="bar_column"
            )

            bar_chart = create_bar_chart(
                cleaned_df,
                selected_category
            )

            st.plotly_chart(
                bar_chart,
                width="stretch"
            )


# SCATTER PLOT

        if len(numeric_columns) >= 2:

            st.subheader("🔵 Relationship Between Variables")

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
                cleaned_df,
                x_column,
                y_column
            )

            st.plotly_chart(
                scatter_chart,
                width="stretch"
            )


# LINE CHART

        if date_columns and numeric_columns:

            st.subheader("📈 Trend Over Time")

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
                cleaned_df,
                selected_date,
                selected_value
            )

            st.plotly_chart(
                line_chart,
                width="stretch"
            )
# CORRELATION HEATMAP

        if len(numeric_columns) >= 2:

            st.subheader("🔥 Correlation Heatmap")

            heatmap = create_correlation_heatmap(
                cleaned_df
            )

            st.plotly_chart(
                heatmap,
                width="stretch"
            )
# MACHINE LEARNING

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write(
        "Train a regression model to predict a numerical value."
    )

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns.tolist()

    target_column = st.selectbox(
        "Select target column",
        numeric_columns,
        key="ml_target"
    )


    @st.cache_data
    def train_model(df, target_column):

        model, y_test, predictions = train_random_forest(
            df,
            target_column
        )

        results = evaluate_regression(
            y_test,
            predictions
        )

        return results


    if st.button("🚀 Train Regression Model"):

        with st.spinner("Training regression model..."):

            results = train_model(
                cleaned_df,
                target_column
            )

        st.success("Model trained successfully!")

        st.subheader("📊 Model Evaluation")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "MAE",
                f"{results['mae']:.2f}"
            )

        with col2:
            st.metric(
                "MSE",
                f"{results['mse']:.2f}"
            )

        with col3:
            st.metric(
                "RMSE",
                f"{results['rmse']:.2f}"
            )

        with col4:
            st.metric(
                "R² Score",
                f"{results['r2']:.2f}")
#PREDICTIONS

    st.divider()

    st.subheader("🔮 Interactive Predictions")

    st.write(
        "Enter values for the features and generate a prediction."
    )

    prediction_model, feature_data = train_model_for_prediction(
        cleaned_df,
        target_column
    )

    input_data = {}

    for column in feature_data.columns:

        if pd.api.types.is_numeric_dtype(
            feature_data[column]
        ):

            default_value = float(
                feature_data[column].iloc[0]
            )

            input_data[column] = st.number_input(
                column,
                value=default_value,
                key=f"prediction_{column}"
            )

        else:

            options = (
                feature_data[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            input_data[column] = st.selectbox(
                column,
                options,
                key=f"prediction_{column}"
            )

    prediction_input = pd.DataFrame(
        [input_data]
    )

    if st.button("🔮 Predict"):

        with st.spinner("Generating prediction..."):

            prediction = prediction_model.predict(
                prediction_input
            )[0]

        st.success(
            "Prediction generated successfully!"
        )

        st.metric(
            "Predicted Value",
            f"{prediction:.2f}"
        )
else:
    st.warning("Excel support will be added next.")


