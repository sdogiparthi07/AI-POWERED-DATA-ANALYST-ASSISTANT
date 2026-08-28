import os
import json

import pandas as pd
from dotenv import load_dotenv
from google import genai


# Load the Gemini API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)


def get_column_information(df):
    """Get the names and data types of the dataset columns."""

    columns = []

    for column in df.columns:
        columns.append({
            "name": column,
            "type": str(df[column].dtype)
        })

    return columns


def find_column(df, column_name):
    """Find a dataset column without worrying about capitalization."""

    if not column_name:
        return None

    wanted = str(column_name).strip().lower()

    for column in df.columns:
        if str(column).strip().lower() == wanted:
            return column

    return None


def calculate_result(df, operation, value_column, group_column=None):
    """Calculate the answer using pandas."""

    if value_column is None:
        return "I could not identify the numerical column."

    if value_column not in df.columns:
        return "The requested column does not exist."

    # The column must contain numbers for calculations such as
    # total, average, highest, or lowest.
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"'{value_column}' is not a numerical column."

    # --------------------------------------------------
    # Calculation without grouping
    # --------------------------------------------------

    if group_column is None:

        if operation == "sum":
            value = df[value_column].sum()
            return f"Total {value_column}: {value:,.2f}"

        if operation == "average":
            value = df[value_column].mean()
            return f"Average {value_column}: {value:,.2f}"

        if operation == "maximum":
            value = df[value_column].max()
            return f"Highest {value_column}: {value:,.2f}"

        if operation == "minimum":
            value = df[value_column].min()
            return f"Lowest {value_column}: {value:,.2f}"

        if operation == "count":
            value = df[value_column].count()
            return f"Number of records: {value:,}"

        return "I could not determine the calculation."

    # --------------------------------------------------
    # Calculation grouped by another column
    # --------------------------------------------------

    if group_column not in df.columns:
        return f"'{group_column}' is not a column in the dataset."

    if operation == "average":
        grouped = df.groupby(group_column)[value_column].mean()

    elif operation == "maximum":
        grouped = df.groupby(group_column)[value_column].max()

    elif operation == "minimum":
        grouped = df.groupby(group_column)[value_column].min()

    elif operation == "count":
        grouped = df.groupby(group_column)[value_column].count()

    else:
        grouped = df.groupby(group_column)[value_column].sum()

    grouped = grouped.sort_values(ascending=False)

    results = []

    for name, value in grouped.items():
        results.append(f"{name}: {value:,.2f}")

    # Return both the text result and the pandas data
    # so Streamlit can create a chart.
    return results, grouped


def answer_question(df, question):
    """Understand a natural-language question and answer it."""

    question = question.strip()

    if not question:
        return "Please enter a question."

    lower_question = question.lower()

    # --------------------------------------------------
    # Simple questions that do not need Gemini
    # --------------------------------------------------

    if "how many rows" in lower_question:
        return f"Your dataset has {len(df):,} rows."

    if "how many columns" in lower_question:
        return f"Your dataset has {len(df.columns):,} columns."

    if "missing values" in lower_question:
        missing = df.isnull().sum().sum()
        return f"Your dataset has {missing:,} missing values."

    # --------------------------------------------------
    # Give Gemini information about the actual dataset
    # --------------------------------------------------

    column_information = get_column_information(df)

    numeric_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)

    categorical_columns = []

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            categorical_columns.append(column)

    prompt = f"""
You are the question understanding part of a general
AI Data Analyst Assistant.

The user has uploaded a dataset.

Dataset columns:
{json.dumps(column_information, indent=2)}

Numerical columns:
{json.dumps(numeric_columns, indent=2)}

Other columns:
{json.dumps(categorical_columns, indent=2)}

User question:
{question}

Determine what calculation the user wants.

Return ONLY valid JSON using this format:

{{
    "operation": "sum",
    "value_column": "column name",
    "group_column": null
}}

Allowed operations:

- sum
- average
- maximum
- minimum
- count

Rules:

- Use only columns that exist in the dataset.
- Never invent a column name.
- Match the user's wording to the closest actual column.
- "total" or "sum" means sum.
- "average" or "mean" means average.
- "highest", "maximum", or "largest" means maximum.
- "lowest", "minimum", or "smallest" means minimum.
- "how many" or "count" means count when appropriate.
- If the question asks "by", "per", or "for each",
  identify the grouping column.
- If there is no grouping, use null.
- Return only JSON.
"""

    # --------------------------------------------------
    # Ask Gemini to understand the question
    # --------------------------------------------------

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

    except Exception as error:
        return f"Gemini connection error: {error}"

    # --------------------------------------------------
    # Read Gemini's answer
    # --------------------------------------------------

    response_text = response.text.strip()

    # Gemini may occasionally return JSON inside
    # Markdown code fences. Remove those fences.
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    try:
        result = json.loads(response_text)

    except json.JSONDecodeError:
        return (
            "I could not understand the question. "
            "Please try asking it another way."
        )

    # --------------------------------------------------
    # Get the calculation details from Gemini
    # --------------------------------------------------

    operation = result.get("operation")
    value_column = result.get("value_column")
    group_column = result.get("group_column")

    # Find the real column names in the dataframe.
    actual_value_column = find_column(
        df,
        value_column
    )

    actual_group_column = None

    if group_column:
        actual_group_column = find_column(
            df,
            group_column
        )

    if actual_value_column is None:
        return (
            f"I could not find the column "
            f"'{value_column}' in the dataset."
        )

    if group_column and actual_group_column is None:
        return (
            f"I could not find the column "
            f"'{group_column}' in the dataset."
        )

    # --------------------------------------------------
    # Perform the actual calculation
    # --------------------------------------------------

    return calculate_result(
        df,
        operation,
        actual_value_column,
        actual_group_column
    )

   