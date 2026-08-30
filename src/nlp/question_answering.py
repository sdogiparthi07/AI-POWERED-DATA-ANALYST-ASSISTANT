import os
import json
import pandas as pd

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=api_key
)


def get_column_information(df):
    """Get useful information about dataset columns."""

    columns = []

    for column in df.columns:

        information = {
            "name": column,
            "type": str(df[column].dtype),
            "unique_values": int(df[column].nunique())
        }

        if df[column].dtype == "object":
            values = df[column].dropna().unique()[:5]

            information["examples"] = [
                str(value) for value in values
            ]

        columns.append(information)

    return columns


def find_column(df, column_name):
    """Find a column without worrying about capitalization."""

    if not column_name:
        return None

    wanted = str(column_name).strip().lower()

    for column in df.columns:

        if str(column).strip().lower() == wanted:
            return column

    return None


def calculate_result(
    df,
    operation,
    value_column,
    group_column=None
):
    """Perform the requested calculation."""

    if value_column is None:
        return "I could not identify the column."

    if value_column not in df.columns:
        return "The requested column does not exist."

    if operation in [
        "sum",
        "average",
        "maximum",
        "minimum"
    ]:

        if not pd.api.types.is_numeric_dtype(
            df[value_column]
        ):
            return (
                f"'{value_column}' is not a numerical column."
            )

    if group_column is None:

        if operation == "sum":

            value = df[value_column].sum()

            return (
                f"Total {value_column}: "
                f"{value:,.2f}"
            )

        if operation == "average":

            value = df[value_column].mean()

            return (
                f"Average {value_column}: "
                f"{value:,.2f}"
            )

        if operation == "maximum":

            value = df[value_column].max()

            return (
                f"Highest {value_column}: "
                f"{value:,.2f}"
            )

        if operation == "minimum":

            value = df[value_column].min()

            return (
                f"Lowest {value_column}: "
                f"{value:,.2f}"
            )

        if operation == "count":

            value = df[value_column].count()

            return (
                f"Number of records: "
                f"{value:,}"
            )

        if operation == "unique":

            value = df[value_column].nunique()

            return (
                f"Unique values in "
                f"{value_column}: {value:,}"
            )

        return "I could not determine the calculation."

    if group_column not in df.columns:

        return (
            f"'{group_column}' is not a column "
            "in the dataset."
        )

    if operation == "sum":

        grouped = (
            df.groupby(group_column)[value_column]
            .sum()
        )

    elif operation == "average":

        grouped = (
            df.groupby(group_column)[value_column]
            .mean()
        )

    elif operation == "maximum":

        grouped = (
            df.groupby(group_column)[value_column]
            .max()
        )

    elif operation == "minimum":

        grouped = (
            df.groupby(group_column)[value_column]
            .min()
        )

    elif operation == "count":

        grouped = (
            df.groupby(group_column)[value_column]
            .count()
        )

    else:

        return "This grouped calculation is not supported."

    grouped = grouped.sort_values(
        ascending=False
    )

    results = []

    for name, value in grouped.items():

        results.append(
            f"{name}: {value:,.2f}"
        )

    return results, grouped


def answer_question(df, question):
    """Understand and answer a natural-language question."""

    question = question.strip()

    if not question:
        return "Please enter a question."

    lower_question = question.lower()

    if "how many rows" in lower_question:

        return (
            f"Your dataset has "
            f"{len(df):,} rows."
        )

    if "how many columns" in lower_question:

        return (
            f"Your dataset has "
            f"{len(df.columns):,} columns."
        )

    if "missing values" in lower_question:

        missing = int(
            df.isna().sum().sum()
        )

        return (
            f"Your dataset has "
            f"{missing:,} missing values."
        )

    column_information = get_column_information(df)

    numeric_columns = []

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            numeric_columns.append(column)

    prompt = f"""
You are the question understanding part of an
AI Data Analyst Assistant.

Dataset columns:
{json.dumps(column_information, indent=2)}

Numeric columns:
{json.dumps(numeric_columns, indent=2)}

User question:
{question}

Understand what the user wants to calculate.

Return ONLY valid JSON in this format:

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
- unique

Rules:

- Use only columns that exist in the dataset.
- Never invent a column name.
- Match the question to the closest actual column.
- "total" or "sum" means sum.
- "average" or "mean" means average.
- "highest", "maximum", "largest" means maximum.
- "lowest", "minimum", "smallest" means minimum.
- "how many" or "count" means count when appropriate.
- "unique", "distinct", or "different values" means unique.
- If the user asks "by", "per", or "for each",
  identify the group column.
- If there is no grouping, use null.
- Return only JSON.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

    except Exception as error:

        return (
            "Gemini connection error: "
            f"{error}"
        )

    response_text = response.text.strip()

    if response_text.startswith("```"):

        response_text = response_text.replace(
            "```json",
            ""
        )

        response_text = response_text.replace(
            "```",
            ""
        )

        response_text = response_text.strip()

    try:

        result = json.loads(response_text)

    except json.JSONDecodeError:

        return (
            "I could not understand the question. "
            "Please try asking it another way."
        )

    operation = result.get("operation")

    value_column = result.get(
        "value_column"
    )

    group_column = result.get(
        "group_column"
    )

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

    return calculate_result(
        df,
        operation,
        actual_value_column,
        actual_group_column
    )