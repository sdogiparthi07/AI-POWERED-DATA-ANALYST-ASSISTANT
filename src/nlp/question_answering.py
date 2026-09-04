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
            "unique_values": int(
                df[column].nunique()
            )
        }

        if df[column].dtype == "object":

            values = (
                df[column]
                .dropna()
                .unique()[:5]
            )

            information["examples"] = [
                str(value)
                for value in values
            ]

        columns.append(information)

    return columns


def find_column(df, column_name):
    """Find a column without worrying about capitalization."""

    if not column_name:
        return None

    wanted = (
        str(column_name)
        .strip()
        .lower()
    )

    for column in df.columns:

        if (
            str(column)
            .strip()
            .lower()
            == wanted
        ):
            return column

    return None


def calculate_result(
    df,
    operation,
    value_column,
    group_column=None,
    result_mode="all"
):
    """
    Perform the requested calculation.

    result_mode:
    - all: show all grouped results
    - top: show only highest result
    - bottom: show only lowest result
    """

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

    # NO GROUPING
    if group_column is None:

        if operation == "sum":

            value = df[value_column].sum()

            return (
                f"The total {value_column.replace('_', ' ')} "
                f"is {value:,.2f}."
            )

        if operation == "average":

            value = df[value_column].mean()

            return (
                f"The average {value_column.replace('_', ' ')} "
                f"is {value:,.2f}."
            )

        if operation == "maximum":

            value = df[value_column].max()

            return (
                f"The highest {value_column.replace('_', ' ')} "
                f"is {value:,.2f}."
            )

        if operation == "minimum":

            value = df[value_column].min()

            return (
                f"The lowest {value_column.replace('_', ' ')} "
                f"is {value:,.2f}."
            )

        if operation == "count":

            value = df[value_column].count()

            return (
                f"There are {value:,} records "
                f"for {value_column.replace('_', ' ')}."
            )

        if operation == "unique":

            value = df[value_column].nunique()

            return (
                f"There are {value:,} unique values in "
                f"{value_column.replace('_', ' ')}."
            )

        return "I could not determine the calculation."

    # GROUPED CALCULATION
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

    grouped = grouped.dropna()

    if grouped.empty:
        return "No results were found for this question."

    # HIGHEST GROUP
    if result_mode == "top":

        highest_group = grouped.idxmax()
        highest_value = grouped.max()

        measure_name = value_column.replace("_", " ")

        if operation == "average":
            return (
                f"{highest_group} has the highest average "
                f"{measure_name} at {highest_value:,.2f}."
            )

        if operation == "count":
            return (
                f"{highest_group} has the highest count "
                f"at {highest_value:,.0f}."
            )

        return (
            f"{highest_group} has the highest total "
            f"{measure_name} at {highest_value:,.2f}."
        )

    # LOWEST GROUP
    if result_mode == "bottom":

        lowest_group = grouped.idxmin()
        lowest_value = grouped.min()

        measure_name = value_column.replace("_", " ")

        if operation == "average":
            return (
                f"{lowest_group} has the lowest average "
                f"{measure_name} at {lowest_value:,.2f}."
            )

        if operation == "count":
            return (
                f"{lowest_group} has the lowest count "
                f"at {lowest_value:,.0f}."
            )

        return (
            f"{lowest_group} has the lowest total "
            f"{measure_name} at {lowest_value:,.2f}."
        )

    # ALL GROUPS
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

    # SIMPLE QUESTIONS
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
You are the question-understanding component
of an AI Data Analyst Assistant.

Dataset columns:
{json.dumps(column_information, indent=2)}

Numeric columns:
{json.dumps(numeric_columns, indent=2)}

User question:
{question}

Understand exactly what the user wants.

Return ONLY valid JSON in this format:

{{
    "operation": "sum",
    "value_column": "sales",
    "group_column": "region",
    "result_mode": "top"
}}

Allowed operations:
- sum
- average
- maximum
- minimum
- count
- unique

Allowed result_mode values:
- all
- top
- bottom

Rules:

1. Use only columns that exist in the dataset.
2. Never invent a column name.
3. Match the user's wording to the closest actual column.
4. "total" or "sum" means operation = "sum".
5. "average" or "mean" means operation = "average".
6. If the user asks for the highest individual value,
   use operation = "maximum".
7. If the user asks for the lowest individual value,
   use operation = "minimum".

8. If the user asks WHICH GROUP has the highest
   numerical measure, usually use:
   operation = "sum"
   result_mode = "top"

Example:
"Which region has the highest sales?"

{{
    "operation": "sum",
    "value_column": "sales",
    "group_column": "region",
    "result_mode": "top"
}}

9. If the user asks WHICH GROUP has the lowest
   numerical measure, usually use:
   operation = "sum"
   result_mode = "bottom"

10. For grouped comparison questions such as:
"Show sales by region"
"Sales for each region"
"Compare sales by region"

use:
operation = "sum"
result_mode = "all"

11. If the question contains "by", "per",
or "for each", identify the group column.

12. If the user asks:
"average sales by region"

use:
operation = "average"
result_mode = "all"

13. If the user asks:
"which region has the highest average sales"

use:
operation = "average"
group_column = "region"
result_mode = "top"

14. If there is no grouping,
group_column must be null.

15. "how many" or "count" means count
when appropriate.

16. "unique", "distinct", or
"different values" means unique.

Return ONLY JSON.
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

        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:

        result = json.loads(response_text)

    except json.JSONDecodeError:

        return (
            "I could not understand the question. "
            "Please try asking it another way."
        )

    operation = result.get("operation")
    value_column = result.get("value_column")
    group_column = result.get("group_column")
    result_mode = result.get("result_mode", "all")

    if result_mode not in [
        "all",
        "top",
        "bottom"
    ]:
        result_mode = "all"

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
        df=df,
        operation=operation,
        value_column=actual_value_column,
        group_column=actual_group_column,
        result_mode=result_mode
    )
