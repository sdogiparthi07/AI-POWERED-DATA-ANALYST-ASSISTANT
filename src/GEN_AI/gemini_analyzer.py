import os
import pandas as pd

from dotenv import load_dotenv
from google import genai


# Load Gemini API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=api_key
)


def create_dataset_summary(df):
    """Create a useful summary of the uploaded dataset."""

    summary = []

# Basic dataset information
    summary.append(
        f"Number of rows: {len(df):,}"
    )

    summary.append(
        f"Number of columns: {len(df.columns):,}"
    )

    summary.append(
        "Columns: "
        + ", ".join(str(column) for column in df.columns)
    )

# Identify identifier columns
    identifier_columns = []

    for column in df.columns:

        column_name = str(column).lower().strip()

        if (
            column_name == "id"
            or column_name.endswith(" id")
            or column_name.endswith("_id")
            or "row id" in column_name
            or "zip" in column_name
            or "postal" in column_name
        ):
            identifier_columns.append(column)

# Numeric columns
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

# Do not analyze IDs or postal codes as business metrics
        if column in identifier_columns:
            continue

        total = df[column].sum()
        average = df[column].mean()
        minimum = df[column].min()
        maximum = df[column].max()

        summary.append(
            f"{column}: "
            f"total={total:,.2f}, "
            f"average={average:,.2f}, "
            f"minimum={minimum:,.2f}, "
            f"maximum={maximum:,.2f}"
        )

# Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        values = df[column].value_counts().head(10)

        summary.append(
            f"{column} top values: "
            f"{values.to_dict()}"
        )

# Date columns
    date_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            date_columns.append(column)

    if date_columns:

        summary.append(
            "Date columns: "
            + ", ".join(
                str(column)
                for column in date_columns
            )
        )

    return "\n".join(summary)


def analyze_dataset(df):
    """Analyze the uploaded dataset using Gemini."""

    dataset_summary = create_dataset_summary(df)

    prompt = f"""
You are an AI Data Analyst.

Analyze the following dataset summary:

{dataset_summary}

Provide exactly four sections.

## Most Important Insights

Give the most important findings from the dataset.

Use bullet points.

## Important Trends or Patterns

Describe meaningful trends, patterns, concentrations, or distributions.

Use bullet points.

## Unusual Findings

Identify unusual values, extreme values, imbalances, or findings that deserve investigation.

Use bullet points.

## Business Recommendations

Give exactly three practical business recommendations.

Use numbered points.

Formatting rules:

- Use clean Markdown.
- Use clear section headings.
- Use bullet points for insights, trends, and unusual findings.
- Use numbered points for the three business recommendations.
- Do not use LaTeX.
- Never use backticks anywhere in the response.
- Never use inline code formatting.
- Do not use HTML.
- Do not use SVG.
- Do not create anchor links.
- Do not produce broken Markdown.
- Always write currency with the $ symbol.
- Example: $2,297,200.86
- Always put spaces between words and numbers.
- Always put spaces around currency values where appropriate.
- Write percentages normally.
- Example: 12.5%
- Use commas for large numbers.
- Do not put asterisks inside numbers or sentences.
- Do not combine words and numbers without spaces.
- Refer to dataset rows as "transaction rows" unless unique orders have actually been calculated.
- Do not call transaction rows "orders" unless unique orders have actually been calculated.
- Do not claim that one factor caused another unless the dataset directly proves it.
- When evidence is insufficient, say "may indicate", "suggests", or "warrants further investigation".
- Do not invent data.
- Use only information contained in the dataset summary.
- Do not invent statistics that are not present in the dataset summary.
- Keep the analysis professional, concise, and easy for a business user to understand.

Important formatting examples:

Correct:
The business generated total sales of $2,297,200.86 and total profit of $286,397.02.

Correct:
The average sales per transaction row is $229.86, while the average profit is $28.66.

Incorrect:
$2,297,200.86andatotalprofitof$286,397.02

Incorrect:
$229.86,whiletheaverageprofitis$28.66

Return only the final business analysis.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if not response or not response.text:

            return (
                "Gemini did not return any analysis. "
                "Please try again."
            )

        return response.text.strip()

    except Exception as error:

        return (
            "Unable to generate AI business insights "
            f"at the moment. Details: {error}"
        )