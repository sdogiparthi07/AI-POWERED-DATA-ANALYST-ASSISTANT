import os
import json
import re

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


IDENTIFIER_NAMES = {
    "id", "row_id", "rowid", "record_id", "recordid",
    "order_id", "customer_id", "product_id", "transaction_id",
    "invoice_id", "user_id", "employee_id", "account_id",
    "postal_code", "postcode", "zip_code", "zipcode", "zip"
}


def is_identifier_column(column_name):
    """Return True for columns that look like identifiers or codes."""

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column_name).strip().lower()
    ).strip("_")

    return (
        name in IDENTIFIER_NAMES
        or name.endswith("_id")
        or name.startswith("id_")
    )


def create_dataset_summary(df):
    """Create a compact, dataset-agnostic summary for Gemini."""

    analytical_columns = [
        column
        for column in df.columns
        if not is_identifier_column(column)
    ]

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "analytical_column_names": [
            str(column)
            for column in analytical_columns
        ],
        "numeric_columns": {},
        "categorical_columns": {},
        "date_columns": {}
    }

    numeric_columns = df[analytical_columns].select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        summary["numeric_columns"][str(column)] = {
            "average": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "minimum": round(float(series.min()), 2),
            "maximum": round(float(series.max()), 2),
            "q1": round(float(series.quantile(0.25)), 2),
            "q3": round(float(series.quantile(0.75)), 2),
            "missing_values": int(df[column].isna().sum())
        }

    categorical_columns = df[analytical_columns].select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        values = (
            df[column]
            .value_counts(dropna=True)
            .head(5)
        )

        total_non_null = int(df[column].notna().sum())

        top_values = []

        for value, count in values.items():
            percentage = (
                (int(count) / total_non_null * 100)
                if total_non_null > 0
                else 0
            )

            top_values.append({
                "value": str(value),
                "count": int(count),
                "percentage": round(percentage, 2)
            })

        summary["categorical_columns"][str(column)] = {
            "unique_values": int(df[column].nunique(dropna=True)),
            "top_values": top_values,
            "missing_values": int(df[column].isna().sum())
        }

    date_columns = df[analytical_columns].select_dtypes(
        include="datetime"
    ).columns

    for column in date_columns:

        non_null_dates = df[column].dropna()

        if non_null_dates.empty:
            continue

        summary["date_columns"][str(column)] = {
            "start": str(non_null_dates.min()),
            "end": str(non_null_dates.max()),
            "missing_values": int(df[column].isna().sum())
        }

    return summary


def analyze_dataset(df):
    """Generate clear, evidence-based AI insights from the dataset."""

    dataset_summary = create_dataset_summary(df)

    prompt = f"""
You are an AI Data Analyst explaining dataset findings to a non-technical user.

Analyze only the dataset summary below:

{json.dumps(dataset_summary, indent=2, default=str)}

The dataset may belong to any domain such as sales, finance, marketing,
healthcare, education, operations, or another domain.

Provide exactly these four sections in this order:

## Most Important Insights

Give the most important high-level findings supported by the summary.
Use bullet points.

## Important Trends or Patterns

Describe meaningful distributions, concentrations, differences, or repeated
patterns that are directly visible in the summary.
Use bullet points.

## Key Observations

Highlight values or patterns that clearly stand out from typical values and may
deserve attention. These are not automatically errors or problems.
Use bullet points.

## Business Recommendations

Give exactly four practical next steps based only on the available evidence.
Use numbered points.

Rules:

- Use only information contained in the dataset summary.
- Do not invent statistics, relationships, causes, or domain facts.
- Do not assume the dataset is a sales dataset or belongs to any specific domain.
- Do not assume one row represents an order, customer, patient, student, or any
  other entity unless that meaning is explicitly supported.
- Refer to rows as "records" when the row meaning is not known.
- Do not treat record frequency as proof of demand, popularity, performance,
  importance, or preference. Say that a value "appears most often" or
  "accounts for the largest share of records" instead.
- Identifier and code columns are intentionally excluded from the analytical
  summary. Do not discuss IDs, postal codes, ZIP codes, or other identifiers.
- Do not claim correlation or association between variables because this summary
  does not contain cross-variable calculations.
- Do not claim that one variable caused another.
- When recommending a business action, first recommend reviewing, comparing,
  or validating the relevant records if the evidence is not sufficient for a
  direct policy or operational change.
- Do not recommend stricter policies, inventory changes, staffing changes,
  pricing changes, or other major actions based only on category counts.
- For Key Observations, focus on genuinely notable differences such as values
  far from the median or quartile range, strong category imbalance, or unusually
  large/small values. Do not label normal repeated activity as unusual.
- Use simple, professional language that a non-technical user can understand.
- Avoid unexplained technical terms such as skewness, cardinality, kurtosis,
  variance, or distributional shift. If a technical idea is necessary, explain
  it in plain English.
- Distinguish clearly between what the data shows and what should be investigated.
- Mention actual column names when useful, but make sentences natural.
- Keep the analysis concise and avoid repeating the same fact in multiple sections.
- Use Markdown headings and bullet points.
- Use numbered points only for the four Business Recommendations.
- Do not use LaTeX, HTML, code blocks, or backticks.
- Format large numbers with commas.
- Format percentages normally.
- Use currency symbols only when the column meaning clearly indicates money.
- Do not assume every numeric field is monetary.
- Return only the final analysis.
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
