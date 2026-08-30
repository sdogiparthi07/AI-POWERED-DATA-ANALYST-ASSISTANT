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


def create_dataset_summary(df):
    """Create a compact summary of the dataset."""

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [
            str(column)
            for column in df.columns
        ],
        "numeric_columns": {},
        "categorical_columns": {},
        "date_columns": []
    }

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        summary["numeric_columns"][str(column)] = {
            "average": round(
                float(df[column].mean()), 2
            ),
            "minimum": round(
                float(df[column].min()), 2
            ),
            "maximum": round(
                float(df[column].max()), 2
            ),
            "missing_values": int(
                df[column].isna().sum()
            )
        }

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        values = (
            df[column]
            .value_counts()
            .head(5)
        )

        summary["categorical_columns"][
            str(column)
        ] = {
            "unique_values": int(
                df[column].nunique()
            ),
            "top_values": values.to_dict(),
            "missing_values": int(
                df[column].isna().sum()
            )
        }

    date_columns = df.select_dtypes(
        include="datetime"
    ).columns

    summary["date_columns"] = [
        str(column)
        for column in date_columns
    ]

    return summary


def analyze_dataset(df):
    """Generate AI business insights from the dataset."""

    dataset_summary = create_dataset_summary(df)

    prompt = f"""
You are an AI Data Analyst.

Analyze the following dataset summary:

{json.dumps(dataset_summary, indent=2, default=str)}

The dataset can belong to any domain such as sales,
finance, marketing, customers, healthcare, education,
operations, or another business domain.

Provide exactly four sections.

## Most Important Insights

Give the most important findings from the data.

Use bullet points.

## Important Trends or Patterns

Describe meaningful trends, patterns, distributions,
concentrations, or differences visible in the data.

Use bullet points.

## Unusual Findings

Identify unusual values, extreme values, imbalances,
or findings that may need investigation.

Use bullet points.

## Business Recommendations

Give exactly four practical recommendations based
only on the available evidence.

Use numbered points.


Rules:

- Use only information contained in the dataset summary.
- Do not invent statistics.
- Do not assume the dataset is a sales dataset.
- Do not assume that rows represent orders, customers,
  patients, students, or any other specific entity.
- Call rows "dataset rows" unless their meaning is clear.
- Do not claim that one variable caused another.
- Correlation does not prove causation.
- Use phrases such as "may indicate", "suggests",
  or "warrants further investigation" when appropriate.
- Mention actual column names when useful.
- Use clear and simple business language.
- Keep the analysis concise.
- Use Markdown headings and bullet points.
- Use numbered points only for the three recommendations.
- Do not use LaTeX.
- Do not use HTML.
- Do not use code blocks.
- Do not use backticks.
- Format large numbers with commas.
- Format percentages normally.
- Use currency symbols only when the dataset clearly
  represents monetary values.
- Do not assume that every numeric column represents money.
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