import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from urllib.parse import quote_plus
# MySQL connection
password = quote_plus("Sriharsha@0704")
engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/superstore_db"
)
# Load data from MySQL

df = pd.read_sql(
    "SELECT * FROM sales_data",
    engine
)
print(df.head())
print(df.shape)
# 1. Sales by Category

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)


fig1 = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    title="Sales by Category",
    text="Sales"
)


fig1.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)
fig1.show()
# 2. Sales by Region

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)


fig2 = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    title="Sales by Region",
    text="Sales"
)


fig2.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)


fig2.show()
# 3. Yearly Sales Trend

year_sales = (
    df.groupby("Order Year")["Sales"]
    .sum()
    .reset_index()
)


fig3 = px.line(
    year_sales,
    x="Order Year",
    y="Sales",
    title="Yearly Sales Trend",
    markers=True
)


fig3.show()