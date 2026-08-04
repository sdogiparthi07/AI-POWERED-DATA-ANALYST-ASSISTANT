import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus


password = quote_plus("Sriharsha@0704")

engine = create_engine(
    f"mysql+mysqlconnector://root:{password}@localhost/superstore_db"
)

df = pd.read_sql(
    "SELECT * FROM sales_data",
    engine
)


#Kpi analysis
# Total Sales
total_sales = df["Sales"].sum()

# Total Profit
total_profit = df["Profit"].sum()


# Total Orders
total_orders = df["Order ID"].nunique()


# Average Order Value
average_order_value = (
    total_sales / total_orders
)

# Profit Margin
profit_margin = (
    total_profit / total_sales
) * 100
#Business Insights
# Best Category

best_category = (
    df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)


# Best Region

best_region = (
    df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

# Top Product

top_product = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .idxmax()
)

# Yearly Sales

yearly_sales = (
    df.groupby("Order Year")["Sales"]
    .sum()
)

print("Total Sales:", round(total_sales,2))

print("Total Profit:", round(total_profit,2))

print("Total Orders:", total_orders)

print("Average Order Value:",
      round(average_order_value,2))

print("Profit Margin:",
      round(profit_margin,2), "%")


print("\nBest Category:",
      best_category)

print("Best Region:",
      best_region)

print("Top Product:",
      top_product)


print("\nYearly Sales:")
print(yearly_sales)
