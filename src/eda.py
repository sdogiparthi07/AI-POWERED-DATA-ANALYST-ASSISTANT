import pandas as pd
# Load cleaned dataset
df = pd.read_csv(       
    "data/cleaned_superstore.csv"
)
#Dataset Overview
print("First 5 Rows:")   
print(df.head()) 

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)
 # Data Quality Check
print("\nMissing Values:")     
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Numerical Statistics
print("\nStatistical Summary:")    
print(df.describe())
# Unique Values Analysis
print("\nUnique Values:")    

for column in df.columns:
    print(column, ":", df[column].nunique())

#numerical distribution analysis and skewness
numerical_columns = [

    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Days"
]

print("\nSkewness:")
print(df[numerical_columns].skew())

# Correlation Analysis

print("\nCorrelation Matrix:")

print(
    df[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
            "Shipping Days"
        ]
    ].corr()
)


# Business Pattern Analysis
# 1. Negative Profit Analysis

negative_profit = df[df["Profit"] < 0]

print("\nLoss Making Transactions:")
print(negative_profit.shape[0])
# 2. Category wise Sales

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Category:")
print(category_sales)
# 3. Region wise Sales

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Region:")
print(region_sales)

# 4. Top 10 Products by Sales

top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products:")
print(top_products)

# 5. Yearly Sales Trend

year_sales = (
    df.groupby("Order Year")["Sales"]
    .sum()
)

print("\nYearly Sales:")
print(year_sales)