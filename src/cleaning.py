import pandas as pd
df = pd.read_csv(
    "data/superstore.csv",
    encoding="latin1"
)
print(df.head())
print(df.shape)
print(df.columns)

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df["Order Date"] = pd.to_datetime(
    df["Order Date"]
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"]
)
df["Order Year"] = (
    df["Order Date"].dt.year
)
df["Order Month"] = (
    df["Order Date"].dt.month
)
df["Shipping Days"] = (
    df["Ship Date"] -
    df["Order Date"]
).dt.days
print(df[["Order Date", "Order Year", "Order Month"]].head(10))
df.to_csv(
    "data/cleaned_superstore.csv",
    index=False
)

print("Cleaning completed!")