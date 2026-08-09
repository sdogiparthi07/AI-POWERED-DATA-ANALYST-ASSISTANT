import pandas as pd

from src.data_cleaner import clean_dataset
from src.Machine_Learning.regression import train_random_forest
from src.Machine_Learning.evaluation import evaluate_regression


df = pd.read_csv("data/superstore.csv")

print("Dataset loaded")


cleaned_df, cleaning_report = clean_dataset(df)

print("Dataset cleaned")


model, y_test, predictions = train_random_forest(
    cleaned_df,
    "sales"
)

print("Model trained")


results = evaluate_regression(
    y_test,
    predictions
)

print("Model Results:")
print(results)