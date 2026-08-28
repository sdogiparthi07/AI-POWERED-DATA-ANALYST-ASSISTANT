import pandas as pd

from src.data_cleaner import clean_dataset
from src.Machine_Learning.regression import train_random_forest
from src.Machine_Learning.evaluation import evaluate_regression
from src.Machine_Learning.clustering import train_kmeans

from src.Machine_Learning.clustering import (
    train_kmeans,
    calculate_elbow
)


print("Loading dataset...")

df = pd.read_csv(
    "data/superstore.csv",
    encoding="latin1"
)

print("Dataset loaded")


cleaned_df, cleaning_report = clean_dataset(df)

print("Dataset cleaned")
print("Starting model training...")


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


print("Starting K-Means clustering...")

model, scaler, clustered_data, score = train_kmeans(
    cleaned_df,
    n_clusters=3
)

print("K-Means completed")

print("Clustered Data:")
print(clustered_data.head())

print("Silhouette Score:")
print(score)

print("Calculating elbow method...")

elbow_results = calculate_elbow(
    cleaned_df
)

print("Elbow Results:")

for result in elbow_results:
    print(result)