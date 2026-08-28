import pandas as pd

from src.GEN_AI.gemini_analyzer import analyze_dataset

print("Starting GenAI test...")

df = pd.read_csv("Data/superstore.csv", encoding="latin1")

print("Dataset loaded:", df.shape)

print("Sending dataset to Gemini...")

answer = analyze_dataset(df)

print("\n===== GEMINI ANALYSIS =====")
print(answer)

print("\n===== TEST COMPLETE =====")