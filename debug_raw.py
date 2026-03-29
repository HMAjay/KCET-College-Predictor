import pandas as pd
import os

path = "data/extracted/KCET_2021_Cutoff_raw.csv"
df = pd.read_csv(path, dtype=str, header=None)

print("Shape:", df.shape)
print("\nFirst 30 rows:")
print(df.iloc[:30].to_string())