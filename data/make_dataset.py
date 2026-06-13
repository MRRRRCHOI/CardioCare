import pandas as pd

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    "data/processed.cleveland.data",
    names=columns,
    na_values="?"
)

df = df.apply(pd.to_numeric)

df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

print(df.head())
print(df.info())
print(df.isnull().sum())

df.to_csv("data/heart.csv", index=False)

print("saved: data/heart.csv")