import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


TARGET_COL = "target"

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError("target column is missing")

    df = df.drop_duplicates()

    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=empty_cols)

    if df[TARGET_COL].nunique() > 2:
        df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 0 if x == 0 else 1)

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def validate_input_ranges(df: pd.DataFrame):
    if "chol" in df.columns:
        if not df["chol"].between(0, 600).all():
            raise ValueError("chol must be between 0 and 600")

    if "trestbps" in df.columns:
        if not df["trestbps"].between(0, 300).all():
            raise ValueError("trestbps must be between 0 and 300")

    if "age" in df.columns:
        if not df["age"].between(0, 120).all():
            raise ValueError("age must be between 0 and 120")