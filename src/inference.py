import joblib
import pandas as pd
from preprocessing import validate_input_ranges


MODEL_PATH = "model.pkl"
SAMPLE_PATH = "data/heart.csv"


def main():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(SAMPLE_PATH)
    X = df.drop(columns=["target"]).head(5)

    validate_input_ranges(X)

    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        print("Prediction probabilities:")
        print(proba)

    print("Predictions:")
    print(pred)


if __name__ == "__main__":
    main()