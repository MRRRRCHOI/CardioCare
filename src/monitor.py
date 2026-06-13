import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from preprocessing import load_data, split_features_target


DATA_PATH = "data/heart.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42


logging.basicConfig(
    filename="prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    model = joblib.load(MODEL_PATH)

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    original_pred = model.predict(X_test)
    original_score = balanced_accuracy_score(y_test, original_pred)

    drifted_X_test = X_test.copy()
    drifted_X_test["chol"] = drifted_X_test["chol"] + 30
    drifted_X_test["chol"] *= 1.4

    drifted_pred = model.predict(drifted_X_test)
    drifted_score = balanced_accuracy_score(y_test, drifted_pred)

    logging.info({
        "model_version": "1.0",
        "input_shape": X_test.shape,
        "original_predictions": original_pred.tolist(),
        "drifted_predictions": drifted_pred.tolist()
    })

    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    print("KS Drift Test Results")
    for col in continuous_cols:
        stat, p_value = ks_2samp(X_train[col], drifted_X_test[col])
        drift_flag = p_value < 0.05

        print(
            f"{col}: statistic={stat:.4f}, "
            f"p-value={p_value:.4f}, drift={drift_flag}"
        )

    print("Original balanced accuracy:", original_score)
    print("Drifted balanced accuracy:", drifted_score)

    scores = [original_score, drifted_score]
    labels = ["original", "drifted"]

    plt.figure()
    plt.plot(labels, scores, marker="o")
    plt.title("Balanced Accuracy Before and After Drift")
    plt.ylabel("Balanced Accuracy")
    plt.savefig("drift_performance.png")


if __name__ == "__main__":
    main()