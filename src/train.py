import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import load_data, split_features_target, build_preprocessor


RANDOM_STATE = 42
DATA_PATH = "data/heart.csv"
MODEL_PATH = "model.pkl"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def run_basic_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "SVC": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE
        )
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("feature_selection", SelectFromModel(
                RandomForestClassifier(random_state=RANDOM_STATE)
            )),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            metrics, cm = evaluate_model(pipeline, X_test, y_test)

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="balanced_accuracy"
            )

            mlflow.log_param("model_family", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metric("cv_balanced_accuracy_mean", cv_scores.mean())

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.log_text(str(cm), "confusion_matrix.txt")
            mlflow.sklearn.log_model(pipeline, "model")

            print("=" * 50)
            print(name)
            print(metrics)
            print("Confusion Matrix:")
            print(cm)
            print("CV mean:", cv_scores.mean())

            # 의료 맥락상 recall 우선 선택
            if metrics["recall"] > best_score:
                best_score = metrics["recall"]
                best_model = pipeline
                best_name = name

    return best_name, best_model, best_score


def run_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("feature_selection", SelectFromModel(
            RandomForestClassifier(random_state=RANDOM_STATE)
        )),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, None],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    with mlflow.start_run(run_name="RandomForest_GridSearchCV"):
        grid.fit(X_train, y_train)

        best_tuned_model = grid.best_estimator_
        metrics, cm = evaluate_model(best_tuned_model, X_test, y_test)

        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_balanced_accuracy", grid.best_score_)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.sklearn.log_model(best_tuned_model, "model")

        print("=" * 50)
        print("RandomForest GridSearchCV")
        print("Best Params:", grid.best_params_)
        print("Best CV Score:", grid.best_score_)
        print(metrics)
        print("Confusion Matrix:")
        print(cm)

    return "RandomForest_GridSearchCV", best_tuned_model, metrics["recall"]


def main():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    mlflow.set_experiment("CardioCare")

    best_name, best_model, best_recall = run_basic_models(
        X_train, X_test, y_train, y_test
    )

    tuned_name, tuned_model, tuned_recall = run_hyperparameter_tuning(
        X_train, X_test, y_train, y_test
    )

    if tuned_recall > best_recall:
        final_name = tuned_name
        final_model = tuned_model
    else:
        final_name = best_name
        final_model = best_model

    joblib.dump(final_model, MODEL_PATH)

    print("=" * 50)
    print("Final Best Model:", final_name)
    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()