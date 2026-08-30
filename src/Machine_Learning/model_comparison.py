import pandas as pd

from src.Machine_Learning.regression import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    train_gradient_boosting
)

from src.Machine_Learning.classification import (
    train_logistic_regression,
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_gradient_boosting_classifier
)

from src.Machine_Learning.evaluation import (
    evaluate_regression,
    evaluate_classification
)


def compare_regression_models(df, target_column):
    """Compare regression models."""

    models = {
        "Linear Regression": train_linear_regression,
        "Decision Tree": train_decision_tree,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting
    }

    results = []

    for model_name, train_function in models.items():

        model, y_test, predictions = train_function(
            df,
            target_column
        )

        metrics = evaluate_regression(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R² Score": metrics["r2"]
        })

    results_df = pd.DataFrame(results)

    best_model = results_df.loc[
        results_df["RMSE"].idxmin(),
        "Model"
    ]

    return results_df, best_model


def compare_classification_models(df, target_column):
    """Compare classification models."""

    models = {
        "Logistic Regression":
            train_logistic_regression,

        "Decision Tree":
            train_decision_tree_classifier,

        "Random Forest":
            train_random_forest_classifier,

        "Gradient Boosting":
            train_gradient_boosting_classifier
    }

    results = []

    for model_name, train_function in models.items():

        model, y_test, predictions = train_function(
            df,
            target_column
        )

        metrics = evaluate_classification(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"]
        })

    results_df = pd.DataFrame(results)

    best_model = results_df.loc[
        results_df["F1 Score"].idxmax(),
        "Model"
    ]

    return results_df, best_model