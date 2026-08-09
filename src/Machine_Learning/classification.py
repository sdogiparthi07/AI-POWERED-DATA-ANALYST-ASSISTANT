from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import (
    create_preprocessor,
    split_features_target,
    split_train_test
)


def train_logistic_regression(df, target_column):
    """Train a Logistic Regression classification model."""

    X, y = split_features_target(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = LogisticRegression(
        max_iter=1000
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    return pipeline, y_test, predictions


def train_random_forest_classifier(df, target_column):
    """Train a Random Forest classification model."""

    X, y = split_features_target(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    return pipeline, y_test, predictions