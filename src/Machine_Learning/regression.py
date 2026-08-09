from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import (
    create_preprocessor,
    split_features_target,
    split_train_test
)


def train_linear_regression(df, target_column):
    """Train a Linear Regression model."""

    X, y = split_features_target(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = LinearRegression()

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


def train_random_forest(df, target_column):
    """Train a Random Forest regression model."""

    X, y = split_features_target(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = RandomForestRegressor(
        n_estimators=10,
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


def train_model_for_prediction(df, target_column):
    """Train a model and prepare the features used for prediction."""

    X, y = split_features_target(
        df,
        target_column
    )

    # Remove obvious ID columns
    prediction_columns = [
        column
        for column in X.columns
        if "id" not in column.lower()
    ]

    X = X[prediction_columns]

    preprocessor = create_preprocessor(X)

    model = RandomForestRegressor(
        n_estimators=10,
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

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline, X