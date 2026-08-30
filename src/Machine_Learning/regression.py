from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
import pandas as pd

from .preprocessing import (
    create_preprocessor,
    split_features_target,
    split_train_test,
    remove_id_columns,
    convert_datetime_columns
)

from .evaluation import evaluate_regression


def prepare_data(df, target_column):

    X, y = split_features_target(
        df,
        target_column
    )

    X = remove_id_columns(X)

    X = convert_datetime_columns(X)

    return X, y


def train_regression_model(
    df,
    target_column,
    model
):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

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

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, y_test, predictions


def train_linear_regression(df, target_column):

    model = LinearRegression()

    return train_regression_model(
        df,
        target_column,
        model
    )


def train_decision_tree(df, target_column):

    model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )

    return train_regression_model(
        df,
        target_column,
        model
    )


def train_random_forest(df, target_column):

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    return train_regression_model(
        df,
        target_column,
        model
    )


def train_gradient_boosting(df, target_column):

    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )

    return train_regression_model(
        df,
        target_column,
        model
    )


def train_model_for_prediction(
    df,
    target_column
):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X,
        y
    )

    return pipeline, X




