from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.pipeline import Pipeline

from .preprocessing import (
    create_preprocessor,
    split_features_target,
    split_train_test,
    remove_id_columns,
    convert_datetime_columns
)


def prepare_data(df, target_column):

    X, y = split_features_target(
        df,
        target_column
    )

    X = remove_id_columns(X)

    X = convert_datetime_columns(X)

    return X, y


def train_logistic_regression(df, target_column):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = LogisticRegression(
        max_iter=500,
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

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, y_test, predictions


def train_decision_tree_classifier(df, target_column):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10
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

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, y_test, predictions


def train_random_forest_classifier(df, target_column):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
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

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, y_test, predictions


def train_gradient_boosting_classifier(df, target_column):

    X, y = prepare_data(
        df,
        target_column
    )

    preprocessor = create_preprocessor(X)

    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=50,
        max_depth=3
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

    predictions = pipeline.predict(
        X_test
    )

    return pipeline, y_test, predictions