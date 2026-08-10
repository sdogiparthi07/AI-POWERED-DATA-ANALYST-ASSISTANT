def make_prediction(model, data):
    """Generate predictions using a trained model."""

    predictions = model.predict(data)

    return predictions


def make_single_prediction(model, data):
    """Generate a prediction for a single record."""

    prediction = model.predict(data)

    return prediction[0]