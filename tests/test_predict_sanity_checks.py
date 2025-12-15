import pytest
from joblib import load
from src.predict import predict_texts, load_model

# Define the expected labels
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

@pytest.fixture(scope="module")
def trained_classifier():
    """
    Fixture that loads the actual, trained sentiment model once per test module.
    
    This ensures the 'load_model' function is tested and the actual model
    is used for our sanity checks.
    """

    # Path used in predict.py
    model_path = "models/sentiment.joblib"

    try:
        classifier = load_model(model_path)
        return classifier
    except FileNotFoundError:
        # If model file is missing, skip the test that rely on it
        pytest.skip(
            f"Sanity check skipped: Model file not found at '{model_path}'. "
            "Please ensure a small, pre-trained model is available"
        )
    except Exception as e:
        # Handle other loading errors
        pytest.fail(f"Failed to load model from '{model_path}' : {e}")

def test_sanity_check_positive_sentence(trained_classifier):
    """
    Sanity Check 1: Passes an obviously postive sentence and asserts the prediction is Positive (1).
    This confirms the model is loading and making reasonable predictions.
    """

    expected_pred = POSITIVE_LABEL

    input_text = ["I love this movie, it was fantastic and inspiring! The best thing I've seen all year."]

    preds, probs = predict_texts(
        classifier = trained_classifier,
        input_texts = input_text
    )

    # Prediction must match the expected positive label
    assert preds == [expected_pred], (
        f"Positive Sanity Check failed: Expected prediction {expected_pred}, "
        f"but got {preds[0]} for text: '{input_text[0]}'"
    )

def test_sanity_check_negative_sentence(trained_classifier):
    """
    Sanity Check 2: Passes an obviously negative sentence and asserts the prediction is Negative (0).
    This confirms the prediction pipeline's handling of the negative class.
    """

    expected_pred = NEGATIVE_LABEL

    input_text = ["The service was terrible and the food was awful. I will never come back to this restaurant."]

    preds, probs = predict_texts(
        classifier=trained_classifier,
        input_texts=input_text
    )

    # The prediction must match the expected negative label
    assert preds == [expected_pred], (
        f"Negative Sanity Check failed: Expected prediction {expected_pred}, "
        f"but got {preds[0]} for text: '{input_text[0]}'"
    )
