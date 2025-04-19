# tests/test_predict.py
import pytest
from pathlib import Path
import joblib # To potentially inspect the loaded model if needed

# Assuming tests are run from the project root directory
# If not, adjust paths accordingly or use fixtures
from sentiment_analysis_service.predict import load_model, predict
from sentiment_analysis_service.config import MODEL_PATH, MODEL_FILE_NAME, MODEL_DIR

# --- Test Fixture for Model Loading (Optional but good practice) ---
# This fixture ensures the model is loaded only once per test session if needed,
# but for simplicity, we can let the predict function handle loading in each test.
# Let's stick to the simpler approach for now. If tests become slow, use fixtures.

# --- Tests for load_model ---

def test_load_model_success():
    """Test if the model loads successfully from the correct path."""
    # Ensure the model file actually exists before running the test
    assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}. Run the notebook first."

    # Reset the internal state if needed (or rely on lazy loading logic)
    # Forcing reload requires accessing the private variable or a dedicated reset function
    # Let's assume predict() handles lazy loading correctly for simplicity here.
    # Or, explicitly call load_model and check the return type.
    pipeline = load_model()
    assert pipeline is not None
    # Check if it looks like a scikit-learn pipeline (basic check)
    assert hasattr(pipeline, 'steps')
    assert hasattr(pipeline, 'predict')

def test_load_model_file_not_found(tmp_path):
    """Test FileNotFoundError when model path is incorrect."""
    # Use pytest's tmp_path fixture for a temporary directory
    non_existent_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(model_path=non_existent_path)

# --- Tests for predict function ---

# Sample valid inputs for prediction tests
SAMPLE_INPUTS = [
    "This product is amazing! Highly recommend.",
    "Very disappointed with the quality.",
    "Works okay, but not great.",
]
EXPECTED_KEYS = {"input_text", "sentiment"} # Expected keys in the output dict

def test_predict_valid_input():
    """Test prediction with a valid batch of text."""
    predictions = predict(SAMPLE_INPUTS)

    assert isinstance(predictions, list)
    assert len(predictions) == len(SAMPLE_INPUTS)

    # Check structure and types of each prediction dict
    for i, result in enumerate(predictions):
        assert isinstance(result, dict)
        assert result.keys() == EXPECTED_KEYS
        assert isinstance(result["input_text"], str)
        assert result["input_text"] == SAMPLE_INPUTS[i]
        assert isinstance(result["sentiment"], str)
        # Check if sentiment is one of the expected classes (adjust if your model has different ones)
        assert result["sentiment"] in ["positive", "negative", "neutral"]

def test_predict_empty_list():
    """Test prediction with an empty list."""
    predictions = predict([])
    assert predictions == []

def test_predict_mixed_types_input():
    """Test prediction with inputs including non-strings."""
    mixed_inputs = [
        "Good stuff!",
        12345, # Invalid type
        "This is bad.",
        None   # Invalid type
    ]
    expected_count = len(mixed_inputs)
    predictions = predict(mixed_inputs)

    assert isinstance(predictions, list)
    assert len(predictions) == expected_count

    # Check that valid inputs got predictions and invalid ones were handled (based on preprocess)
    assert predictions[0]["sentiment"] in ["positive", "negative", "neutral"]
    assert predictions[1]["sentiment"] in ["positive", "negative", "neutral"] # Preprocess converts int to "" -> likely neutral/neg prediction
    assert predictions[2]["sentiment"] in ["positive", "negative", "neutral"]
    assert predictions[3]["sentiment"] in ["positive", "negative", "neutral"] # Preprocess converts None to "" -> likely neutral/neg prediction


def test_predict_single_item_list():
    """Test prediction with a list containing a single item."""
    single_input = ["Excellent customer service"]
    predictions = predict(single_input)
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert predictions[0]["input_text"] == single_input[0]
    assert predictions[0]["sentiment"] in ["positive", "negative", "neutral"]

# Note: Testing for specific prediction outputs (e.g., assert predict("good")[0]['sentiment'] == 'positive')
# can be brittle if the model changes slightly during retraining. It's often better to test the
# structure, types, and validity of the output, rather than exact prediction values, unless
# specific known cases MUST yield a certain result (golden tests).