# src/sentiment_analysis_service/predict.py
import joblib
import logging
from typing import List, Dict, Any  # For type hinting
from pathlib import Path

# Import configurations and preprocessing function
from .config import MODEL_PATH, LOG_FORMAT, LOG_LEVEL, LOG_FILE  # Relative import
from .preprocessing import preprocess_batch

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE, filemode="a")
logger = logging.getLogger(__name__)

# Global variable to hold the loaded model pipeline
# Initialize to None, load lazily or on startup
_model_pipeline = None


def load_model(model_path: Path = MODEL_PATH):
    """Loads the trained pipeline from the specified path."""
    global _model_pipeline
    if _model_pipeline is not None:
        logger.info("Model pipeline already loaded.")
        return _model_pipeline

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        logger.info(f"Loading model from {model_path}...")
        _model_pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        return _model_pipeline
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)  # Log stack trace
        raise


def predict(input_data: List[str]) -> List[Dict[str, Any]]:
    """
    Makes sentiment predictions on a batch of text data.

    Args:
        input_data (List[str]): A list of raw text strings.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the
                               original text and its predicted sentiment.
                               Returns empty list on error or empty input.
    """
    global _model_pipeline
    if _model_pipeline is None:
        # Attempt to load the model if not already loaded
        # In a web server context, you'd typically load this at startup
        load_model()
        if _model_pipeline is None:  # Check again if loading failed
            logger.error("Model pipeline is not loaded. Cannot predict.")
            # Depending on requirements, could return error dicts or raise exception
            return [{"error": "Model not available"} for _ in input_data]

    if not input_data:
        logger.warning("Received empty list for prediction.")
        return []

    try:
        logger.info(f"Received {len(input_data)} items for prediction.")
        # 1. Preprocess the input text batch
        cleaned_batch = preprocess_batch(input_data)
        logger.debug(
            f"Preprocessed data: {cleaned_batch}"
        )  # Use DEBUG for verbose logs

        # 2. Make predictions using the loaded pipeline
        predictions = _model_pipeline.predict(cleaned_batch)
        logger.info(f"Generated {len(predictions)} predictions.")

        # (Optional) 3. Predict probabilities if needed
        # try:
        #     probabilities = _model_pipeline.predict_proba(cleaned_batch)
        #     # Process probabilities (e.g., get probability for the predicted class)
        # except AttributeError:
        #     logger.warning("Model pipeline does not support predict_proba.")
        #     probabilities = [None] * len(predictions) # Placeholder

        # 4. Format the output
        results = []
        for text, prediction in zip(input_data, predictions):
            results.append({"input_text": text, "sentiment": prediction})
            # Example adding probability if available:
            # results.append({"input_text": text, "sentiment": prediction, "confidence": max(prob)})

        logger.info("Prediction batch completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        # Return error indications for the whole batch or per item as needed
        return [{"input_text": text, "error": str(e)} for text in input_data]


# Example Usage (for testing the module directly)
if __name__ == "__main__":
    # Ensure model is loaded (will happen inside predict if not already)
    # load_model() # Or let predict handle it

    sample_batch = [
        "This is fantastic!",
        "This is terrible.",
        "It's okay I guess.",
        "",  # Empty string
        None,  # Invalid input type (will be handled by preprocess)
    ]

    predictions = predict(sample_batch)
    print("\nPrediction Results:")
    for result in predictions:
        print(result)

    # Check log file content in logs/service.log
    print(f"\nCheck log file at: {LOG_FILE}")
