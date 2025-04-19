# src/sentiment_analysis_service/main.py
import logging
import time # Import time module for latency calculation
import json # Import json for structured logging
from fastapi import FastAPI, HTTPException, Request, Response # Added Response
from fastapi.responses import JSONResponse

# Import schemas, config, and prediction function
from .schemas import PredictRequest, PredictResponse, PredictionResult
from .config import LOG_FORMAT, LOG_LEVEL, LOG_FILE, MODEL_PATH
from .predict import predict, load_model, _model_pipeline # Import _model_pipeline for check
from . import __version__

# --- Logging Setup ---
# Configure logging (ensure it's set up before FastAPI instance)
# Consider using a JSON formatter for better parsing in Cloud Logging later
# For now, stick to basic formatting configured in config.py
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE, filemode='a')
logger = logging.getLogger(__name__)

# --- Application Metadata ---
DESCRIPTION = """
Sentiment Analysis API using a simple TF-IDF + Logistic Regression model.

Use the /predict endpoint to classify text passages. Includes basic monitoring logs.
"""

# --- FastAPI App Instance ---
app = FastAPI(
    title="Sentiment Analysis API",
    description=DESCRIPTION,
    version=__version__,
    contact={ # ... (keep metadata as before) ...
    },
    license_info={ # ... (keep metadata as before) ...
    },
)

# --- Middleware for Request Logging and Timing ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests and calculate processing time."""
    start_time = time.time()
    # Log basic request info before processing
    # logger.info(f"Request started: {request.method} {request.url.path}") # Can be verbose

    response = await call_next(request) # Process the request

    process_time = (time.time() - start_time) * 1000 # Calculate time in ms
    formatted_process_time = f'{process_time:.2f}'

    # Log basic info after processing, including status code and time
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"Status={response.status_code} Duration={formatted_process_time}ms"
    )
    # Add custom header with process time
    response.headers["X-Process-Time-Ms"] = formatted_process_time
    return response


# --- Model Loading on Startup ---
@app.on_event("startup")
async def startup_event():
    """Load the ML model when the application starts."""
    logger.info("Application startup: Loading model...")
    try:
        load_model() # Call the load_model function from predict.py
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Application startup: Failed to load model: {e}", exc_info=True)


# --- Custom Exception Handler ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handles unexpected errors gracefully."""
    # Log the full error with stack trace for internal debugging
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    # Return a generic error message to the client
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": "An internal server error occurred."}},
    )


# --- API Endpoints ---

@app.get("/", tags=["General"])
# ... (keep root endpoint as before) ...
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Sentiment Analysis API!", "version": __version__}


@app.get("/health", tags=["General"])
# ... (keep health endpoint as before, maybe add check for model object) ...
async def health_check():
    model_loaded = _model_pipeline is not None # Check if model object exists
    status = "OK" if model_loaded else "ERROR"
    status_code = 200 if model_loaded else 503 # 503 Service Unavailable
    logger.info(f"Health check performed. Model loaded: {model_loaded}")
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "model_loaded": model_loaded}
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def post_predict(request: PredictRequest) -> PredictResponse:
    """
    Perform sentiment analysis on a batch of text inputs.

    Logs input summary and prediction results for monitoring.
    """
    request_id = request.headers.get("X-Request-ID", "N/A") # Get request ID if available from upstream (e.g., API Gateway/LB)
    num_items = len(request.inputs)
    logger.info(f"Prediction request received. RequestID={request_id}, Items={num_items}")

    # Check if model is loaded (important after startup)
    if _model_pipeline is None:
        logger.error("Prediction attempt failed: Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not available. Please check service health.")

    # Extract the raw text strings from the request model
    try:
        input_texts = [item.text for item in request.inputs]
    except Exception as e:
        logger.error(f"Error extracting text from request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid request format.")

    # Call the prediction logic from predict.py
    try:
        prediction_dicts = predict(input_texts) # This function already has some logging
        prediction_results = [PredictionResult(**p) for p in prediction_dicts]

        # --- Enhanced Logging for Monitoring ---
        # Log summary and potentially sampled data
        # Avoid logging all data if volume is high or data is sensitive
        log_entry = {
            "request_id": request_id,
            "input_count": len(input_texts),
            "output_count": len(prediction_results),
            # Example: Log first input text (truncated) and its prediction
            "sample_input": input_texts[0][:100] if input_texts else None, # Log first 100 chars
            "sample_output": prediction_results[0].model_dump() if prediction_results else None, # Use .model_dump() for Pydantic v2+
            # Example: Log distribution of predictions in this batch
            "prediction_distribution": {
                sentiment: sum(1 for p in prediction_results if p.sentiment == sentiment)
                for sentiment in set(p.sentiment for p in prediction_results if p.sentiment)
            }
        }
        # Use json.dumps for structured logging (easier parsing in Cloud Logging)
        logger.info(f"Prediction batch processed: {json.dumps(log_entry)}")

        return PredictResponse(predictions=prediction_results)

    except FileNotFoundError as e:
         logger.error(f"Prediction error: Model file not found: {e}", exc_info=True)
         raise HTTPException(status_code=503, detail="Model not available.")
    except Exception as e:
        logger.error(f"Prediction error: An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")