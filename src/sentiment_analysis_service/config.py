# src/sentiment_analysis_service/config.py
import os
from pathlib import Path

# Define the base directory of the project
# This assumes config.py is in src/sentiment_analysis_service
# Adjust if your structure differs or use environment variables for more robustness
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define paths relative to the base directory
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"  # We'll create this directory later if needed

# Create LOG_DIR if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model file name
MODEL_FILE_NAME = "sentiment_pipeline.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILE_NAME

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"  # Set log level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = LOG_DIR / "service.log"

if __name__ == "__main__":
    # Print paths to verify they are correct when running this file directly
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Log File Path: {LOG_FILE}")
