# src/sentiment_analysis_service/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any # Use Any for flexibility in sentiment type if needed

# --- Request Models ---

class TextInput(BaseModel):
    """Schema for a single text input item."""
    text: str = Field(..., min_length=1, description="The text content to analyze.")
    # Example: Add optional ID if needed
    # id: Optional[str] = None

class PredictRequest(BaseModel):
    """Schema for the prediction request body."""
    inputs: List[TextInput] = Field(..., description="List of text inputs for sentiment analysis.")

# --- Response Models ---

class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    input_text: str
    sentiment: str
    error: Optional[str] = None # Include field for potential errors per item

class PredictResponse(BaseModel):
    """Schema for the prediction response body."""
    predictions: Optional[List[PredictionResult]] = None
    error: Optional[str] = None # For top-level errors (e.g., model loading failed)
    