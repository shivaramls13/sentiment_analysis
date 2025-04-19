# src/sentiment_analysis_service/preprocessing.py
import re
import string
from typing import List  # Use List for type hinting


def preprocess_text(text: str) -> str:
    """
    Applies basic text cleaning:
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text string. Returns empty string if input is not a string.
    """
    if not isinstance(text, str):
        # Handle potential non-string data gracefully
        # Consider logging a warning here if appropriate
        return ""
    text = text.lower()
    # Ensure punctuation removal handles unicode correctly if necessary
    # Using str.maketrans is generally efficient
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(
        r"\s+", " ", text
    ).strip()  # Replace multiple spaces with single and strip ends
    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Applies preprocessing to a list of text strings.

    Args:
        texts (List[str]): A list of text strings.

    Returns:
        List[str]: A list of cleaned text strings.
    """
    return [preprocess_text(text) for text in texts]


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    sample_texts = [
        "This Product is AMAZING!! Highly recommend.",
        "Works okay, but not great...",
        12345,  # Example of non-string input
        "  Extra   spaces  here  ",
    ]
    cleaned_texts = preprocess_batch(sample_texts)
    for original, cleaned in zip(sample_texts, cleaned_texts):
        print(f"Original: '{original}'\nCleaned:  '{cleaned}'\n---")
