# tests/test_preprocessing.py
import pytest  # Import pytest
from sentiment_analysis_service.preprocessing import (
    preprocess_text,
    preprocess_batch,
)  # Import functions to test

# --- Tests for preprocess_text ---


def test_preprocess_text_standard():
    """Test standard case with mixed case and punctuation."""
    assert (
        preprocess_text("This Product is AMAZING!! Highly recommend.")
        == "this product is amazing highly recommend"
    )


def test_preprocess_text_extra_spaces():
    """Test handling of leading/trailing/multiple spaces."""
    assert preprocess_text("  Extra   spaces  here  ") == "extra spaces here"


def test_preprocess_text_only_punctuation():
    """Test text containing only punctuation."""
    assert preprocess_text("!!!???.") == ""


def test_preprocess_text_empty_string():
    """Test empty string input."""
    assert preprocess_text("") == ""


def test_preprocess_text_non_string():
    """Test non-string input (should return empty string)."""
    assert preprocess_text(12345) == ""
    assert preprocess_text(None) == ""
    assert preprocess_text(["list", "is", "not", "string"]) == ""


def test_preprocess_text_numbers_and_text():
    """Test text containing numbers (numbers should remain)."""
    assert (
        preprocess_text("Version 5.0 is great!") == "version 50 is great"
    )  # Note: '.' is removed


# --- Tests for preprocess_batch ---


def test_preprocess_batch_standard():
    """Test batch preprocessing with various inputs."""
    inputs = [
        "First text.",
        "SECOND TEXT WITH CAPITALS!!",
        "  spaces galore  ",
        123,  # Non-string
        "",  # Empty string
    ]
    expected = ["first text", "second text with capitals", "spaces galore", "", ""]
    assert preprocess_batch(inputs) == expected


def test_preprocess_batch_empty_list():
    """Test batch preprocessing with an empty list."""
    assert preprocess_batch([]) == []


# Optional: Parameterized testing for cleaner code (more advanced pytest feature)
@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Simple case.", "simple case"),
        ("UPPERCASE!!", "uppercase"),
        ("  leading/trailing spaces   ", "leadingtrailing spaces"),
        ("", ""),
        ("Text with numbers 123", "text with numbers 123"),
    ],
)
def test_preprocess_text_parameterized(input_text, expected_output):
    """Example of parameterized test for preprocess_text."""
    assert preprocess_text(input_text) == expected_output
