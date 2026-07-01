import pytest
from pydantic import ValidationError

from app.schemas import AnalyzeRequest, BatchRequest


def test_analyze_rejects_blank_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="   ")


def test_analyze_rejects_overlong_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="x" * 2001)


def test_analyze_strips_whitespace():
    assert AnalyzeRequest(text="  hello  ").text == "hello"


def test_batch_rejects_blank_items():
    with pytest.raises(ValidationError):
        BatchRequest(texts=["fine", "   "])


def test_batch_rejects_over_500_items():
    with pytest.raises(ValidationError):
        BatchRequest(texts=["x"] * 501)
