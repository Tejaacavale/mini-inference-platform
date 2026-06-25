"""Pydantic request/response models for the inference API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Single input text to classify.")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label, e.g. POSITIVE/NEGATIVE.")
    score: float = Field(..., description="Confidence score for the predicted label.")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of input texts to classify.")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
