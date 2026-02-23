"""Security helpers for lightweight API key enforcement."""

from fastapi import Header, HTTPException, status

from src.config import settings


def verify_feedback_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    """Ensure the caller is authorized to submit feedback."""

    if not settings.feedback_api_enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Feedback API disabled")

    expected = settings.feedback_api_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Feedback API key not configured")

    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
