"""Structured telemetry logging for routing decisions and outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from src.config import settings
from src.services.telemetry_store import telemetry_store

logger = logging.getLogger(settings.telemetry_channel)


def _prompt_digest(prompt: str) -> Dict[str, Any]:
    preview = prompt.strip().replace("\n", " ")[:120]
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {"prompt_preview": preview, "prompt_hash": digest}


def log_event(event_type: str, request_id: str, payload: Dict[str, Any]) -> None:
    if not settings.telemetry_enabled:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "request_id": request_id,
        **payload,
    }
    logger.info(json.dumps(record))


def _submit_background(func, *args, **kwargs) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        func(*args, **kwargs)
    else:
        loop.create_task(asyncio.to_thread(func, *args, **kwargs))


def log_routing_decision(
    request_id: str,
    client_id: str,
    prompt: str,
    decision_payload: Dict[str, Any],
    prompt_embedding: Optional[Sequence[float]] = None,
) -> None:
    if not settings.telemetry_enabled:
        return
    digest = _prompt_digest(prompt)
    payload = {"client_id": client_id, **digest, **decision_payload}
    log_event("routing_decision", request_id, payload)
    _submit_background(
        telemetry_store.persist_decision,
        request_id,
        client_id,
        digest["prompt_preview"],
        digest["prompt_hash"],
        decision_payload,
        prompt_embedding,
    )


def log_routing_outcome(
    request_id: str,
    client_id: str,
    prompt: str,
    outcome_payload: Dict[str, Any],
) -> None:
    if not settings.telemetry_enabled:
        return
    digest = _prompt_digest(prompt)
    payload = {"client_id": client_id, **digest, **outcome_payload}
    log_event("routing_outcome", request_id, payload)
    _submit_background(
        telemetry_store.persist_outcome,
        request_id,
        client_id,
        digest["prompt_preview"],
        digest["prompt_hash"],
        outcome_payload,
    )


def log_feedback_event(request_id: str, payload: Dict[str, Any]) -> None:
    if not settings.telemetry_enabled:
        return
    log_event("routing_feedback", request_id, payload)
