"""SQLite-backed storage for router telemetry and feedback."""

from __future__ import annotations

import json
import logging
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from src.config import settings

logger = logging.getLogger(__name__)


def _serialize_json(payload: Dict[str, Any] | Sequence[Any] | None) -> Optional[str]:
    if payload is None:
        return None
    return json.dumps(payload, separators=(",", ":"))


def _encode_embedding(vector: Optional[Sequence[float]]) -> Tuple[Optional[bytes], Optional[int]]:
    if vector is None:
        return None, None
    try:
        buf = array("f", (float(x) for x in vector))
    except (TypeError, ValueError):
        logger.warning("Skipping embedding serialization due to invalid payload")
        return None, None
    return buf.tobytes(), len(buf)


class TelemetryStore:
    """Persist decisions, outcomes, and feedback for HITL training."""

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        if self._db_path.parent:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_decisions (
                    request_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_preview TEXT,
                    target_model TEXT,
                    fallback_model TEXT,
                    confidence REAL,
                    task_type TEXT,
                    features TEXT,
                    reasons TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_outcomes (
                    request_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_preview TEXT,
                    model_used TEXT,
                    latency_ms REAL,
                    model_latency_sec REAL,
                    cost REAL,
                    cached INTEGER,
                    fallback_used INTEGER,
                    routing_confidence REAL,
                    task_type TEXT,
                    extra TEXT,
                    FOREIGN KEY(request_id) REFERENCES telemetry_decisions(request_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    preferred_model TEXT,
                    reviewer TEXT,
                    notes TEXT,
                    quality_score INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES telemetry_decisions(request_id) ON DELETE CASCADE
                )
                """
            )

    def persist_decision(
        self,
        request_id: str,
        client_id: str,
        prompt_preview: str,
        prompt_hash: str,
        decision_payload: Dict[str, Any],
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        features_json = _serialize_json(decision_payload.get("features"))
        reasons_json = _serialize_json(decision_payload.get("reasons"))
        embedding_blob, embedding_dim = _encode_embedding(embedding)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO telemetry_decisions (
                    request_id, ts, client_id, prompt_hash, prompt_preview,
                    target_model, fallback_model, confidence, task_type, features,
                    reasons, embedding, embedding_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    ts,
                    client_id,
                    prompt_hash,
                    prompt_preview,
                    decision_payload.get("target_model"),
                    decision_payload.get("fallback_model"),
                    decision_payload.get("confidence"),
                    decision_payload.get("task_type"),
                    features_json,
                    reasons_json,
                    embedding_blob,
                    embedding_dim,
                ),
            )

    def persist_outcome(
        self,
        request_id: str,
        client_id: str,
        prompt_preview: str,
        prompt_hash: str,
        outcome_payload: Dict[str, Any],
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        cached = outcome_payload.get("cached")
        extra_payload = {
            k: v
            for k, v in outcome_payload.items()
            if k
            not in {
                "cached",
                "model_used",
                "latency_ms",
                "model_latency_sec",
                "cost",
                "fallback_used",
                "routing_confidence",
                "task_type",
            }
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO telemetry_outcomes (
                    request_id, ts, client_id, prompt_hash, prompt_preview, model_used, latency_ms,
                    model_latency_sec, cost, cached, fallback_used, routing_confidence,
                    task_type, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    ts,
                    client_id,
                    prompt_hash,
                    prompt_preview,
                    outcome_payload.get("model_used"),
                    outcome_payload.get("latency_ms"),
                    outcome_payload.get("model_latency_sec"),
                    outcome_payload.get("cost"),
                    int(bool(cached)) if cached is not None else None,
                    int(bool(outcome_payload.get("fallback_used")))
                    if outcome_payload.get("fallback_used") is not None
                    else None,
                    outcome_payload.get("routing_confidence"),
                    outcome_payload.get("task_type"),
                    _serialize_json(extra_payload) if extra_payload else None,
                ),
            )

    def record_feedback(
        self,
        request_id: str,
        label: str,
        preferred_model: Optional[str],
        reviewer: Optional[str],
        notes: Optional[str],
        quality_score: Optional[int],
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    request_id, label, preferred_model, reviewer, notes, quality_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    label,
                    preferred_model,
                    reviewer,
                    notes,
                    quality_score,
                    ts,
                ),
            )

    def decision_exists(self, request_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM telemetry_decisions WHERE request_id = ? LIMIT 1",
                (request_id,),
            )
            return cursor.fetchone() is not None

    def feedback_count(self, request_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(1) FROM feedback WHERE request_id = ?",
                (request_id,),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0


telemetry_store = TelemetryStore(settings.telemetry_db_path)
