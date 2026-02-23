#!/usr/bin/env python
"""Train router weights from telemetry + human feedback."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.config import settings
from src.services.telemetry_store import TelemetryStore

logger = logging.getLogger(__name__)


def _label_to_target(label: str) -> float:
    return 1.0 if label == "correct" else 0.0


def _load_rows(conn: sqlite3.Connection) -> Tuple[List[str], Dict[str, List[Tuple[Dict[str, float], float, float]]]]:
    cursor = conn.execute(
        """
        SELECT d.target_model, d.features, f.label, COALESCE(f.quality_score, 5)
        FROM telemetry_decisions AS d
        JOIN feedback AS f ON d.request_id = f.request_id
        WHERE d.features IS NOT NULL AND f.label IS NOT NULL
        """
    )
    feature_names: List[str] = []
    per_model: Dict[str, List[Tuple[Dict[str, float], float, float]]] = defaultdict(list)
    seen_features = set()
    for model_name, features_json, label, quality_score in cursor.fetchall():
        try:
            features = json.loads(features_json)
        except (TypeError, json.JSONDecodeError):
            continue
        target = _label_to_target(label)
        if target not in {0.0, 1.0}:
            continue
        weight = max(1.0, float(quality_score)) / 5.0
        per_model[model_name].append((features, target, weight))
        for key in features.keys():
            if key not in seen_features:
                seen_features.add(key)
                feature_names.append(key)
    return feature_names, per_model


def _train_heads(default_heads: Dict[str, Dict[str, float]], feature_names: Iterable[str], samples: Dict[str, List[Tuple[Dict[str, float], float, float]]]) -> Dict[str, Dict[str, float]]:
    trained: Dict[str, Dict[str, float]] = {}
    ordered_features = list(dict.fromkeys(feature_names))  # preserve insertion order
    if not ordered_features:
        ordered_features = sorted({k for head in default_heads.values() for k in head.keys() if k != "bias"})

    for model_name, rows in samples.items():
        if len(rows) < 3:
            trained[model_name] = default_heads.get(model_name, {})
            continue
        X_rows: List[List[float]] = []
        y_vals: List[float] = []
        for features, target, sample_weight in rows:
            vector = [1.0] + [features.get(name, 0.0) for name in ordered_features]
            scaled_vector = [val * sample_weight for val in vector]
            X_rows.append(scaled_vector)
            y_vals.append(target * sample_weight)
        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_vals, dtype=np.float32)
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            logger.warning("Least squares failed for %s: %s", model_name, exc)
            trained[model_name] = default_heads.get(model_name, {})
            continue
        weights = {"bias": float(coeffs[0])}
        for idx, feature in enumerate(ordered_features, start=1):
            weights[feature] = float(coeffs[idx]) if idx < len(coeffs) else 0.0
        trained[model_name] = weights
    return trained


def train_router(db_path: str, output_path: str) -> Path:
    TelemetryStore(db_path)  # ensure schema exists
    conn = sqlite3.connect(db_path)
    feature_names, per_model_samples = _load_rows(conn)
    conn.close()

    if not per_model_samples:
        logger.warning("No training rows found; falling back to config defaults")
        weights = deepcopy(settings.router_heads)
    else:
        trained = _train_heads(settings.router_heads, feature_names, per_model_samples)
        weights = deepcopy(settings.router_heads)
        weights.update(trained)

    output = Path(output_path)
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(weights, indent=2, sort_keys=True))
    logger.info("Wrote %d router heads to %s", len(weights), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router weights from telemetry + feedback")
    parser.add_argument("--db", default=settings.telemetry_db_path, help="Path to router sqlite database")
    parser.add_argument("--out", default=settings.router_weights_path, help="Destination JSON for trained weights")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_router(args.db, args.out)


if __name__ == "__main__":
    main()
