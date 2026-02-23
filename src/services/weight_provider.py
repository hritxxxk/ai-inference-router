"""WeightProvider loads router coefficients from disk and hot-reloads them."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict

from src.config import settings

logger = logging.getLogger(__name__)


class WeightProvider:
    """Manage model routing weights with optional hot-reload support."""

    def __init__(self, weights_path: str, default_heads: Dict[str, Dict[str, float]]):
        self._path = Path(weights_path)
        self._default_heads = deepcopy(default_heads)
        self._weights: Dict[str, Dict[str, float]] = deepcopy(default_heads)
        self._mtime: float | None = None
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        if not self._path.exists():
            if force and not self._path.parent.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
            self._weights = deepcopy(self._default_heads)
            self._mtime = None
            return

        mtime = self._path.stat().st_mtime
        if not force and self._mtime == mtime:
            return

        try:
            data = json.loads(self._path.read_text())
            sanitized: Dict[str, Dict[str, float]] = {}
            for model_name, weights in data.items():
                sanitized[model_name] = {
                    key: float(value)
                    for key, value in weights.items()
                    if isinstance(value, (int, float))
                }
                if "bias" not in sanitized[model_name]:
                    sanitized[model_name]["bias"] = 0.0
            self._weights = sanitized
            self._mtime = mtime
        except Exception as exc:
            logger.warning("Failed to load router weights from %s: %s", self._path, exc)
            self._weights = deepcopy(self._default_heads)
            self._mtime = None

    def get_heads(self) -> Dict[str, Dict[str, float]]:
        self.reload()
        return self._weights


_weight_provider: WeightProvider | None = None


def get_weight_provider() -> WeightProvider:
    global _weight_provider
    if _weight_provider is None:
        _weight_provider = WeightProvider(settings.router_weights_path, settings.router_heads)
    return _weight_provider
