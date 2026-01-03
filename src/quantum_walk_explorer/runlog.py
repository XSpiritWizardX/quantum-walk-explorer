"""Run logging helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union


def create_payload(
    graph: dict[str, Any],
    params: dict[str, Any],
    times,
    probabilities,
) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graph": graph,
        "params": params,
        "times": list(times),
        "probabilities": probabilities,
    }


def save_run(path: Union[str, Path], payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_run(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
