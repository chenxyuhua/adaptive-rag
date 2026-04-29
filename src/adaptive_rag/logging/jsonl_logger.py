"""Append-only JSONL writer for prediction records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import PredictionRecord


class JsonlLogger:
    """Writes one PredictionRecord per line. Streams to disk so a crash mid-run
    does not lose previously-completed predictions.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: PredictionRecord) -> None:
        self._fh.write(record.model_dump_json() + "\n")
        self._fh.flush()

    def write_meta(self, side_path: str, payload: dict[str, Any]) -> None:
        Path(side_path).parent.mkdir(parents=True, exist_ok=True)
        with open(side_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
