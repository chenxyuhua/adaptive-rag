"""FAISS-backed dense retriever.

At query time:
  1. Embed the query with the same SentenceTransformer used at index time.
  2. Search the FAISS index for top-k passage IDs.
  3. Look up the passage texts from a side JSONL file.

The index file (.faiss) and passages file (.jsonl) are produced by
`scripts/build_index.py` (or `scripts/build_index_colab.ipynb` for GPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schemas import RetrievedDoc
from .base import Retriever


class FaissRetriever(Retriever):
    def __init__(
        self,
        index_path: str | Path,
        passages_path: str | Path,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_path = Path(index_path)
        self.passages_path = Path(passages_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Build it first via scripts/build_index.py (or the Colab notebook)."
            )
        if not self.passages_path.exists():
            raise FileNotFoundError(
                f"Passages JSONL not found at {self.passages_path}."
            )

        self._faiss = faiss
        self.index = faiss.read_index(str(self.index_path))
        self.embedder = SentenceTransformer(embedder_model, device=device)
        self.normalize = normalize
        self.passages = self._load_passages(self.passages_path)

    @staticmethod
    def _load_passages(path: Path) -> list[dict]:
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        emb = self.embedder.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, idxs = self.index.search(emb, k)

        out: list[RetrievedDoc] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.passages):
                continue
            row = self.passages[idx]
            out.append(
                RetrievedDoc(
                    doc_id=str(row.get("doc_id", idx)),
                    score=float(score),
                    text=row.get("text", ""),
                    source=row.get("source", "wiki"),
                    meta={k: row[k] for k in ("title", "url") if k in row},
                )
            )
        return out
