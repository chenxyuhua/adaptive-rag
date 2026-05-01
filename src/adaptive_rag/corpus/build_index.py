"""Build a FAISS index over a Wikipedia passage subset.

Outputs two files in `out_dir`:
  - wiki_subset.faiss   (FAISS index, IndexFlatIP if normalized embeddings)
  - wiki_subset.jsonl   (one passage per line: {doc_id, title, text, url, source})

Designed to run on Colab GPU (one-shot) or locally on CPU. The same script
covers both — set `device='cuda'` on Colab and re-run.

Default corpus: `wiki_dpr` (DPR's Wikipedia passage dump, 100-word chunks).
Truncate to `num_passages` for a manageable subset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def build(
    out_dir: str,
    num_passages: int = 200_000,
    embedder_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    device: str = "cpu",
    hf_dataset: str = "wiki_dpr",
    hf_config: str = "psgs_w100.nq.no_index.no_embeddings",
    seed: int = 42,
) -> dict:
    """Build the index. Returns a dict of paths and stats."""
    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    passages_path = out / "wiki_subset.jsonl"
    index_path = out / "wiki_subset.faiss"

    print(f"[corpus] Loading {hf_dataset}/{hf_config}...")
    ds = load_dataset(hf_dataset, hf_config, split="train", trust_remote_code=True)

    if num_passages < len(ds):
        # Deterministic subset.
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(ds), size=num_passages, replace=False)
        idxs.sort()
        ds = ds.select(idxs.tolist())

    print(f"[corpus] Materializing {len(ds)} passages to {passages_path}")
    with passages_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(tqdm(ds, desc="passages")):
            payload = {
                "doc_id": str(row.get("id", i)),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "source": "wiki",
            }
            fh.write(json.dumps(payload) + "\n")

    print(f"[corpus] Loading embedder {embedder_model} on {device}")
    model = SentenceTransformer(embedder_model, device=device)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # cosine sim with normalized embeddings

    texts = [str(row.get("text", "")) for row in ds]
    n = len(texts)
    print(f"[corpus] Embedding {n} passages (batch_size={batch_size})")
    for start in tqdm(range(0, n, batch_size), desc="embed"):
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        ).astype(np.float32)
        index.add(emb)

    faiss.write_index(index, str(index_path))
    print(f"[corpus] Wrote index to {index_path} (n={index.ntotal}, dim={dim})")

    return {
        "index_path": str(index_path),
        "passages_path": str(passages_path),
        "num_passages": n,
        "embedder": embedder_model,
        "dim": dim,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build FAISS index over Wikipedia subset.")
    p.add_argument("--out-dir", default="data/index")
    p.add_argument("--num-passages", type=int, default=200_000)
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "cpu"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    info = build(
        out_dir=args.out_dir,
        num_passages=args.num_passages,
        embedder_model=args.embedder,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(info, indent=2))
