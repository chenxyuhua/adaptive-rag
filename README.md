# adaptive-rag

Selective retrieval-augmented generation: instead of always retrieving, the model decides per-query whether retrieval is needed. This repository hosts the **core RAG pipeline + baselines** that the rest of the team builds on.

## What's in here

- `src/adaptive_rag/` — the package: schemas, LLM client, retriever, strategies, pipeline, eval stubs
- `configs/` — YAML configs (start with `default.yaml`)
- `prompts/` — prompt templates for no-retrieval and with-retrieval generation
- `scripts/` — runnable CLIs (`run_baseline.py`, `build_index.py`)
- `tests/` — smoke tests
- `data/`, `runs/` — gitignored; corpus indexes and prediction logs land here

## Quickstart

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Get a free Gemini API key
Visit https://aistudio.google.com/app/apikey, create a key, then:
```bash
cp .env.example .env
# Edit .env and paste the key into GOOGLE_API_KEY
```

### 3. Build the corpus index (one-shot, on Colab GPU)
The Wikipedia subset is too large to embed on a laptop. Open `scripts/build_index_colab.ipynb` (TODO) on Colab with a GPU runtime, run all cells, and download the resulting `wiki_subset.faiss` + `wiki_subset.jsonl` into `data/index/`.

### 4. Run a baseline
```bash
python scripts/run_baseline.py --dataset nq --strategy fixed_k --k 5 --n 50
```
Outputs land in `runs/{dataset}/{strategy}/{run_id}/predictions.jsonl`.

## Strategies

- `no_retrieval` — base LLM answers without external evidence
- `fixed_k` — top-k retrieval for every query (the standard RAG baseline)
- `adaptive` — stub; the adaptive policy plugs in here

## Standard prediction record (for downstream evaluation)

Every run writes one JSONL line per query with this schema. Evaluation/analysis code reads this format directly.

```json
{
  "qid": "...", "dataset": "nq", "question": "...",
  "gold_answers": ["..."],
  "strategy": "fixed_k", "strategy_config": {"k": 5},
  "retrieved": [{"doc_id": "...", "score": 0.83, "text": "...", "source": "wiki"}],
  "prompt": "...", "raw_answer": "...", "parsed_answer": "...",
  "latency_ms": 412, "prompt_tokens": 1024, "completion_tokens": 18,
  "retrieval_calls": 1, "retrieved_token_count": 950,
  "model": "gemini-2.0-flash", "config_hash": "abc123",
  "timestamp": "2026-04-29T12:34:56Z"
}
```

## Roadmap (Caroline's slice)

- [x] Repo skeleton + I/O contract
- [ ] Dataset loaders (NQ, TriviaQA, HotpotQA)
- [ ] FAISS retriever + Colab corpus build notebook
- [ ] Gemini LLM client
- [ ] Baseline strategies + pipeline runner
- [ ] EM/F1 eval stubs + prediction loader
- [ ] CLI scripts + smoke tests
- [ ] Baseline runs (200/dataset, 3 strategies)
