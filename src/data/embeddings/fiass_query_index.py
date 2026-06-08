"""Simple querying script for the FAISS index and metadata.

Usage:
    python query_index.py "your query here" --k 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

ROOT = Path(__file__).resolve().parents[3]
# Use the repository root so the script works no matter the current working directory.
PROC = ROOT / "data" / "processed"
INDEX_PATH = PROC / "kb_index.faiss"
META_PATH = PROC / "kb_meta.json"


def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index file not found at {INDEX_PATH}. Run embed.py first (create the index in {PROC}).")
    if faiss is None:
        raise RuntimeError("faiss not installed")
    return faiss.read_index(str(INDEX_PATH))


def load_meta():
    return json.loads(META_PATH.read_text())


def query(text, model_name="all-MiniLM-L6-v2", k=5):
    model = SentenceTransformer(model_name)
    qv = model.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)

    index = load_index()
    D, I = index.search(qv, k)

    meta = load_meta()
    results = []
    for score, idx in zip(D[0], I[0]):
        info = meta[idx]
        info = dict(info)
        info["score"] = float(score)
        results.append(info)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    res = query(args.query, model_name=args.model, k=args.k)
    print(json.dumps(res, indent=2))
