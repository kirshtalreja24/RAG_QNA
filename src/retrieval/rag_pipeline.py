"""
hybrid_retriever.py

Pipeline:
    Query
      ↓
    FAISS Top-N
      ↓
    BM25 Top-N
      ↓
    RRF Fusion
      ↓
    CrossEncoder Reranking
      ↓
    Final Top-K chunks

Usage:
    python hybrid_retriever.py "what is life" --k 5
"""

import sys
from pathlib import Path
from collections import defaultdict

from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Make src importable
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.embeddings.fiass_query_index import query as faiss_query
from data.embeddings.bm25_query import bm25_search


# --------------------------------------------------
# Lazy-load reranker
# --------------------------------------------------

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        print("Loading CrossEncoder reranker...")
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker


# --------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# --------------------------------------------------

def rrf_merge(
    faiss_results,
    bm25_results,
    rrf_k=60,
    faiss_weight=0.7,
    bm25_weight=0.3
):
    scores = defaultdict(float)
    doc_map = {}

    for rank, doc in enumerate(faiss_results):
        doc_id = doc["id"]

        scores[doc_id] += (
            faiss_weight *
            (1.0 / (rrf_k + rank + 1))
        )

        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]

        scores[doc_id] += (
            bm25_weight *
            (1.0 / (rrf_k + rank + 1))
        )

        doc_map[doc_id] = doc

    ranked_ids = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    merged = []
    seen = set()

    for doc_id, _ in ranked_ids:
        if doc_id not in seen:
            merged.append(doc_map[doc_id])
            seen.add(doc_id)

    return merged


# --------------------------------------------------
# CrossEncoder reranking
# --------------------------------------------------

def rerank(query, docs, top_k=5):
    if not docs:
        return []

    reranker = get_reranker()

    pairs = [
        (query, doc["text"])
        for doc in docs
    ]

    scores = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    reranked = []

    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]


# --------------------------------------------------
# Main search API
# --------------------------------------------------

def search(
    query,
    final_k=5,
    candidate_k=50
):
    # Dense retrieval
    faiss_results = faiss_query(
        query,
        k=candidate_k
    )

    # Sparse retrieval
    bm25_results = bm25_search(
        query,
        k=candidate_k
    )

    # Hybrid fusion
    merged = rrf_merge(
        faiss_results,
        bm25_results
    )

    # Reranking
    final_results = rerank(
        query,
        merged,
        top_k=final_k
    )

    return final_results


# --------------------------------------------------
# Context builder for LLM
# --------------------------------------------------

def build_context(results):
    return "\n\n".join(
        doc["text"]
        for doc in results
    )


# --------------------------------------------------
# CLI test
# --------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        final_k=args.k,
        candidate_k=args.candidate_k
    )

    print(json.dumps(results, indent=2))