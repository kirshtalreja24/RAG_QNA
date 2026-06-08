import json
import pickle
from pathlib import Path
import re
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
META_PATH = PROC / "kb_meta.json"
BM25_PATH = PROC / "bm25.pkl"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def bm25_search(query, k=5):
    meta = json.loads(META_PATH.read_text())
    bm25 = load_bm25()

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for i in top_k_idx:
        item = dict(meta[i])
        item["score"] = float(scores[i])
        results.append(item)

    return results

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results = bm25_search(args.query, k=args.k)
    print(json.dumps(results, indent=2))