import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import re

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
KB_JSON = PROC / "knowledge_base_50k.json"
BM25_PATH = PROC / "bm25.pkl"


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_bm25():
    if not KB_JSON.exists():
        raise FileNotFoundError(f"Knowledge base not found at {KB_JSON}. Run the script that builds the chunks first.")

    chunks = json.loads(KB_JSON.read_text())

    # Each chunk text becomes a document
    corpus = [tokenize(c["text"]) for c in chunks]

    bm25 = BM25Okapi(corpus)

    PROC.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 index saved → {BM25_PATH}")


if __name__ == "__main__":
    build_bm25()