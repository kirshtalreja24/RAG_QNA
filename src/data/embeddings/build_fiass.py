import argparse
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

import torch


ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
KB_JSON = PROC / "knowledge_base_50k.json"
INDEX_PATH = PROC / "kb_index.faiss"
META_PATH = PROC / "kb_meta.json"


def load_chunks(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {path}")
    return json.loads(path.read_text())


def main(model_name: str, batch_size: int, device: str):

    chunks = load_chunks(KB_JSON)
    texts = [c["text"] for c in chunks]

    print(f"\nEmbedding {len(texts)} chunks using {model_name} on {device}\n")

    # Force safe device handling
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available. Switching to CPU.")
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)

    # Get embedding dimension
    sample_emb = model.encode(
        texts[:2],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    dim = sample_emb.shape[1]

    if faiss is None:
        raise RuntimeError("faiss not installed (pip install faiss-cpu)")

    index = faiss.IndexFlatIP(dim)
    meta = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]

        emb = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        index.add(emb)

        meta.extend([
            {
                "id": c.get("id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "chunk_index": c.get("chunk_index"),
                "text": f"{c['text']}"
            }
            for c in batch_chunks
        ])

    PROC.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved FAISS index → {INDEX_PATH}")
    print(f"Saved metadata   → {META_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)

    # FIXED: safe default
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    main(args.model, args.batch_size, args.device)