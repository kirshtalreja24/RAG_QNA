import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from loader import load_nq_dataset, load_wikipedia_subset
from preprocessor import process_document

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PROC_DIR = Path("data/processed")
KB_JSON = PROC_DIR / "knowledge_base.json"
KB_CSV  = PROC_DIR / "knowledge_base.csv"
CKPT    = PROC_DIR / ".ckpt.json"


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────

def load_ckpt():
    if CKPT.exists():
        ck = json.load(open(CKPT))
        print(f"[ckpt] Resume: i={ck['i']} chunks={ck['n']}")
        return ck
    return {"i": -1, "n": 0}


def save_ckpt(i, n):
    CKPT.write_text(json.dumps({"i": i, "n": n}))
    print(f"[ckpt] Saved: i={i} chunks={n}")


def clear_ckpt():
    if CKPT.exists():
        CKPT.unlink()
        print("[ckpt] Cleared")


# ─────────────────────────────────────────────
# NQ build (matches simplified loader)
# ─────────────────────────────────────────────

def build_nq(raw_dir, force=False, max_nq=None):
    if KB_JSON.exists() and not force:
        print("[NQ] Already built — skipping")
        return

    print("[NQ] Loading dataset...")
    train, dev, test = load_nq_dataset(
        raw_dir=raw_dir,
        max_examples=max_nq,
        force=force
    )

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    def dump(name, data):
        path = PROC_DIR / name
        path.write_text(json.dumps(data, indent=2))
        return path

    dump("nq_train.json", train)
    dump("nq_dev.json", dev)
    dump("nq_test.json", test)

    print(f"[NQ] Done → train={len(train)} dev={len(dev)} test={len(test)}")


# ─────────────────────────────────────────────
# Knowledge Base builder
# ─────────────────────────────────────────────

def build_kb(limit, batch_size, force=False):

    if KB_JSON.exists() and not force:
        print("[KB] Already exists — skipping")
        return

    print("[KB] Loading Wikipedia subset...")
    docs = load_wikipedia_subset(limit=limit, force_rebuild=force)

    ckpt = load_ckpt()
    start, total_chunks = ckpt["i"] + 1, ckpt["n"]

    print(f"[KB] Resume at doc={start}, chunks={total_chunks}")

    chunks = []
    if start > 0 and KB_JSON.exists():
        chunks = json.loads(KB_JSON.read_text())

    for i in range(start, len(docs)):

        if i % 1000 == 0 and i != start:
            print(f"[KB] Processing {i}/{len(docs)} | chunks={total_chunks}")

        new_chunks = process_document(
            str(i),
            docs[i]["title"],
            docs[i]["text"]
        )

        chunks.extend(new_chunks)
        total_chunks += len(new_chunks)

        if i % batch_size == 0:
            save_ckpt(i, total_chunks)

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    KB_JSON.write_text(json.dumps(chunks, indent=2))
    pd.DataFrame(chunks).to_csv(KB_CSV, index=False)

    clear_ckpt()

    print(f"[KB] Done → total chunks = {len(chunks)}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--wiki-limit", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--max-nq", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--nq-only", action="store_true")
    p.add_argument("--kb-only", action="store_true")

    args = p.parse_args()

    if not args.kb_only:
        build_nq(args.raw_dir, args.force, args.max_nq)

    if not args.nq_only:
        build_kb(args.wiki_limit, args.batch_size, args.force)


if __name__ == "__main__":
    main()