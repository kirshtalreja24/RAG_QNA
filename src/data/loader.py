"""
Minimal loader.py — CSV-only NQ loader + Wikipedia subset
"""

import csv
import json
import random
from pathlib import Path

from .preprocessor import is_valid_question, process_answers

DEFAULT_RAW = Path("data/raw")
NQ_CACHE    = Path("data/processed/nq.json")
WIKI_CACHE  = Path("data/raw/wiki.json")


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _split(data):
    n = len(data)
    return (
        data[:int(0.8 * n)],
        data[int(0.8 * n):int(0.9 * n)],
        data[int(0.9 * n):]
    )


def _dedup(data):
    seen, out = set(), []
    for r in data:
        q = r["question"].strip().lower()
        if q not in seen:
            seen.add(q)
            out.append(r)
    return out


def _parse_answers(raw):
    sep = " , " if " , " in raw else "|"
    return process_answers([a.strip() for a in raw.split(sep) if a.strip()])


# ─────────────────────────────────────────────
# CSV Loader (ONLY SOURCE)
# ─────────────────────────────────────────────

def _load_csv(path: Path, limit=None):
    records = []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            q = (row.get("question") or "").strip()
            a = (row.get("short_answers") or "").strip()
            c = (row.get("long_answers") or "").strip()

            if not q or not a or not is_valid_question(q):
                continue

            answers = _parse_answers(a)
            if not answers:
                continue

            rec = {
                "example_id": str(i),
                "question": q,
                "answers": answers
            }

            if c:
                rec["context"] = c

            records.append(rec)

            if limit and len(records) >= limit:
                break

    return records


# ─────────────────────────────────────────────
# Public API — NQ
# ─────────────────────────────────────────────

def load_nq_dataset(raw_dir=str(DEFAULT_RAW), max_examples=None, force=False):

    # cache
    if NQ_CACHE.exists() and not force:
        data = json.loads(NQ_CACHE.read_text())
        return _split(data[:max_examples] if max_examples else data)

    raw_dir = Path(raw_dir)
    csv_file = next(raw_dir.glob("*.csv"), None)

    if not csv_file:
        raise FileNotFoundError("No CSV file found in data/raw")

    data = _load_csv(csv_file, max_examples)

    # cleanup
    data = _dedup(data)
    random.shuffle(data)

    # cache
    NQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    NQ_CACHE.write_text(json.dumps(data, indent=2))

    return _split(data)


# ─────────────────────────────────────────────
# Wikipedia subset (unchanged)
# ─────────────────────────────────────────────

def load_wikipedia_subset(limit=10000, force_rebuild=False):

    if WIKI_CACHE.exists() and not force_rebuild:
        return json.loads(WIKI_CACHE.read_text())

    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split=f"train[:{limit}]"
    )

    data = [{"title": x["title"], "text": x["text"]} for x in ds]

    WIKI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WIKI_CACHE.write_text(json.dumps(data, indent=2))

    return data