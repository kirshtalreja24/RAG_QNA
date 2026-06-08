import json
import random
from pathlib import Path

random.seed(42)

kb_path = Path("data/processed/knowledge_base.json")

with open(kb_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

subset = random.sample(chunks, 50000)

out_path = Path("data/processed/knowledge_base_50k.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(subset, f)

print(f"Saved {len(subset)} chunks to {out_path}")