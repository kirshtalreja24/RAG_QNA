"""
prompt.py
=========
Final stable RAG generation module
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────
#facebook/BART-large-cnn -> main model

MODEL_NAME = "google/flan-t5-base"
FALLBACK_MODEL = "google/flan-t5-small"
MAX_NEW_TOKENS = 64
NUM_BEAMS = 4

PROMPT_TEMPLATE = """\
You are a precise question-answering assistant for a retrieval-based system.

RULES:
- Use ONLY the information provided in the Context.
- Do NOT use external knowledge or assumptions.
- If the Context does not contain the answer, reply exactly: "I don't know."
- If the Context contains partial information, use it to form the best possible answer.
- Prefer extracting exact facts from the Context.
- Do not repeat the question in the answer.
- Be concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""


# ─────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────
def build_prompt(
    question: str,
    retrieved_docs: List[Dict],
    max_context_docs: int = 5,
) -> str:

    if not retrieved_docs:
        context = "[No relevant context retrieved]"
    else:
        chunks = []
        for i, doc in enumerate(retrieved_docs[:max_context_docs], 1):
            title = doc.get("title", "Unknown")
            text = doc.get("text", "")

            chunks.append(f"[{i}] {title}\n{text}")

        context = "\n\n".join(chunks)

    return PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
    )


# ─────────────────────────────────────────────────────────────
# Model loader (SAFE + CACHED)
# ─────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_id = None


def _load_model(model_name: str = MODEL_NAME):
    global _model, _tokenizer, _model_id

    if _model is not None and _model_id == model_name:
        return _tokenizer, _model

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for m in [model_name, FALLBACK_MODEL]:
        try:
            logger.info(f"Loading model: {m}")

            tok = AutoTokenizer.from_pretrained(m)
            mod = AutoModelForSeq2SeqLM.from_pretrained(m)

            mod = mod.to(device)
            mod.eval()

            _tokenizer = tok
            _model = mod
            _model_id = m

            return tok, mod

        except Exception as e:
            logger.warning(f"Model load failed ({m}): {e}")

    raise RuntimeError("Both primary and fallback models failed.")


# ─────────────────────────────────────────────────────────────
# Generation core
# ─────────────────────────────────────────────────────────────
def _generate(prompt: str, model_name: str) -> str:
    import torch

    tokenizer, model = _load_model(model_name)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=3,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ── safe output parsing ───────────────────────────────
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def _postprocess(text: str) -> str:
    for p in ("Answer:", "answer:", "A:"):
        if text.startswith(p):
            return text[len(p):].strip()
    return text


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def generate_answer(
    question: str,
    retrieved_docs: List[Dict],
    max_context_docs: int = 5,
    model_name: str = MODEL_NAME,
    return_prompt: bool = False,
) -> Union[str, Dict]:

    prompt = build_prompt(question, retrieved_docs, max_context_docs)

    try:
        raw = _generate(prompt, model_name)
        answer = _postprocess(raw)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        answer = "I don't know."

    if not return_prompt:
        return answer

    return {
        "answer": answer,
        "prompt": prompt,
        "sources": [
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "rerank_score": doc.get("rerank_score"),
                "snippet": (doc.get("text", "")[:150] if doc.get("text") else ""),
            }
            for doc in retrieved_docs[:max_context_docs]
        ],
    }


# ─────────────────────────────────────────────────────────────
# Full pipeline wrapper (optional)
# ─────────────────────────────────────────────────────────────
def ask(
    question: str,
    final_k: int = 5,
    candidate_k: int = 50,
    model_name: str = MODEL_NAME,
    verbose: bool = False,
) -> Dict:

    from src.retrieval.rag_pipeline import search

    retrieved_docs = search(
        question,
        final_k=final_k,
        candidate_k=candidate_k,
    )

    result = generate_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        max_context_docs=final_k,
        model_name=model_name,
        return_prompt=True,
    )

    output = {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "prompt": result["prompt"],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        for i, s in enumerate(output["sources"], 1):
            score = s.get("rerank_score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
            print(f"[{i}] {s.get('title')} | score={score_str}")
            print(s.get("snippet", ""))

        print("=" * 60)
        print("ANSWER:", output["answer"])
        print("=" * 60)

    return output


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.question:
        print("Please provide a question.")
        sys.exit(0)

    result = ask(
        question=args.question,
        final_k=args.k,
        candidate_k=args.candidate_k,
        model_name=args.model,
        verbose=args.verbose,
    )

    print("\nAnswer:", result["answer"])