"""
app.py
======
RAG QA Controller — CLI + Flask Web API
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# app/app.py  →  parent = app/  →  parent.parent = project root (where src/ lives)
APP_DIR = Path(__file__).resolve().parent          # .../app/
ROOT    = APP_DIR.parent                           # .../project_root/
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

_pipeline_ready = False


def ensure_pipeline():
    global _pipeline_ready
    if _pipeline_ready:
        return

    try:
        import src.retrieval.rag_pipeline  # noqa
        from src.generation.prompt import generate_answer  # noqa
    except ImportError as e:
        print(f"[ERROR] Missing module: {e}")
        sys.exit(1)

    _pipeline_ready = True


def run_query(
    query: str,
    final_k: int = 5,
    candidate_k: int = 50,
    model_name: Optional[str] = None,
    verbose: bool = False,
) -> Dict:

    ensure_pipeline()

    from src.retrieval.rag_pipeline import search
    from src.generation.prompt import generate_answer

    start = time.time()

    retrieved_docs = search(
        query=query,
        final_k=final_k,
        candidate_k=candidate_k,
    )

    result = generate_answer(
        question=query,
        retrieved_docs=retrieved_docs,
        max_context_docs=final_k,
        model_name=model_name,
        return_prompt=True,
    )

    latency = round(time.time() - start, 2)

    output = {
        "question": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_s": latency,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("QUESTION:", query)
        print("=" * 60)

        for i, s in enumerate(output["sources"], 1):
            print(f"[{i}] {s.get('title')} | score={s.get('rerank_score')}")
            print(s.get("snippet", "")[:120])

        print("=" * 60)
        print("ANSWER:", output["answer"])
        print("=" * 60)

    return output


def interactive_loop(final_k: int, candidate_k: int, model_name: Optional[str]):
    print("\n=== RAG QA System ===")
    print("Type 'exit' to quit\n")

    ensure_pipeline()

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        result = run_query(q, final_k, candidate_k, model_name)

        print("\nAnswer:", result["answer"])
        print(f"(sources={len(result['sources'])}, {result['latency_s']}s)\n")



def create_app():
    """Create and configure the Flask application."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("[ERROR] Flask not installed. Run: pip install flask flask-cors")
        sys.exit(1)

    # static/ sits next to app.py inside the app/ folder
    flask_app = Flask(__name__, static_folder=str(APP_DIR / "static"), static_url_path="")
    CORS(flask_app)

   
    @flask_app.route("/")
    def index():
        
        return flask_app.send_static_file("index.html")

    # ── health check ────────────────────────────────────────────────────────
    @flask_app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "pipeline_ready": _pipeline_ready})

    # ── main query endpoint ─────────────────────────────────────────────────
    @flask_app.route("/api/query", methods=["POST"])
    def query():
        data = request.get_json(silent=True) or {}

        q = (data.get("question") or "").strip()
        if not q:
            return jsonify({"error": "question is required"}), 400

        k           = int(data.get("k", 5))
        candidate_k = int(data.get("candidate_k", 50))
        model_name  = data.get("model") or None

        k           = max(1, min(k, 20))
        candidate_k = max(10, min(candidate_k, 500))

        try:
            result = run_query(
                query=q,
                final_k=k,
                candidate_k=candidate_k,
                model_name=model_name,
                verbose=False,
            )
            return jsonify(result)
        except Exception as e:
            logger.exception("query failed")
            return jsonify({"error": str(e)}), 500

    # ── suggestions endpoint ────────────────────────────────────────────────
    @flask_app.route("/api/suggestions")
    def suggestions():
        import random
        pool = [
            "Who invented the telephone?",
            "What is the capital of Australia?",
            "When did World War II end?",
            "What is photosynthesis?",
            "How does the human immune system work?",
            "Who wrote the theory of relativity?",
            "What causes earthquakes?",
            "What is the speed of light?",
            "Who was the first person on the moon?",
            "What is the population of China?",
            "How do vaccines work?",
            "What is machine learning?",
            "Who discovered penicillin?",
            "What is the tallest mountain on Earth?",
            "How long does it take light to reach Earth from the Sun?",
            "What language has the most native speakers?",
            "Who painted the Sistine Chapel?",
            "What is the deepest ocean trench?",
            "How does DNA replication work?",
            "When was the internet invented?",
        ]
        return jsonify({"suggestions": random.sample(pool, 4)})

    return flask_app



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query",       type=str)
    p.add_argument("--k",                 type=int,  default=5)
    p.add_argument("--candidate-k",       type=int,  default=50)
    p.add_argument("--model",             type=str,  default=None)
    p.add_argument("-v", "--verbose",     action="store_true")
    p.add_argument("--json",              action="store_true")
    # web server flags
    p.add_argument("--serve",             action="store_true",
                   help="Start the Flask web server")
    p.add_argument("--host",              type=str,  default="0.0.0.0")
    p.add_argument("--port",              type=int,  default=5000)
    p.add_argument("--debug",             action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── web server mode ──────────────────────────────────────────────────────
    if args.serve:
        print(f"[RAG] Starting web server at http://{args.host}:{args.port}")
        print(f"[RAG] Open http://localhost:{args.port} in your browser")
        ensure_pipeline()
        flask_app = create_app()
        flask_app.run(host=args.host, port=args.port, debug=args.debug)
        return

    # ── single query mode ────────────────────────────────────────────────────
    if args.query:
        result = run_query(
            args.query,
            args.k,
            args.candidate_k,
            args.model,
            args.verbose,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\nAnswer:", result["answer"])

    # ── interactive REPL ─────────────────────────────────────────────────────
    else:
        interactive_loop(args.k, args.candidate_k, args.model)


if __name__ == "__main__":
    main()