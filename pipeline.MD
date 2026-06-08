1. DATA INGESTION
   └── Load NQ questions + Wikipedia corpus subset

2. DATA PREPROCESSING
   └── Clean text (strip HTML, normalize)
   └── Split into 100-word disjoint chunks
   └── Attach metadata {id, title, chunk_index}

3. KNOWLEDGE BASE CREATION
   └── Store chunks as list of dicts (JSON/CSV)

4. EMBEDDING GENERATION (x2 indexes)
   └── Dense: DPR context encoder → 768-dim vectors
   └── BM25: tokenize + build inverted index (rank_bm25)

5. VECTOR DATABASE SETUP
   └── FAISS IndexFlatIP (inner product) for dense
   └── BM25Okapi index for lexical

6. RETRIEVAL
   └── Dense: encode query → MIPS top-K via FAISS
   └── BM25: tokenize query → BM25 score top-K

7. RE-RANKING
   └── Not in paper or proposal → SKIP

8. PROMPT CONSTRUCTION
   └── Concatenate [question + retrieved_passage] per document
   └── Pass K concatenated inputs to generator

9. LLM RESPONSE GENERATION
   └── RAG-Sequence: generate per doc, marginalize
   └── RAG-Token: autoregressive with per-token marginalization
   └── Fallback: simple concat + local LLM (LangChain/Haystack)

10. EVALUATION
    └── Exact Match (EM) on NQ test set
    └── Compare BM25 vs Dense retrieval EM scores
    └── Log retrieval time, answer quality