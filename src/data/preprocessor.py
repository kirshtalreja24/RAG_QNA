import re
import unicodedata

CHUNK_SIZE = 100
MIN_WORDS = 20


# ─────────────────────────────────────────────
# Precompiled regex (faster + cleaner)
# ─────────────────────────────────────────────

RE_HTML = re.compile(r"<[^>]+>")
RE_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
RE_TEMPLATES = re.compile(r"\{\{.*?\}\}")
RE_HEADINGS = re.compile(r"={2,}.*?={2,}")
RE_CITATION_NUM = re.compile(r"\[\d+\]")
RE_CITATION_TEXT = re.compile(r"\[citation needed\]", re.I)
RE_WHITESPACE = re.compile(r"\s+")


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────

def clean(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = RE_HTML.sub(" ", text)
    text = RE_WIKI_LINK.sub(r"\1", text)
    text = RE_TEMPLATES.sub(" ", text)
    text = RE_HEADINGS.sub(" ", text)
    text = RE_CITATION_NUM.sub(" ", text)
    text = RE_CITATION_TEXT.sub(" ", text)

    return RE_WHITESPACE.sub(" ", text).strip()


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk(text: str):
    words = text.split()
    if len(words) < MIN_WORDS:
        return []

    return [
        " ".join(words[i:i + CHUNK_SIZE])
        for i in range(0, len(words), CHUNK_SIZE)
        if len(words[i:i + CHUNK_SIZE]) == CHUNK_SIZE
        or i + CHUNK_SIZE >= len(words)
    ]


# ─────────────────────────────────────────────
# Question / answer utilities
# ─────────────────────────────────────────────

def is_valid_question(question: str) -> bool:
    return bool(question and question.strip() and len(question.strip()) > 5)


def process_answers(answers):
    seen = set()
    clean_answers = []
    for answer in answers:
        if not answer:
            continue
        normalized = answer.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            clean_answers.append(normalized)
    return clean_answers


# ─────────────────────────────────────────────
# Document processor (RAG-ready)
# ─────────────────────────────────────────────

def process_document(doc_id, title, text):
    text = clean(text)
    if not text:
        return []

    title = clean(title)
    chunks = chunk(text)

    return [
        {
            "id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": i,
            "text": f"{title} {chunk_text}",
            "content": chunk_text
        }
        for i, chunk_text in enumerate(chunks)
    ]