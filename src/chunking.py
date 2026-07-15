"""
chunking.py
------------
Implements multiple chunking strategies used to split documents before
embedding + indexing. Each strategy returns a list of chunk dicts:
    {"chunk_id": str, "doc_id": str, "title": str, "text": str}

Strategies included:
  1. fixed_size      - naive fixed character-length windows (no overlap awareness)
  2. fixed_overlap    - fixed character windows WITH overlap
  3. sentence_based   - splits on sentence boundaries, groups N sentences per chunk
  4. whole_document   - baseline: no chunking at all (1 chunk = 1 doc)
"""

import re


def _split_sentences(text):
    # Lightweight sentence splitter (avoids heavy NLP deps to stay laptop-friendly)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_fixed_size(doc, chunk_size=200):
    """Naive fixed-size chunking with NO overlap. Common beginner mistake:
    can split a sentence (and its meaning) directly in half."""
    text = doc["text"]
    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i:i + chunk_size].strip()
        if piece:
            chunks.append({
                "chunk_id": f"{doc['id']}_fixed_{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": piece,
                "strategy": "fixed_size"
            })
    return chunks


def chunk_fixed_overlap(doc, chunk_size=200, overlap=50):
    """Fixed-size chunking WITH overlap, reduces boundary information loss."""
    text = doc["text"]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size].strip()
        if piece:
            chunks.append({
                "chunk_id": f"{doc['id']}_overlap_{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": piece,
                "strategy": "fixed_overlap"
            })
        if i + chunk_size >= len(text):
            break
    return chunks


def chunk_sentence_based(doc, sentences_per_chunk=3):
    """Groups whole sentences together, respecting sentence boundaries."""
    sentences = _split_sentences(doc["text"])
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i:i + sentences_per_chunk]
        piece = " ".join(group)
        chunks.append({
            "chunk_id": f"{doc['id']}_sent_{i}",
            "doc_id": doc["id"],
            "title": doc["title"],
            "text": piece,
            "strategy": "sentence_based"
        })
    return chunks


def chunk_whole_document(doc):
    """Baseline: treats the entire document as a single chunk."""
    return [{
        "chunk_id": f"{doc['id']}_whole",
        "doc_id": doc["id"],
        "title": doc["title"],
        "text": doc["text"],
        "strategy": "whole_document"
    }]


STRATEGIES = {
    "fixed_size": lambda doc: chunk_fixed_size(doc, chunk_size=120),
    "fixed_overlap": lambda doc: chunk_fixed_overlap(doc, chunk_size=120, overlap=30),
    "sentence_based": lambda doc: chunk_sentence_based(doc, sentences_per_chunk=2),
    "whole_document": lambda doc: chunk_whole_document(doc),
}


def chunk_corpus(corpus, strategy_name):
    """Applies a named strategy to every document in the corpus."""
    fn = STRATEGIES[strategy_name]
    all_chunks = []
    for doc in corpus:
        all_chunks.extend(fn(doc))
    return all_chunks
