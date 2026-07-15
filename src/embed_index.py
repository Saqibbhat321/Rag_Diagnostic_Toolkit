"""
embed_index.py
--------------
Handles embedding chunks with Sentence Transformers and building
FAISS indices. Uses a small CPU-friendly model (all-MiniLM-L6-v2, ~80MB)
so this runs fine on low-spec laptops without a GPU.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model_cache = {}


def get_model():
    """Loads (and caches) the embedding model. Only loaded once per run."""
    if _MODEL_NAME not in _model_cache:
        _model_cache[_MODEL_NAME] = SentenceTransformer(_MODEL_NAME)
    return _model_cache[_MODEL_NAME]


def embed_texts(texts):
    """Encodes a list of texts into normalized embeddings (for cosine sim via dot product)."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings.astype("float32")


def build_faiss_index(chunks):
    """
    Builds a flat FAISS index (exact search) over the given chunks.
    Flat index chosen deliberately: our corpus is small (<1000 chunks),
    so exact search is fast and removes ANN-approximation as a variable
    when diagnosing retrieval failures.
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vectors
    index.add(embeddings)
    return index, embeddings


def search(index, chunks, query, top_k=3):
    """Runs a single query against a FAISS index, returns top_k chunk matches with scores."""
    query_emb = embed_texts([query])
    scores, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk": chunks[idx],
            "score": float(score)
        })
    return results
