"""
demo_offline.py
----------------
A zero-download, zero-internet demo of the diagnostic pipeline.

Instead of Sentence Transformers (which needs to download a model from
Hugging Face on first run), this uses a TF-IDF vectorizer (scikit-learn,
fully offline) as a drop-in embedding backend. This lets anyone clone the
repo and see the full chunking -> indexing -> search -> failure-classification
pipeline run in seconds, with zero setup friction.

NOTE: TF-IDF is a much weaker semantic representation than a real sentence
embedding model, so accuracy numbers here are NOT representative of the
toolkit's real-world performance -- this script exists purely to demonstrate
the pipeline mechanics instantly. For the full experiment with real
embeddings, run `python run_experiment.py` (requires ~80MB one-time model
download via sentence-transformers).
"""
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from src.chunking import chunk_corpus, STRATEGIES
from src.diagnose import run_diagnosis, summarize

with open("data/corpus.json") as f:
    corpus = json.load(f)
with open("data/eval_queries.json") as f:
    eval_queries = json.load(f)

# Fit ONE global vectorizer across all strategies' chunks + queries so vector
# spaces are comparable (mimics a shared pretrained embedding space)
all_texts_for_vocab = []
per_strategy_chunks = {}
for strategy_name in STRATEGIES:
    chunks = chunk_corpus(corpus, strategy_name)
    per_strategy_chunks[strategy_name] = chunks
    all_texts_for_vocab.extend([c["text"] for c in chunks])
all_texts_for_vocab.extend([q["query"] for q in eval_queries])

vectorizer = TfidfVectorizer().fit(all_texts_for_vocab)


def embed(texts):
    vecs = vectorizer.transform(texts).toarray().astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_index(chunks):
    embeddings = embed([c["text"] for c in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search(index, chunks, query, top_k=3):
    q_emb = embed([query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "score": float(score)})
    return results

# monkeypatch diagnose.search to use our offline TF-IDF search for this test run
import src.diagnose as diagnose_mod
diagnose_mod.search = search

print(f"{'Strategy':<18}{'#Chunks':<10}{'Accuracy':<12}{'Label counts'}")
print("-" * 80)

all_summaries = {}
for strategy_name, chunks in per_strategy_chunks.items():
    index = build_index(chunks)
    results = run_diagnosis(index, chunks, eval_queries, top_k=3)
    summary = summarize(results)
    all_summaries[strategy_name] = summary
    print(f"{strategy_name:<18}{len(chunks):<10}{summary['accuracy']:<12.2%}{summary['label_counts']}")

import os
os.makedirs("results", exist_ok=True)
with open("results/summary.json", "w") as f:
    json.dump(all_summaries, f, indent=2)

print("\nSaved results/summary.json (demo/TF-IDF numbers -- run run_experiment.py for real results)")
print("Run `python visualize.py` next to generate charts.")
