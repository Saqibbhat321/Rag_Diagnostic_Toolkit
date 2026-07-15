"""
run_experiment.py
------------------
Entry point: runs the RAG Failure Diagnostic Toolkit end-to-end.

For every chunking strategy defined in src/chunking.py, this script:
  1. Chunks the corpus
  2. Embeds chunks + builds a FAISS index
  3. Runs the labeled evaluation query set against the index
  4. Classifies each result (SUCCESS / WRONG_DOCUMENT / FRAGMENTED_CONTEXT / etc.)
  5. Prints + saves a comparison summary and generates charts

Usage:
    python run_experiment.py
"""

import json
import os
from src.chunking import chunk_corpus, STRATEGIES
from src.embed_index import build_faiss_index
from src.diagnose import run_diagnosis, summarize

DATA_DIR = "data"
RESULTS_DIR = "results"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    corpus = load_json(os.path.join(DATA_DIR, "corpus.json"))
    eval_queries = load_json(os.path.join(DATA_DIR, "eval_queries.json"))

    all_results = {}
    all_summaries = {}

    for strategy_name in STRATEGIES:
        print(f"\n=== Strategy: {strategy_name} ===")
        chunks = chunk_corpus(corpus, strategy_name)
        print(f"  {len(chunks)} chunks created")

        index, _ = build_faiss_index(chunks)
        results = run_diagnosis(index, chunks, eval_queries, top_k=3)
        summary = summarize(results)

        all_results[strategy_name] = results
        all_summaries[strategy_name] = summary

        print(f"  Accuracy: {summary['accuracy']:.2%}")
        print(f"  Label breakdown: {summary['label_counts']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "detailed_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nSaved detailed_results.json and summary.json to results/")
    return all_results, all_summaries


if __name__ == "__main__":
    main()
