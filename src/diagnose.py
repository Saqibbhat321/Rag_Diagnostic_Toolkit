"""
diagnose.py
-----------
Runs the evaluation query set against a FAISS index built with a given
chunking strategy, and classifies WHY each retrieval succeeded or failed.

Failure categories (rule-based, no LLM required to run):
  - SUCCESS            : correct source document retrieved in top_k
  - WRONG_DOCUMENT      : a confidently-retrieved chunk belongs to the wrong doc
                          (embedding/semantic mismatch)
  - FRAGMENTED_CONTEXT  : correct doc WAS retrieved but only a partial/cut-off
                          chunk, suggesting the chunk boundary split the answer
  - LOW_CONFIDENCE      : correct doc retrieved, but similarity score is low,
                          meaning it barely made the cut and could get pushed
                          out by a larger top_k or noisier corpus
  - NOT_RETRIEVED       : correct doc missing from top_k entirely

An optional LLM-based explainer (via Groq) can add a natural-language
explanation on top of the rule-based label -- see explain_with_llm().
"""

import os
from src.embed_index import search

LOW_CONFIDENCE_THRESHOLD = 0.45


def classify_result(query_obj, retrieved):
    """Applies rule-based classification to a single query's retrieval results."""
    expected_doc = query_obj["expected_doc_id"]
    retrieved_doc_ids = [r["chunk"]["doc_id"] for r in retrieved]

    if expected_doc not in retrieved_doc_ids:
        return "NOT_RETRIEVED", None

    # find the best-scoring chunk that belongs to the correct doc
    match = next(r for r in retrieved if r["chunk"]["doc_id"] == expected_doc)
    top_result = retrieved[0]

    if top_result["chunk"]["doc_id"] != expected_doc:
        return "WRONG_DOCUMENT", match

    if match["score"] < LOW_CONFIDENCE_THRESHOLD:
        return "LOW_CONFIDENCE", match

    # crude fragmentation heuristic: chunk is short relative to typical doc length
    # AND doesn't contain key terms from the query -> likely split mid-answer
    if len(match["chunk"]["text"]) < 120:
        return "FRAGMENTED_CONTEXT", match

    return "SUCCESS", match


def run_diagnosis(index, chunks, eval_queries, top_k=3):
    """Runs the full eval query set through the index and returns per-query diagnostics."""
    results = []
    for q in eval_queries:
        retrieved = search(index, chunks, q["query"], top_k=top_k)
        label, match = classify_result(q, retrieved)
        results.append({
            "query": q["query"],
            "expected_doc_id": q["expected_doc_id"],
            "label": label,
            "top_score": retrieved[0]["score"] if retrieved else None,
            "matched_chunk_text": match["chunk"]["text"] if match else None,
            "matched_score": match["score"] if match else None,
            "retrieved_doc_ids": [r["chunk"]["doc_id"] for r in retrieved],
        })
    return results


def summarize(results):
    """Aggregates label counts and computes overall retrieval accuracy."""
    total = len(results)
    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    success = counts.get("SUCCESS", 0)
    accuracy = success / total if total else 0.0
    return {"total": total, "accuracy": accuracy, "label_counts": counts}


def explain_with_llm(query, expected_text, matched_text, label):
    """
    Optional: uses Groq's Llama-3.3-70B to generate a natural-language
    explanation of a failure case. Requires GROQ_API_KEY env var.
    Gracefully returns None if no key is set, so the rest of the toolkit
    works fully offline without this.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    from groq import Groq
    client = Groq(api_key=api_key)

    prompt = f"""You are debugging a RAG retrieval failure.

Query: {query}
Failure type: {label}
Retrieved chunk text: {matched_text}

In 1-2 sentences, explain the most likely reason this chunk was retrieved
(or failed to be retrieved correctly), from an information-retrieval
engineering perspective."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content
