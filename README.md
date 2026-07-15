# RAG Failure Diagnostic Toolkit

**Most RAG tutorials show you how to build a retrieval pipeline. This project answers a harder question: when retrieval gives you the wrong answer, *why*?**

A lightweight, fully local toolkit that runs a labeled query set against multiple chunking strategies, then automatically classifies *why* each retrieval succeeded or failed — wrong document, low confidence, fragmented context, or missed entirely.

---

## The Problem

RAG pipelines fail silently. A chatbot confidently answers with the wrong context, and there's rarely an easy way to tell whether the issue is:
- The **chunking strategy** splitting an answer across two chunks
- The **embedding model** missing the semantic connection between query and document
- The **retrieval threshold** being too strict or too loose

Most teams debug this by eyeballing a handful of examples. This toolkit turns that into a repeatable, measurable experiment.

## What It Does

1. Takes a small corpus of technical documents (12 docs on Docker, PostgreSQL, FAISS, MLflow, Kubernetes, etc.)
2. Chunks it 4 different ways: `fixed_size`, `fixed_overlap`, `sentence_based`, `whole_document`
3. Builds a FAISS index for each chunking strategy using Sentence Transformers embeddings
4. Runs 12 hand-labeled queries (each with a known correct source document) against every index
5. **Classifies each result** into one of 5 failure categories using rule-based diagnostics:

| Label | Meaning |
|---|---|
| `SUCCESS` | Correct document retrieved with high confidence |
| `WRONG_DOCUMENT` | Top result is confidently retrieved but from the wrong document — semantic mismatch |
| `FRAGMENTED_CONTEXT` | Correct document retrieved, but the chunk is too small/cut-off — chunk boundary likely split the answer |
| `LOW_CONFIDENCE` | Correct document retrieved, but the similarity score barely clears the threshold — fragile match |
| `NOT_RETRIEVED` | Correct document missing from top-k entirely |

6. Generates comparison charts across strategies

## Why This Is Different From a Typical RAG Project

Most portfolio RAG projects stop at "I built a chatbot that answers questions from documents." This project instead treats retrieval quality as an **empirical, measurable engineering problem** — the same way you'd benchmark any other system component. It's the difference between using a tool and understanding its failure modes.

## Quick Start (Zero Setup Demo)

Want to see the pipeline run in 10 seconds with no model download?

```bash
pip install -r requirements.txt
python demo_offline.py    # uses TF-IDF instead of Sentence Transformers — fully offline
python visualize.py
```

This validates the full pipeline (chunking → indexing → search → classification) without needing to download anything. Note: TF-IDF is a weak semantic representation, so accuracy numbers here are for pipeline demonstration only, not representative of real performance.

## Full Experiment (Real Embeddings)

```bash
pip install -r requirements.txt
python run_experiment.py   # downloads all-MiniLM-L6-v2 (~80MB) on first run
python visualize.py
```

Results are saved to `results/`:
- `summary.json` — accuracy + failure breakdown per strategy
- `detailed_results.json` — per-query classification with matched chunk text
- `accuracy_comparison.png` — bar chart comparing strategies
- `failure_breakdown.png` — stacked chart of failure types per strategy

## Optional: LLM-Powered Failure Explanations

Set a `GROQ_API_KEY` environment variable to get natural-language explanations of failures using Llama-3.3-70B via Groq:

```bash
export GROQ_API_KEY=your_key_here
```

The toolkit works fully without this — it's an optional enrichment layer on top of the rule-based classification.

## Project Structure

```
rag-diagnostic-toolkit/
├── data/
│   ├── corpus.json          # 12 technical documents
│   └── eval_queries.json    # 12 labeled queries with expected source doc
├── src/
│   ├── chunking.py          # 4 chunking strategies
│   ├── embed_index.py       # Sentence Transformers + FAISS
│   └── diagnose.py          # Rule-based failure classification + optional LLM explainer
├── run_experiment.py        # full pipeline (real embeddings)
├── demo_offline.py          # zero-download demo (TF-IDF)
├── visualize.py             # generates comparison charts
└── results/                 # generated output (charts, JSON)
```

## Design Decisions Worth Noting

- **Flat FAISS index (`IndexFlatIP`)**, not IVF/ANN: the corpus is small enough that exact search is fast, and this removes approximate-search error as a confounding variable when diagnosing chunking/embedding issues specifically.
- **`all-MiniLM-L6-v2`** was chosen over larger embedding models specifically because it runs well on CPU-only, low-RAM machines — this toolkit is meant to be runnable, not just theoretically correct.
- **Rule-based classification first, LLM second**: the core diagnostic labels are deterministic and reproducible without any API key. The LLM layer only adds a natural-language explanation on top — it never gates core functionality.

## Possible Extensions

- Add more chunking strategies (recursive character splitting, semantic chunking via embedding similarity)
- Swap in a larger embedding model and compare
- Extend the query set to include multi-hop questions
- Add a Streamlit UI for interactive exploration

## Author

Built by Saqib — exploring the internals of retrieval systems beyond the "build a RAG chatbot" tutorial level.

