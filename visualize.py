"""
visualize.py
------------
Generates comparison charts from results/summary.json (produced by
run_experiment.py). Saves PNGs into results/ for use in the README
and LinkedIn post.

Usage:
    python visualize.py
"""

import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def load_summary():
    path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "results/summary.json not found. Run `python run_experiment.py` first."
        )
    with open(path) as f:
        return json.load(f)


def plot_accuracy_comparison(summary):
    strategies = list(summary.keys())
    accuracies = [summary[s]["accuracy"] * 100 for s in strategies]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(strategies, accuracies, color="#4C72B0")
    plt.ylabel("Retrieval Accuracy (%)")
    plt.title("Retrieval Accuracy by Chunking Strategy")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{acc:.1f}%", ha="center")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_failure_breakdown(summary):
    strategies = list(summary.keys())
    all_labels = sorted({
        label for s in strategies for label in summary[s]["label_counts"]
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0] * len(strategies)
    colors = {
        "SUCCESS": "#55A868",
        "LOW_CONFIDENCE": "#DD8452",
        "FRAGMENTED_CONTEXT": "#C44E52",
        "WRONG_DOCUMENT": "#8172B2",
        "NOT_RETRIEVED": "#4C4C4C",
    }

    for label in all_labels:
        counts = [summary[s]["label_counts"].get(label, 0) for s in strategies]
        ax.bar(strategies, counts, bottom=bottom, label=label,
               color=colors.get(label, "#333333"))
        bottom = [b + c for b, c in zip(bottom, counts)]

    ax.set_ylabel("Number of Queries")
    ax.set_title("Failure Mode Breakdown by Chunking Strategy")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "failure_breakdown.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    summary = load_summary()
    plot_accuracy_comparison(summary)
    plot_failure_breakdown(summary)


if __name__ == "__main__":
    main()
