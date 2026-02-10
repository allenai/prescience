"""LACER score vs reference category diversity (entropy) analysis for followup prediction (multi-model)."""
import os
import argparse
from collections import Counter
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import utils


MODELS = [
    ("gpt-5.2-2025-12-11", "GPT-5.2", "#0072B2"),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5", "#E69F00"),
    ("gpt-4o-2024-11-20", "GPT-4o", "#9467bd"),
]

BIN_EDGES = [0, 0.5, 1.0, 1.5, 2.0, float("inf")]
BIN_LABELS = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2+"]


def compute_category_entropy(categories):
    """Compute Shannon entropy of category distribution."""
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_bucket_stats(data, corpus_dict):
    """Compute mean LACER and stderr per entropy bucket, plus correlation."""
    num_bins = len(BIN_LABELS)
    bucket_scores = [[] for _ in range(num_bins)]
    all_entropies = []
    all_lacer_scores = []

    for record in data:
        if "lacer_score" not in record or not record.get("key_references"):
            continue
        ref_categories = []
        for ref in record["key_references"]:
            ref_paper = corpus_dict.get(ref["corpus_id"])
            if ref_paper and ref_paper.get("categories"):
                ref_categories.append(ref_paper["categories"][0])
        if ref_categories:
            entropy = compute_category_entropy(ref_categories)
            all_entropies.append(entropy)
            all_lacer_scores.append(record["lacer_score"])
            bucket_idx = np.searchsorted(BIN_EDGES[1:], entropy, side="right")
            bucket_idx = min(bucket_idx, num_bins - 1)
            bucket_scores[bucket_idx].append(record["lacer_score"])

    # Compute correlation
    if len(all_entropies) > 1:
        corr, p_value = stats.pearsonr(all_entropies, all_lacer_scores)
    else:
        corr, p_value = np.nan, np.nan

    means = []
    stderrs = []
    counts = []
    for bucket_idx in range(num_bins):
        values = bucket_scores[bucket_idx]
        counts.append(len(values))
        if len(values) > 0:
            means.append(np.mean(values))
            stderrs.append(np.std(values) / np.sqrt(len(values)))
        else:
            means.append(np.nan)
            stderrs.append(np.nan)
    return means, stderrs, counts, corr, p_value


def main():
    parser = argparse.ArgumentParser(description="Analyze LACER score vs reference category diversity (multi-model)")
    parser.add_argument("--scored_dir", type=str, default="data/task_followup_prediction/test/lacer_scored", help="Directory with LACER-scored files")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    corpus, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    corpus_dict = {p["corpus_id"]: p for p in corpus}
    utils.log(f"Loaded {len(corpus)} papers")

    # Load data for each model
    model_data = {}
    for model_id, model_label, _ in MODELS:
        path = os.path.join(args.scored_dir, f"generations.{model_id}.lacer_scored.json")
        if os.path.exists(path):
            utils.log(f"Loading {model_label} from {path}")
            data_file, _ = utils.load_json(path)
            data = data_file["data"] if "data" in data_file else data_file
            model_data[model_id] = data
            utils.log(f"  Loaded {len(data)} instances")
        else:
            utils.log(f"  WARNING: {path} not found, skipping {model_label}")

    # Compute stats for each model
    model_stats = {}
    for model_id, model_label, _ in MODELS:
        if model_id in model_data:
            means, stderrs, counts, corr, p_value = compute_bucket_stats(model_data[model_id], corpus_dict)
            model_stats[model_id] = {"means": means, "stderrs": stderrs, "counts": counts, "corr": corr, "p_value": p_value}
            utils.log(f"{model_label} bucket stats (ρ={corr:.2f}, p={p_value:.2e}):")
            for i, label in enumerate(BIN_LABELS):
                utils.log(f"  {label}: {counts[i]} instances, mean LACER = {means[i]:.2f}")

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(len(BIN_LABELS))
    for model_id, model_label, color in MODELS:
        if model_id in model_stats:
            model_stat = model_stats[model_id]
            p_str = f"p={model_stat['p_value']:.0e}" if model_stat['p_value'] < 0.01 else f"p={model_stat['p_value']:.2f}"
            legend_label = f"{model_label} (ρ={model_stat['corr']:.2f}, {p_str})"
            plt.errorbar(x, model_stat["means"], yerr=model_stat["stderrs"], label=legend_label,
                         color=color, linewidth=1.5, alpha=0.9, marker="o", markersize=4, capsize=2)

    plt.xlabel("Reference Category Entropy (bits)", fontsize=8)
    plt.ylabel("Mean LACER Score", fontsize=8)
    plt.title("LACER vs Reference Diversity", fontsize=9, fontweight="bold")
    plt.xticks(x, BIN_LABELS, fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, 10)
    plt.legend(loc="upper left", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "lacer_vs_reference_diversity.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
