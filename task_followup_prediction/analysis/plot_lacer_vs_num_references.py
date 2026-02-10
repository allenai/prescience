"""LACER score vs number of background papers analysis for followup prediction (multi-model)."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


MODELS = [
    ("gpt-5.2-2025-12-11", "GPT-5.2", "#0072B2"),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5", "#E69F00"),
    ("gpt-4o-2024-11-20", "GPT-4o", "#9467bd"),
]

BUCKET_LABELS = ["1", "2", "3", "4", "5", "6-10"]


def compute_bucket_stats(data):
    """Compute mean LACER and stderr per bucket for a model's data."""
    num_buckets = len(BUCKET_LABELS)
    results = [[] for _ in range(num_buckets)]
    for instance in data:
        num_refs = len(instance["key_references"])
        lacer_score = instance["lacer_score"]
        if num_refs <= 5:
            bucket_idx = num_refs - 1
        else:
            bucket_idx = 5
        results[bucket_idx].append(lacer_score)

    means = []
    stderrs = []
    counts = []
    for bucket_idx in range(num_buckets):
        values = results[bucket_idx]
        counts.append(len(values))
        if len(values) > 0:
            means.append(np.mean(values))
            stderrs.append(np.std(values) / np.sqrt(len(values)))
        else:
            means.append(np.nan)
            stderrs.append(np.nan)
    return means, stderrs, counts


def main():
    parser = argparse.ArgumentParser(description="Analyze LACER score vs number of background papers (multi-model)")
    parser.add_argument("--scored_dir", type=str, default="data/task_followup_prediction/test/lacer_scored_opus", help="Directory with LACER-scored files")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

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
            means, stderrs, counts = compute_bucket_stats(model_data[model_id])
            model_stats[model_id] = {"means": means, "stderrs": stderrs, "counts": counts}
            utils.log(f"{model_label} bucket stats:")
            for i, label in enumerate(BUCKET_LABELS):
                utils.log(f"  {label}: {counts[i]} instances, mean LACER = {means[i]:.2f}")

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(len(BUCKET_LABELS))
    for model_id, model_label, color in MODELS:
        if model_id in model_stats:
            stats = model_stats[model_id]
            plt.errorbar(x, stats["means"], yerr=stats["stderrs"], label=model_label,
                         color=color, linewidth=1.5, alpha=0.9, marker="o", markersize=4, capsize=2)

    plt.xlabel("# Background Papers", fontsize=8)
    plt.ylabel("Mean LACER Score", fontsize=8)
    plt.title("LACER Score vs # Background Papers", fontsize=9, fontweight="bold")
    plt.xticks(x, BUCKET_LABELS, fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, 10)
    plt.legend(loc="lower right", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "lacer_vs_num_references.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
