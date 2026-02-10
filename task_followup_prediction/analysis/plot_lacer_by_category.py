"""LACER score by primary arXiv category analysis for followup prediction."""
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import utils


def main():
    parser = argparse.ArgumentParser(description="Analyze LACER score by primary arXiv category")
    parser.add_argument("--scored_path", type=str, default="data/task_followup_prediction/test/lacer_scored/generations.gpt-5-2025-08-07.lacer_scored.json", help="Path to LACER-scored file")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/analysis", help="Output directory")
    parser.add_argument("--min_samples", type=int, default=50, help="Minimum samples per category to include")
    args = parser.parse_args()

    utils.log(f"Loading scored data from {args.scored_path}")
    scored_data, _ = utils.load_json(args.scored_path)
    utils.log(f"Loaded {len(scored_data)} instances")

    # Group by primary category
    category_scores = defaultdict(list)
    for record in scored_data:
        if "lacer_score" not in record or not record.get("categories"):
            continue
        primary_cat = record["categories"][0]
        category_scores[primary_cat].append(record["lacer_score"])

    # Filter categories with enough samples and compute stats
    categories = []
    means = []
    stderrs = []
    counts = []
    for cat, scores in sorted(category_scores.items()):
        if len(scores) >= args.min_samples:
            categories.append(cat)
            means.append(np.mean(scores))
            stderrs.append(np.std(scores) / np.sqrt(len(scores)))
            counts.append(len(scores))

    utils.log(f"Found {len(categories)} categories with >= {args.min_samples} samples")
    for i, cat in enumerate(categories):
        utils.log(f"  {cat}: n={counts[i]}, mean LACER={means[i]:.2f} ± {stderrs[i]:.2f}")

    # Sort by mean LACER score
    sorted_indices = np.argsort(means)
    categories = [categories[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stderrs = [stderrs[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(4.5, 3.0))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(len(categories))
    bars = plt.bar(x, means, yerr=stderrs, capsize=2, color="#0072B2", alpha=0.8, edgecolor="black", linewidth=0.5)

    plt.xlabel("Primary arXiv Category", fontsize=8)
    plt.ylabel("Mean LACER Score", fontsize=8)
    plt.title("LACER Score by Primary Category", fontsize=9, fontweight="bold")
    plt.xticks(x, categories, fontsize=6, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.ylim(0, 10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "lacer_by_category.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
