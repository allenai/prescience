"""nDCG vs number of references analysis for prior work prediction baselines."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


BASELINES = [
    "frequency",
    "rank_fusion.grit",
    "embedding_fusion.grit",
    "hierarchical.grit.pca64.power0.5",
    "mean_pooling_projected.grit",
]

BASELINE_LABELS = {
    "frequency": "Frequency",
    "rank_fusion.grit": "Rank Fusion",
    "embedding_fusion.grit": "Embedding Fusion (Refs)",
    "hierarchical.grit.pca64.power0.5": "Hierarchical",
    "mean_pooling_projected.grit": "Embedding Fusion (Papers) Projected",
}

BASELINE_STYLES = {
    "frequency": {"color": "#0072B2", "linestyle": "-"},
    "rank_fusion.grit": {"color": "#9467bd", "linestyle": "-"},
    "embedding_fusion.grit": {"color": "#E69F00", "linestyle": "-"},
    "hierarchical.grit.pca64.power0.5": {"color": "#2CA02C", "linestyle": "-"},
    "mean_pooling_projected.grit": {"color": "#CC79A7", "linestyle": "-"},
}

BUCKET_LABELS = ["1", "2", "3", "4", "5", "6-10", "11+"]


def main():
    parser = argparse.ArgumentParser(description="Analyze nDCG vs number of references")
    parser.add_argument("--predictions_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--scored_dir", type=str, default="data/task_priorwork_prediction/test/scored", help="Scored predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    # Load predictions and scored files
    utils.log("Loading predictions and scores")
    baseline_predictions = {}
    baseline_scores = {}
    for baseline in BASELINES:
        pred_path = os.path.join(args.predictions_dir, f"predictions.{baseline}.json")
        scored_path = os.path.join(args.scored_dir, f"predictions.{baseline}.eval.json")
        if os.path.exists(pred_path) and os.path.exists(scored_path):
            pred_file, _ = utils.load_json(pred_path)
            pred_data = pred_file["data"] if "data" in pred_file else pred_file
            scored_data, _ = utils.load_json(scored_path)
            baseline_predictions[baseline] = {p["corpus_id"]: p for p in pred_data}
            baseline_scores[baseline] = {s["corpus_id"]: s["ndcg"] for s in scored_data["per_instance"]}
            utils.log(f"  {baseline}: {len(baseline_predictions[baseline])} predictions, {len(baseline_scores[baseline])} scores")
        else:
            utils.log(f"  WARNING: Missing files for {baseline}")

    # Find common corpus_ids across all baselines' scores
    common_ids = None
    for baseline in baseline_scores:
        if common_ids is None:
            common_ids = set(baseline_scores[baseline].keys())
        else:
            common_ids = common_ids & set(baseline_scores[baseline].keys())
    utils.log(f"Found {len(common_ids)} common instances across all baselines")

    # Compute number of references for each common instance
    utils.log("Computing number of references")
    instance_num_refs = {}
    for corpus_id in tqdm(common_ids, desc="Computing # references"):
        pred = baseline_predictions[BASELINES[0]][corpus_id]
        num_refs = len(pred["gt_reference_ids"])
        instance_num_refs[corpus_id] = num_refs

    # Bucket instances by number of references
    num_buckets = len(BUCKET_LABELS)
    results = {baseline: [[] for _ in range(num_buckets)] for baseline in baseline_scores}
    for corpus_id in common_ids:
        num_refs = instance_num_refs[corpus_id]
        if num_refs <= 5:
            bucket_idx = num_refs - 1
        elif num_refs <= 10:
            bucket_idx = 5
        else:
            bucket_idx = 6
        for baseline in baseline_scores:
            results[baseline][bucket_idx].append(baseline_scores[baseline][corpus_id])

    # Compute means and standard errors
    bucket_means = {baseline: [] for baseline in baseline_scores}
    bucket_stderrs = {baseline: [] for baseline in baseline_scores}
    for baseline in baseline_scores:
        for bucket_idx in range(num_buckets):
            values = results[baseline][bucket_idx]
            if len(values) > 0:
                bucket_means[baseline].append(np.mean(values))
                bucket_stderrs[baseline].append(np.std(values) / np.sqrt(len(values)))
            else:
                bucket_means[baseline].append(np.nan)
                bucket_stderrs[baseline].append(np.nan)

    # Log bucket counts
    utils.log("Instances per bucket:")
    for bucket_idx in range(num_buckets):
        count = len(results[BASELINES[0]][bucket_idx])
        utils.log(f"  {BUCKET_LABELS[bucket_idx]}: {count}")

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(num_buckets)
    for baseline in baseline_scores:
        style = BASELINE_STYLES[baseline]
        label = BASELINE_LABELS[baseline]
        means = bucket_means[baseline]
        stderrs = bucket_stderrs[baseline]
        plt.errorbar(x, means, yerr=stderrs, label=label, color=style["color"],
                     linestyle=style["linestyle"], linewidth=1.5, alpha=0.9,
                     marker="o", markersize=3, capsize=2)

    plt.xlabel("# References", fontsize=8)
    plt.ylabel("Mean nDCG", fontsize=8)
    plt.title("nDCG vs # References", fontsize=9, fontweight="bold")
    plt.xticks(x, BUCKET_LABELS, fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ndcg_vs_num_references.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
