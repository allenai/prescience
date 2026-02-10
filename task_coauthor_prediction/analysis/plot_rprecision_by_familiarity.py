"""R-Precision decomposition by collaborator familiarity for coauthor prediction baselines."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils


BASELINES = [
    "frequency.one_shot",
    "rank_fusion.grit.one_shot",
    "embedding_fusion.grit.one_shot",
    "hierarchical.grit.pca64.power0.5.first",
    "mean_pooling_projected.grit.first",
]

BASELINE_LABELS = {
    "frequency.one_shot": "Frequency",
    "rank_fusion.grit.one_shot": "Rank Fusion",
    "embedding_fusion.grit.one_shot": "Embedding Fusion",
    "hierarchical.grit.pca64.power0.5.first": "Hierarchical",
    "mean_pooling_projected.grit.first": "Embedding Fusion (Projected)",
}

BASELINE_STYLES = {
    "frequency.one_shot": {"color": "#0072B2", "linestyle": "-"},
    "rank_fusion.grit.one_shot": {"color": "#9467bd", "linestyle": "-"},
    "embedding_fusion.grit.one_shot": {"color": "#E69F00", "linestyle": "-"},
    "hierarchical.grit.pca64.power0.5.first": {"color": "#D55E00", "linestyle": "-"},
    "mean_pooling_projected.grit.first": {"color": "#CC79A7", "linestyle": "-"},
}

FAMILIARITY_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 5), (6, float("inf"))]
BUCKET_LABELS = ["0", "1", "2-3", "4-5", "6+"]


def get_bucket_index(familiarity):
    """Get bucket index for a familiarity count."""
    for i, (low, high) in enumerate(FAMILIARITY_BUCKETS):
        if low <= familiarity <= high:
            return i
    return len(FAMILIARITY_BUCKETS) - 1


def compute_coauthor_familiarity(first_author_id, coauthor_id, cutoff_date, sd2publications, all_papers_dict):
    """Count prior collaborations between first_author and coauthor before cutoff_date."""
    count = 0
    if first_author_id not in sd2publications:
        return count

    for pub_id in sd2publications[first_author_id]:
        paper = all_papers_dict[pub_id]
        if paper["date"] >= cutoff_date:
            break
        if "authors" in paper:
            for author in paper["authors"]:
                if author["author_id"] == coauthor_id:
                    count += 1
                    break
    return count


def main():
    parser = argparse.ArgumentParser(description="Analyze R-Precision by collaborator familiarity")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} authors")

    utils.log("Loading predictions")
    baseline_predictions = {}
    for baseline in BASELINES:
        pred_path = os.path.join(args.predictions_dir, f"predictions.{baseline}.json")
        if os.path.exists(pred_path):
            pred_data, _ = utils.load_json(pred_path)
            baseline_predictions[baseline] = {p["corpus_id"]: p for p in pred_data}
            utils.log(f"  {baseline}: {len(baseline_predictions[baseline])} predictions")
        else:
            utils.log(f"  WARNING: Missing {pred_path}")

    common_ids = None
    for baseline in baseline_predictions:
        if common_ids is None:
            common_ids = set(baseline_predictions[baseline].keys())
        else:
            common_ids = common_ids & set(baseline_predictions[baseline].keys())
    utils.log(f"Found {len(common_ids)} common instances across all baselines")

    utils.log("Computing hit rates by familiarity bucket")
    results = {baseline: [[] for _ in FAMILIARITY_BUCKETS] for baseline in baseline_predictions}

    for corpus_id in tqdm(common_ids, desc="Processing instances"):
        paper = all_papers_dict[corpus_id]
        cutoff_date = paper["date"]

        pred = baseline_predictions[BASELINES[0]][corpus_id]
        first_author_id = pred["first_author_id"]
        gt_coauthor_ids = pred["gt_coauthor_ids"]

        if len(gt_coauthor_ids) == 0:
            continue

        k = len(gt_coauthor_ids)

        for baseline in baseline_predictions:
            pred = baseline_predictions[baseline][corpus_id]
            predicted_ids = pred["predicted_coauthor_ids"][:k]
            predicted_set = set(predicted_ids)

            for gt_id in gt_coauthor_ids:
                familiarity = compute_coauthor_familiarity(first_author_id, gt_id, cutoff_date, sd2publications, all_papers_dict)
                bucket_idx = get_bucket_index(familiarity)
                is_hit = 1.0 if gt_id in predicted_set else 0.0
                results[baseline][bucket_idx].append(is_hit)

    bucket_means = {baseline: [] for baseline in baseline_predictions}
    bucket_stderrs = {baseline: [] for baseline in baseline_predictions}
    for baseline in baseline_predictions:
        for bucket_idx in range(len(FAMILIARITY_BUCKETS)):
            values = results[baseline][bucket_idx]
            if len(values) > 0:
                bucket_means[baseline].append(np.mean(values))
                bucket_stderrs[baseline].append(np.std(values) / np.sqrt(len(values)))
            else:
                bucket_means[baseline].append(np.nan)
                bucket_stderrs[baseline].append(np.nan)

    utils.log("Instances per bucket:")
    for bucket_idx in range(len(FAMILIARITY_BUCKETS)):
        count = len(results[BASELINES[0]][bucket_idx])
        utils.log(f"  {BUCKET_LABELS[bucket_idx]}: {count}")

    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(len(FAMILIARITY_BUCKETS))
    for baseline in baseline_predictions:
        style = BASELINE_STYLES[baseline]
        label = BASELINE_LABELS[baseline]
        means = bucket_means[baseline]
        stderrs = bucket_stderrs[baseline]
        plt.errorbar(x, means, yerr=stderrs, label=label, color=style["color"],
                     linestyle=style["linestyle"], linewidth=1.5, alpha=0.9,
                     marker="o", markersize=3, capsize=2)

    plt.xlabel("Prior Collaborations", fontsize=8)
    plt.ylabel("Hit Rate", fontsize=8)
    plt.xticks(x, BUCKET_LABELS, fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc="upper left", fontsize=5, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "rprecision_by_familiarity.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    summary = {
        "bucket_labels": BUCKET_LABELS,
        "bucket_counts": [len(results[BASELINES[0]][i]) for i in range(len(FAMILIARITY_BUCKETS))],
        "baseline_hit_rates": {baseline: bucket_means[baseline] for baseline in baseline_predictions},
    }
    summary_path = os.path.join(args.output_dir, "rprecision_by_familiarity_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
