"""nDCG vs collaboration novelty analysis for coauthor prediction baselines."""
import os
import argparse
from bisect import bisect_left

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


def get_prior_collaborators(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get set of authors who have collaborated with author_id before cutoff_date."""
    collaborators = set()
    author_pubs = sd2publications[author_id] if author_id in sd2publications else None
    if author_pubs is None:
        return collaborators

    pub_dates = [all_papers_dict[p]["date"] for p in author_pubs]
    idx = bisect_left(pub_dates, cutoff_date)
    prior_pubs = author_pubs[:idx]

    for pub_id in prior_pubs:
        paper = all_papers_dict[pub_id]
        if "authors" in paper:
            for author in paper["authors"]:
                if author["author_id"] != author_id:
                    collaborators.add(author["author_id"])

    return collaborators


def compute_novelty_fraction(first_author_id, gt_coauthor_ids, cutoff_date, sd2publications, all_papers_dict):
    """Compute fraction of GT co-authors who are new (never collaborated with first author before)."""
    if len(gt_coauthor_ids) == 0:
        return 0.0

    prior_collaborators = get_prior_collaborators(first_author_id, cutoff_date, sd2publications, all_papers_dict)
    new_count = sum(1 for cid in gt_coauthor_ids if cid not in prior_collaborators)
    return new_count / len(gt_coauthor_ids)


def main():
    parser = argparse.ArgumentParser(description="Analyze nDCG vs collaboration novelty")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--scored_dir", type=str, default="data/task_coauthor_prediction/test/scored", help="Scored predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/analysis", help="Output directory")
    parser.add_argument("--num_buckets", type=int, default=5, help="Number of novelty buckets")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} authors")

    # Load predictions and scored files
    utils.log("Loading predictions and scores")
    baseline_predictions = {}
    baseline_scores = {}
    for baseline in BASELINES:
        pred_path = os.path.join(args.predictions_dir, f"predictions.{baseline}.json")
        scored_path = os.path.join(args.scored_dir, f"predictions.{baseline}.eval.json")
        if os.path.exists(pred_path) and os.path.exists(scored_path):
            pred_data, _ = utils.load_json(pred_path)
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

    # Compute novelty fraction for each common instance
    utils.log("Computing novelty fractions")
    instance_novelty = {}
    for corpus_id in tqdm(common_ids, desc="Computing novelty"):
        pred = baseline_predictions[BASELINES[0]][corpus_id]
        first_author_id = pred["first_author_id"]
        gt_coauthor_ids = pred["gt_coauthor_ids"]
        cutoff_date = all_papers_dict[corpus_id]["date"]
        novelty = compute_novelty_fraction(first_author_id, gt_coauthor_ids, cutoff_date, sd2publications, all_papers_dict)
        instance_novelty[corpus_id] = novelty

    # Bucket instances by novelty fraction
    bucket_edges = np.linspace(0, 1, args.num_buckets + 1)
    bucket_labels = [f"{int(bucket_edges[i]*100)}-{int(bucket_edges[i+1]*100)}%" for i in range(args.num_buckets)]

    # Compute mean nDCG per bucket per baseline
    results = {baseline: [[] for _ in range(args.num_buckets)] for baseline in baseline_scores}
    for corpus_id in common_ids:
        novelty = instance_novelty[corpus_id]
        bucket_idx = min(int(novelty * args.num_buckets), args.num_buckets - 1)
        for baseline in baseline_scores:
            results[baseline][bucket_idx].append(baseline_scores[baseline][corpus_id])

    # Compute means and standard errors
    bucket_means = {baseline: [] for baseline in baseline_scores}
    bucket_stderrs = {baseline: [] for baseline in baseline_scores}
    for baseline in baseline_scores:
        for bucket_idx in range(args.num_buckets):
            values = results[baseline][bucket_idx]
            if len(values) > 0:
                bucket_means[baseline].append(np.mean(values))
                bucket_stderrs[baseline].append(np.std(values) / np.sqrt(len(values)))
            else:
                bucket_means[baseline].append(np.nan)
                bucket_stderrs[baseline].append(np.nan)

    # Log bucket counts
    utils.log("Instances per bucket:")
    for bucket_idx in range(args.num_buckets):
        count = len(results[BASELINES[0]][bucket_idx])
        utils.log(f"  {bucket_labels[bucket_idx]}: {count}")

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    x = np.arange(args.num_buckets)
    for baseline in baseline_scores:
        style = BASELINE_STYLES[baseline]
        label = BASELINE_LABELS[baseline]
        means = bucket_means[baseline]
        stderrs = bucket_stderrs[baseline]
        plt.errorbar(x, means, yerr=stderrs, label=label, color=style["color"],
                     linestyle=style["linestyle"], linewidth=1.5, alpha=0.9,
                     marker="o", markersize=3, capsize=2)

    plt.xlabel("Novelty Fraction", fontsize=8)
    plt.ylabel("Mean nDCG", fontsize=8)
    plt.title("nDCG vs Collaboration Novelty", fontsize=9, fontweight="bold")
    plt.xticks(x, bucket_labels, fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ndcg_vs_novelty.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
