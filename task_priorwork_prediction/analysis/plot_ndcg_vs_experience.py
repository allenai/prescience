"""nDCG vs mean author experience analysis for prior work prediction baselines."""
import os
import argparse
from bisect import bisect_left

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

BUCKET_EDGES = [0, 4, 7, 11, 21, 51, float("inf")]
BUCKET_LABELS = ["1-3", "4-6", "7-10", "11-20", "21-50", "51+"]


def get_experience(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get number of publications by author_id before cutoff_date."""
    author_pubs = sd2publications[author_id] if author_id in sd2publications else None
    if author_pubs is None:
        return 0
    pub_dates = [all_papers_dict[p]["date"] for p in author_pubs]
    idx = bisect_left(pub_dates, cutoff_date)
    return idx


def get_mean_experience(author_ids, cutoff_date, sd2publications, all_papers_dict):
    """Get mean experience across all authors."""
    if len(author_ids) == 0:
        return 0.0
    experiences = [get_experience(aid, cutoff_date, sd2publications, all_papers_dict) for aid in author_ids]
    return np.mean(experiences)


def main():
    parser = argparse.ArgumentParser(description="Analyze nDCG vs mean author experience")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--scored_dir", type=str, default="data/task_priorwork_prediction/test/scored", help="Scored predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/analysis", help="Output directory")
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

    # Compute mean experience for each common instance
    utils.log("Computing mean author experience")
    instance_experience = {}
    for corpus_id in tqdm(common_ids, desc="Computing experience"):
        paper = all_papers_dict[corpus_id]
        author_ids = [a["author_id"] for a in paper["authors"]]
        cutoff_date = paper["date"]
        experience = get_mean_experience(author_ids, cutoff_date, sd2publications, all_papers_dict)
        instance_experience[corpus_id] = experience

    # Bucket instances by experience
    num_buckets = len(BUCKET_LABELS)
    results = {baseline: [[] for _ in range(num_buckets)] for baseline in baseline_scores}
    for corpus_id in common_ids:
        experience = instance_experience[corpus_id]
        bucket_idx = bisect_left(BUCKET_EDGES[1:], experience)
        bucket_idx = min(bucket_idx, num_buckets - 1)
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

    plt.xlabel("Mean Author Experience (# Papers)", fontsize=8)
    plt.ylabel("Mean nDCG", fontsize=8)
    plt.title("nDCG vs Author Experience", fontsize=9, fontweight="bold")
    plt.xticks(x, BUCKET_LABELS, fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ndcg_vs_experience.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
