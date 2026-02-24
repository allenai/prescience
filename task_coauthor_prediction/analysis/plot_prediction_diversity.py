"""Cumulative prediction diversity analysis for coauthor prediction baselines."""
import os
import math
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta

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
    "frequency.one_shot": {"color": "#0072B2"},
    "rank_fusion.grit.one_shot": {"color": "#9467bd"},
    "embedding_fusion.grit.one_shot": {"color": "#E69F00"},
    "hierarchical.grit.pca64.power0.5.first": {"color": "#D55E00"},
    "mean_pooling_projected.grit.first": {"color": "#CC79A7"},
}


def compute_entropy(counts):
    """Compute entropy from frequency counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def compute_metric(counts, metric):
    """Compute diversity metric from frequency counts."""
    if metric == "effective":
        return math.exp(compute_entropy(counts))
    else:
        return len(counts)


def count_authors_with_pubs_before(cutoff_date, sd2publications, all_papers_dict):
    """Count authors with at least one publication before cutoff_date."""
    count = 0
    for author_id, pubs in sd2publications.items():
        if pubs is None:
            continue
        pub_dates = [all_papers_dict[p]["date"] for p in pubs]
        idx = bisect_left(pub_dates, cutoff_date)
        if idx > 0:
            count += 1
    return count


def get_week_end_date(date_str):
    """Get the Sunday (end of ISO week) for a given date string."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    days_until_sunday = (6 - dt.weekday()) % 7
    if days_until_sunday == 0 and dt.weekday() != 6:
        days_until_sunday = 7
    week_end = dt + timedelta(days=days_until_sunday)
    return week_end.strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Analyze cumulative prediction diversity")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/analysis", help="Output directory")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top predictions per instance (default: match ground truth count)")
    parser.add_argument("--metric", type=str, default="effective", choices=["effective", "unique"], help="Diversity metric")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} authors")

    utils.log("Loading predictions")
    baseline_predictions = {}
    for baseline in BASELINES:
        path = os.path.join(args.predictions_dir, f"predictions.{baseline}.json")
        if os.path.exists(path):
            data, _ = utils.load_json(path)
            baseline_predictions[baseline] = data
            utils.log(f"  Loaded {len(data)} predictions for {baseline}")
        else:
            utils.log(f"  WARNING: {path} not found, skipping {baseline}")

    corpus_id_to_date = {p["corpus_id"]: p["date"] for p in all_papers}

    all_dates = set()
    for baseline, predictions in baseline_predictions.items():
        for pred in predictions:
            corpus_id = pred["corpus_id"]
            if corpus_id in corpus_id_to_date:
                all_dates.add(corpus_id_to_date[corpus_id])

    week_end_dates = sorted(set(get_week_end_date(d) for d in all_dates))
    utils.log(f"Found {len(week_end_dates)} weekly checkpoints from {week_end_dates[0]} to {week_end_dates[-1]}")

    for baseline, predictions in baseline_predictions.items():
        sorted_preds = sorted(predictions, key=lambda p: corpus_id_to_date[p["corpus_id"]] if p["corpus_id"] in corpus_id_to_date else "9999-99-99")
        baseline_predictions[baseline] = sorted_preds

    first_baseline = list(baseline_predictions.keys())[0]
    gt_sorted_preds = baseline_predictions[first_baseline]

    utils.log("Computing cumulative diversity metrics")
    cumulative_counts = {baseline: {} for baseline in baseline_predictions}
    pred_indices = {baseline: 0 for baseline in baseline_predictions}
    gt_cumulative_counts = {}
    gt_pred_index = 0
    gt_cumulative_slots = 0

    final_results = {baseline: 0 for baseline in baseline_predictions}
    final_gt_value = 0
    final_reference_value = 0

    for week_end in tqdm(week_end_dates, desc="Processing weeks"):
        for baseline, predictions in baseline_predictions.items():
            while pred_indices[baseline] < len(predictions):
                pred = predictions[pred_indices[baseline]]
                corpus_id = pred["corpus_id"]
                if corpus_id not in corpus_id_to_date:
                    pred_indices[baseline] += 1
                    continue
                pred_date = corpus_id_to_date[corpus_id]
                if pred_date > week_end:
                    break
                num_preds = args.top_k if args.top_k is not None else len(pred["gt_coauthor_ids"])
                for author_id in pred["predicted_coauthor_ids"][:num_preds]:
                    if author_id not in cumulative_counts[baseline]:
                        cumulative_counts[baseline][author_id] = 0
                    cumulative_counts[baseline][author_id] += 1
                pred_indices[baseline] += 1

        while gt_pred_index < len(gt_sorted_preds):
            pred = gt_sorted_preds[gt_pred_index]
            corpus_id = pred["corpus_id"]
            if corpus_id not in corpus_id_to_date:
                gt_pred_index += 1
                continue
            pred_date = corpus_id_to_date[corpus_id]
            if pred_date > week_end:
                break
            for author_id in pred["gt_coauthor_ids"]:
                if author_id not in gt_cumulative_counts:
                    gt_cumulative_counts[author_id] = 0
                gt_cumulative_counts[author_id] += 1
            gt_cumulative_slots += len(pred["gt_coauthor_ids"])
            gt_pred_index += 1

        for baseline in baseline_predictions:
            final_results[baseline] = compute_metric(cumulative_counts[baseline], args.metric)

        if args.top_k is None:
            final_gt_value = compute_metric(gt_cumulative_counts, args.metric)

        pool_size = count_authors_with_pubs_before(week_end, sd2publications, all_papers_dict)
        final_reference_value = min(gt_cumulative_slots, pool_size)

    utils.log("Creating bar plot")
    plt.figure(figsize=(4.5, 3.2))
    plt.rcParams.update({"font.size": 8})

    baselines_ordered = list(baseline_predictions.keys())
    labels = [BASELINE_LABELS[b] for b in baselines_ordered]
    values = [final_results[b] for b in baselines_ordered]
    colors = [BASELINE_STYLES[b]["color"] for b in baselines_ordered]

    x_pos = np.arange(len(baselines_ordered))
    plt.bar(x_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    if args.top_k is None and final_gt_value > 0:
        plt.axhline(final_gt_value, color="#888888", linestyle="--", linewidth=1.5, label="Ground Truth")
    plt.axhline(final_reference_value, color="black", linestyle=":", linewidth=1.5, label="Max Possible (Random)")

    metric_label = "Effective # of Authors" if args.metric == "effective" else "Unique Authors"
    suffix = f"_top{args.top_k}" if args.top_k is not None else ""
    plt.xticks(x_pos, labels, rotation=30, ha="right", fontsize=7)
    plt.ylabel(metric_label, fontsize=8)
    plt.title("Prediction Diversity", fontsize=9, fontweight="bold")
    plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"prediction_diversity_{args.metric}{suffix}_bar.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved bar plot to {output_path}")


if __name__ == "__main__":
    main()
