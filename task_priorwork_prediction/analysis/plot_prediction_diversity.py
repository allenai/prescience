"""Cumulative prediction diversity analysis for prior work prediction baselines."""
import os
import math
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta

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


def count_papers_before(cutoff_date, all_papers_dict):
    """Count papers published before cutoff_date."""
    count = 0
    for paper in all_papers_dict.values():
        if paper["date"] < cutoff_date:
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
    parser = argparse.ArgumentParser(description="Analyze cumulative prediction diversity for prior work")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_priorwork_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/analysis", help="Output directory")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top predictions per instance (default: match ground truth count)")
    parser.add_argument("--metric", type=str, default="effective", choices=["effective", "unique"], help="Diversity metric")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers")

    # Load predictions for each baseline
    utils.log("Loading predictions")
    baseline_predictions = {}
    for baseline in BASELINES:
        path = os.path.join(args.predictions_dir, f"predictions.{baseline}.json")
        if os.path.exists(path):
            data_file, _ = utils.load_json(path)
            data = data_file["data"] if "data" in data_file else data_file
            baseline_predictions[baseline] = data
            utils.log(f"  Loaded {len(data)} predictions for {baseline}")
        else:
            utils.log(f"  WARNING: {path} not found, skipping {baseline}")

    # Get corpus_id -> date mapping and sort predictions by date
    corpus_id_to_date = {p["corpus_id"]: p["date"] for p in all_papers}

    # Get unique weeks from all predictions
    all_dates = set()
    for baseline, predictions in baseline_predictions.items():
        for pred in predictions:
            corpus_id = pred["corpus_id"]
            if corpus_id in corpus_id_to_date:
                all_dates.add(corpus_id_to_date[corpus_id])

    week_end_dates = sorted(set(get_week_end_date(d) for d in all_dates))
    utils.log(f"Found {len(week_end_dates)} weekly checkpoints from {week_end_dates[0]} to {week_end_dates[-1]}")

    # For each baseline, sort predictions by date and compute cumulative metrics
    results = {baseline: [] for baseline in baseline_predictions}
    ground_truth_values = []
    reference_values = []
    weeks = []

    for baseline, predictions in baseline_predictions.items():
        sorted_preds = sorted(predictions, key=lambda p: corpus_id_to_date[p["corpus_id"]] if p["corpus_id"] in corpus_id_to_date else "9999-99-99")
        baseline_predictions[baseline] = sorted_preds

    # Use first baseline's sorted predictions as reference for ground truth
    first_baseline = list(baseline_predictions.keys())[0]
    gt_sorted_preds = baseline_predictions[first_baseline]

    utils.log("Computing cumulative diversity metrics")
    cumulative_counts = {baseline: {} for baseline in baseline_predictions}
    cumulative_slot_counts = {baseline: 0 for baseline in baseline_predictions}
    pred_indices = {baseline: 0 for baseline in baseline_predictions}
    gt_cumulative_counts = {}
    gt_pred_index = 0
    gt_cumulative_slots = 0

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
                # Add top k predictions (or top R where R = ground truth count if k not specified)
                num_preds = args.top_k if args.top_k is not None else len(pred["gt_reference_ids"])
                for paper_id in pred["predicted_reference_ids"][:num_preds]:
                    if paper_id not in cumulative_counts[baseline]:
                        cumulative_counts[baseline][paper_id] = 0
                    cumulative_counts[baseline][paper_id] += 1
                cumulative_slot_counts[baseline] += num_preds
                pred_indices[baseline] += 1

        # Process ground truth up to this week
        while gt_pred_index < len(gt_sorted_preds):
            pred = gt_sorted_preds[gt_pred_index]
            corpus_id = pred["corpus_id"]
            if corpus_id not in corpus_id_to_date:
                gt_pred_index += 1
                continue
            pred_date = corpus_id_to_date[corpus_id]
            if pred_date > week_end:
                break
            for paper_id in pred["gt_reference_ids"]:
                if paper_id not in gt_cumulative_counts:
                    gt_cumulative_counts[paper_id] = 0
                gt_cumulative_counts[paper_id] += 1
            gt_cumulative_slots += len(pred["gt_reference_ids"])
            gt_pred_index += 1

        for baseline in baseline_predictions:
            metric_value = compute_metric(cumulative_counts[baseline], args.metric)
            results[baseline].append(metric_value)

        # Compute ground truth metric (only when not using fixed top_k)
        if args.top_k is None:
            gt_metric_value = compute_metric(gt_cumulative_counts, args.metric)
            ground_truth_values.append(gt_metric_value)

        # Compute reference line
        pool_size = count_papers_before(week_end, all_papers_dict)
        if args.top_k is not None:
            max_slots = max(cumulative_slot_counts[b] for b in baseline_predictions)
            reference_values.append(min(max_slots, pool_size))
        else:
            reference_values.append(min(gt_cumulative_slots, pool_size))

        weeks.append(week_end)

    # Plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    week_dates = [datetime.strptime(w, "%Y-%m-%d") for w in weeks]

    for baseline in baseline_predictions:
        style = BASELINE_STYLES[baseline]
        label = BASELINE_LABELS[baseline]
        plt.plot(week_dates, results[baseline], label=label, color=style["color"], linestyle=style["linestyle"], linewidth=1.5, alpha=0.9)

    # Plot ground truth line (only when not using fixed top_k)
    if args.top_k is None:
        plt.plot(week_dates, ground_truth_values, label="Ground Truth", color="#888888", linestyle="--", linewidth=1.5, alpha=0.9)

    # Plot reference line (max possible if picking at random)
    plt.plot(week_dates, reference_values, label="Max Possible (Random)", color="black", linestyle=":", linewidth=1.5, alpha=0.7)

    metric_label = "Effective # of Papers" if args.metric == "effective" else "Unique Papers"
    plt.xlabel("Date", fontsize=8)
    plt.ylabel(metric_label, fontsize=8)
    plt.title("Cumulative Prediction Diversity", fontsize=9, fontweight="bold")
    plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_top{args.top_k}" if args.top_k is not None else ""
    output_path = os.path.join(args.output_dir, f"prediction_diversity_{args.metric}{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
