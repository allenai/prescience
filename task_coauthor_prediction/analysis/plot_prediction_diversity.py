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
    "frequency.one_shot": {"color": "#0072B2", "linestyle": "-"},
    "rank_fusion.grit.one_shot": {"color": "#9467bd", "linestyle": "-"},
    "embedding_fusion.grit.one_shot": {"color": "#E69F00", "linestyle": "-"},
    "hierarchical.grit.pca64.power0.5.first": {"color": "#D55E00", "linestyle": "-"},
    "mean_pooling_projected.grit.first": {"color": "#CC79A7", "linestyle": "-"},
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


def get_iso_week(date_str):
    """Get ISO week string (e.g., '2024-W41') for a date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def extrapolate_effective_count(observed_slots, observed_effective, target_slots):
    """Extrapolate effective count using power law fit: effective = A * slots^alpha."""
    if len(observed_slots) < 2 or observed_slots[-1] <= 0:
        return observed_effective[-1] if observed_effective else 0
    # Filter to positive values for log transform
    valid_mask = (np.array(observed_slots) > 0) & (np.array(observed_effective) > 0)
    slots = np.array(observed_slots)[valid_mask]
    effective = np.array(observed_effective)[valid_mask]
    if len(slots) < 2:
        return observed_effective[-1]
    # Fit log-log linear: log(eff) = log(A) + alpha * log(slots)
    log_slots = np.log(slots)
    log_eff = np.log(effective)
    alpha, log_A = np.polyfit(log_slots, log_eff, 1)
    return np.exp(log_A + alpha * np.log(target_slots))


def main():
    parser = argparse.ArgumentParser(description="Analyze cumulative prediction diversity")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--predictions_dir", type=str, default="data/task_coauthor_prediction/test/predictions", help="Predictions directory")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/analysis", help="Output directory")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top predictions per instance (default: match ground truth count)")
    parser.add_argument("--metric", type=str, default="effective", choices=["effective", "unique"], help="Diversity metric")
    parser.add_argument("--plot_type", type=str, default="line", choices=["line", "bar"], help="Type of plot to generate")
    args = parser.parse_args()

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} authors")

    # Load predictions for each baseline
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
        # Sort predictions by date
        sorted_preds = sorted(predictions, key=lambda p: corpus_id_to_date[p["corpus_id"]] if p["corpus_id"] in corpus_id_to_date else "9999-99-99")
        baseline_predictions[baseline] = sorted_preds

    # Use first baseline's sorted predictions as reference for ground truth (all have same instances)
    first_baseline = list(baseline_predictions.keys())[0]
    gt_sorted_preds = baseline_predictions[first_baseline]

    utils.log("Computing cumulative diversity metrics")
    cumulative_counts = {baseline: {} for baseline in baseline_predictions}
    cumulative_slot_counts = {baseline: 0 for baseline in baseline_predictions}
    slot_counts_history = {baseline: [] for baseline in baseline_predictions}  # Track slots at each checkpoint
    pred_indices = {baseline: 0 for baseline in baseline_predictions}
    last_week_with_data = {baseline: None for baseline in baseline_predictions}  # Track last week with new data
    gt_cumulative_counts = {}
    gt_pred_index = 0
    gt_cumulative_slots = 0

    for week_end in tqdm(week_end_dates, desc="Processing weeks"):
        # Process all predictions up to this week for each baseline
        for baseline, predictions in baseline_predictions.items():
            prev_index = pred_indices[baseline]
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
                num_preds = args.top_k if args.top_k is not None else len(pred["gt_coauthor_ids"])
                for author_id in pred["predicted_coauthor_ids"][:num_preds]:
                    if author_id not in cumulative_counts[baseline]:
                        cumulative_counts[baseline][author_id] = 0
                    cumulative_counts[baseline][author_id] += 1
                cumulative_slot_counts[baseline] += num_preds
                pred_indices[baseline] += 1
            # Track last week where this baseline had data
            if pred_indices[baseline] > prev_index:
                last_week_with_data[baseline] = len(weeks)  # Current week index (will be appended below)
            # Record slot count at this checkpoint
            slot_counts_history[baseline].append(cumulative_slot_counts[baseline])

        # Process ground truth up to this week (using first baseline's sorted predictions)
        while gt_pred_index < len(gt_sorted_preds):
            pred = gt_sorted_preds[gt_pred_index]
            corpus_id = pred["corpus_id"]
            if corpus_id not in corpus_id_to_date:
                gt_pred_index += 1
                continue
            pred_date = corpus_id_to_date[corpus_id]
            if pred_date > week_end:
                break
            # Add ground truth coauthors to cumulative counts
            for author_id in pred["gt_coauthor_ids"]:
                if author_id not in gt_cumulative_counts:
                    gt_cumulative_counts[author_id] = 0
                gt_cumulative_counts[author_id] += 1
            gt_cumulative_slots += len(pred["gt_coauthor_ids"])
            gt_pred_index += 1

        # Compute metrics for this week
        for baseline in baseline_predictions:
            metric_value = compute_metric(cumulative_counts[baseline], args.metric)
            results[baseline].append(metric_value)

        # Compute ground truth metric (only when not using fixed top_k)
        if args.top_k is None:
            gt_metric_value = compute_metric(gt_cumulative_counts, args.metric)
            ground_truth_values.append(gt_metric_value)

        # Compute reference line
        pool_size = count_authors_with_pubs_before(week_end, sd2publications, all_papers_dict)
        if args.top_k is not None:
            max_slots = max(cumulative_slot_counts[b] for b in baseline_predictions)
            reference_values.append(min(max_slots, pool_size))
        else:
            reference_values.append(min(gt_cumulative_slots, pool_size))

        weeks.append(week_end)

    # Log baseline coverage and determine completeness
    is_incomplete = {}
    for baseline in baseline_predictions:
        end_idx = last_week_with_data[baseline]
        if end_idx is not None and end_idx + 1 < len(weeks):
            utils.log(f"  {baseline}: data ends at {weeks[end_idx]} ({end_idx + 1}/{len(weeks)} weeks)")
            is_incomplete[baseline] = True
        else:
            utils.log(f"  {baseline}: complete ({len(weeks)} weeks)")
            is_incomplete[baseline] = False

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_top{args.top_k}" if args.top_k is not None else ""
    metric_label = "Effective # of Authors" if args.metric == "effective" else "Unique Authors"

    if args.plot_type == "line":
        # Line plot (existing behavior)
        utils.log("Creating line plot")
        plt.figure(figsize=(3.5, 2.8))
        plt.rcParams.update({"font.size": 8})
        week_dates = [datetime.strptime(w, "%Y-%m-%d") for w in weeks]

        # Plot each baseline (only up to last week with data for incomplete runs)
        for baseline in baseline_predictions:
            style = BASELINE_STYLES[baseline]
            label = BASELINE_LABELS[baseline]
            end_idx = last_week_with_data[baseline]
            if end_idx is not None:
                end_idx = end_idx + 1  # Include the last week with data
                plt.plot(week_dates[:end_idx], results[baseline][:end_idx], label=label, color=style["color"], linestyle=style["linestyle"], linewidth=1.5, alpha=0.9)
            else:
                plt.plot(week_dates, results[baseline], label=label, color=style["color"], linestyle=style["linestyle"], linewidth=1.5, alpha=0.9)

        # Plot ground truth line (only when not using fixed top_k)
        if args.top_k is None:
            plt.plot(week_dates, ground_truth_values, label="Ground Truth", color="#888888", linestyle="--", linewidth=1.5, alpha=0.9)

        # Plot reference line (max possible if picking at random)
        plt.plot(week_dates, reference_values, label="Max Possible (Random)", color="black", linestyle=":", linewidth=1.5, alpha=0.7)

        # Format plot
        plt.xlabel("Date", fontsize=8)
        plt.ylabel(metric_label, fontsize=8)
        plt.title(f"Cumulative Prediction Diversity", fontsize=9, fontweight="bold")
        plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
        plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout(pad=0.5)

        output_path = os.path.join(args.output_dir, f"prediction_diversity_{args.metric}{suffix}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        utils.log(f"Saved line plot to {output_path}")

    else:
        # Bar plot with extrapolation for incomplete runs
        utils.log("Creating bar plot with extrapolation")
        plt.figure(figsize=(4.5, 3.2))
        plt.rcParams.update({"font.size": 8})

        # Get target slots (from complete baselines)
        target_slots = max(slot_counts_history[b][-1] for b in baseline_predictions if not is_incomplete[b])
        utils.log(f"  Target slots for extrapolation: {target_slots}")

        # Compute final/extrapolated values for each baseline
        final_values = {}
        for baseline in baseline_predictions:
            if is_incomplete[baseline]:
                # Extrapolate using power law
                end_idx = last_week_with_data[baseline] + 1
                observed_slots = slot_counts_history[baseline][:end_idx]
                observed_effective = results[baseline][:end_idx]
                extrapolated = extrapolate_effective_count(observed_slots, observed_effective, target_slots)
                final_values[baseline] = extrapolated
                utils.log(f"  {baseline}: extrapolated {results[baseline][end_idx-1]:.0f} -> {extrapolated:.0f}")
            else:
                final_values[baseline] = results[baseline][-1]

        # Prepare bar data
        baselines_ordered = list(baseline_predictions.keys())
        labels = [BASELINE_LABELS[b] for b in baselines_ordered]
        values = [final_values[b] for b in baselines_ordered]
        colors = [BASELINE_STYLES[b]["color"] for b in baselines_ordered]

        # Create bars
        x_pos = np.arange(len(baselines_ordered))
        plt.bar(x_pos, values, color=colors, edgecolor="black", linewidth=0.5)

        # Reference lines
        gt_final = ground_truth_values[-1] if ground_truth_values else 0
        max_possible = reference_values[-1] if reference_values else 0
        if args.top_k is None and gt_final > 0:
            plt.axhline(gt_final, color="#888888", linestyle="--", linewidth=1.5, label="Ground Truth")
        plt.axhline(max_possible, color="black", linestyle=":", linewidth=1.5, label="Max Possible (Random)")

        # Format plot
        plt.xticks(x_pos, labels, rotation=30, ha="right", fontsize=7)
        plt.ylabel(metric_label, fontsize=8)
        plt.title(f"Prediction Diversity (Final)", fontsize=9, fontweight="bold")
        plt.legend(loc="upper right", fontsize=6, framealpha=0.9)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        plt.yticks(fontsize=7)
        plt.tight_layout(pad=0.5)

        output_path = os.path.join(args.output_dir, f"prediction_diversity_{args.metric}{suffix}_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        utils.log(f"Saved bar plot to {output_path}")


if __name__ == "__main__":
    main()
