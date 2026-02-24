"""Plot topic proportions for natural and synthetic target papers."""
import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import utils

OTHER_LABEL = "other"
PALETTE = ["#0072B2", "#E69F00", "#2CA02C", "#D55E00", "#9467bd", "#CC79A7", "#888888"]


def partition_by_date(papers):
    date2papers = defaultdict(list)
    for paper in papers.values():
        date = paper["date"]
        date2papers[date].append(paper)
    return dict(sorted(date2papers.items()))


def bucket_daily_to_weekly(partitioned):
    weekly = defaultdict(list)
    for date_str, papers in partitioned.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        bucket_start = datetime.strptime(f"{iso_year}-W{iso_week}-1", "%G-W%V-%u")
        weekly[bucket_start].extend(papers)
    return dict(sorted(weekly.items()))


def compute_topic_lookup_natural(papers):
    lookup = {}
    for paper in papers.values():
        categories = paper["categories"] if "categories" in paper else None
        if categories:
            lookup[paper["corpus_id"]] = categories[0]
        else:
            lookup[paper["corpus_id"]] = OTHER_LABEL
    return lookup


def load_synthetic_topics(predictions_path):
    predictions, _ = utils.load_json(predictions_path)
    lookup = {}
    for corpus_id, value in predictions.items():
        if isinstance(value, dict) and "primary_category" in value:
            lookup[corpus_id] = value["primary_category"]
        else:
            lookup[corpus_id] = value
    return lookup


def compute_topic_proportions(partitioned, topic_lookup, topics):
    topic2series = {topic: {"dates": [], "values": []} for topic in topics}
    proportions_by_date = {}
    for dt, papers in partitioned.items():
        total = len(papers)
        if total == 0:
            continue
        counts = Counter()
        for paper in papers:
            corpus_id = paper["corpus_id"]
            label = topic_lookup[corpus_id] if corpus_id in topic_lookup else OTHER_LABEL
            counts[label] += 1
        proportions = {}
        for topic in topics:
            proportion = counts[topic] / total if topic in counts else 0.0
            topic2series[topic]["dates"].append(dt)
            topic2series[topic]["values"].append(proportion)
            proportions[topic] = proportion
        proportions_by_date[dt] = proportions
    return topic2series, proportions_by_date


def main():
    parser = argparse.ArgumentParser(description="Plot topic proportions for natural and synthetic target papers.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic experiment directory under data/multiturn/")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07", help="Model name used for synthetic classifications")
    parser.add_argument("--predictions_path", type=str, default=None, help="Optional explicit path to synthetic topic predictions JSON")
    parser.add_argument("--top_k_topics", type=int, default=6, help="Number of most common topics to display (plus 'other')")
    parser.add_argument("--output_path", type=str, default="figures/multiturn/topic_distribution_over_time.png", help="Path to save the figure")
    parser.add_argument("--sampled", action="store_true", help="Use LLM-classified topics for both natural and synthetic (for sampled predictions)")
    args = parser.parse_args()

    all_papers_nat, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    all_papers_syn = utils.load_json(f"data/multiturn/{args.synthetic_dir}/all_papers.json")[0]

    all_papers_nat_dict = {paper["corpus_id"]: paper for paper in all_papers_nat}
    all_papers_syn_dict = {paper["corpus_id"]: paper for paper in all_papers_syn}

    target_nat = {cid: paper for cid, paper in all_papers_nat_dict.items() if "target" in paper["roles"]}
    target_syn = {
        cid: paper for cid, paper in all_papers_syn_dict.items()
        if "synthetic" in paper["roles"]
    }

    sampled_suffix = "_sampled" if args.sampled else ""

    if args.sampled:
        nat_predictions_path = os.path.join("data/multiturn", args.synthetic_dir, f"natural_primary_categories_{args.model}{sampled_suffix}.json")
        natural_topic_lookup = load_synthetic_topics(nat_predictions_path)
        target_nat = {cid: paper for cid, paper in target_nat.items() if cid in natural_topic_lookup}
        print(f"Using LLM-classified topics for {len(target_nat)} natural papers.")
    else:
        natural_topic_lookup = compute_topic_lookup_natural(target_nat)

    if args.predictions_path:
        predictions_path = args.predictions_path
    else:
        predictions_path = os.path.join("data/multiturn", args.synthetic_dir, f"synthetic_primary_categories_{args.model}{sampled_suffix}.json")

    synthetic_topic_lookup = load_synthetic_topics(predictions_path)
    if args.sampled:
        target_syn = {cid: paper for cid, paper in target_syn.items() if cid in synthetic_topic_lookup}
        print(f"Using LLM-classified topics for {len(target_syn)} synthetic papers.")
    else:
        missing = [cid for cid in target_syn if cid not in synthetic_topic_lookup]
        if missing:
            print(f"Warning: missing synthetic topic predictions for {len(missing)} papers; assigning '{OTHER_LABEL}'.")
            for cid in missing:
                synthetic_topic_lookup[cid] = OTHER_LABEL

    partitioned_nat_daily = partition_by_date(target_nat)
    partitioned_syn_daily = partition_by_date(target_syn)
    partitioned_nat = bucket_daily_to_weekly(partitioned_nat_daily)
    partitioned_syn = bucket_daily_to_weekly(partitioned_syn_daily)

    overall_counts = Counter()
    for cid in target_nat:
        overall_counts[natural_topic_lookup[cid]] += 1
    for cid in target_syn:
        label = synthetic_topic_lookup[cid] if cid in synthetic_topic_lookup else OTHER_LABEL
        overall_counts[label] += 1

    topics = [topic for topic, _ in overall_counts.most_common(args.top_k_topics)]
    if OTHER_LABEL not in topics:
        topics.append(OTHER_LABEL)

    nat_series, nat_proportions_by_date = compute_topic_proportions(partitioned_nat, natural_topic_lookup, topics)
    syn_series, syn_proportions_by_date = compute_topic_proportions(partitioned_syn, synthetic_topic_lookup, topics)

    plt.rcParams.update({"font.size": 8})
    colors = {}
    for idx, topic in enumerate(topics):
        colors[topic] = PALETTE[idx % len(PALETTE)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    max_value = 0.0
    for topic in topics:
        max_value = max(max_value, max(nat_series[topic]["values"], default=0.0))
        max_value = max(max_value, max(syn_series[topic]["values"], default=0.0))
    upper = min(1.0, max(0.05, max_value) + 0.05)
    ylim = (0.0, round(upper, 2))

    left_ax, right_ax = axes
    left_ax.set_title("Natural target papers", fontsize=9, fontweight="bold")
    right_ax.set_title("Synthetic target papers", fontsize=9, fontweight="bold")

    for topic in topics:
        if nat_series[topic]["dates"]:
            left_ax.plot(
                nat_series[topic]["dates"],
                nat_series[topic]["values"],
                label=topic,
                color=colors[topic],
                linewidth=1.5, alpha=0.9,
            )
        if syn_series[topic]["dates"]:
            right_ax.plot(
                syn_series[topic]["dates"],
                syn_series[topic]["values"],
                label=topic,
                color=colors[topic],
                linewidth=1.5, alpha=0.9,
            )

    left_ax.set_ylabel("Proportion of target papers", fontsize=8)
    right_ax.set_ylabel("")

    nat_dates = sorted({dt for topic in topics for dt in nat_series[topic]["dates"]})
    syn_dates = sorted({dt for topic in topics for dt in syn_series[topic]["dates"]})

    for ax, dates in ((left_ax, nat_dates), (right_ax, syn_dates)):
        ax.set_ylim(ylim)
        ax.set_xlabel("Week starting", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        if dates:
            step = max(len(dates) // 14, 1)
            ax.set_xticks(dates[::step])
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            ax.set_xticks([])

    if left_ax.lines:
        left_ax.legend(fontsize=6, framealpha=0.9)
    if right_ax.lines:
        right_ax.legend(fontsize=6, framealpha=0.9)

    output_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(pad=0.5)
    print(f"Saving figure to {args.output_path}")
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close()

    def compute_entropy(proportions_by_date):
        dates = sorted(proportions_by_date.keys())
        entropy_values = []
        for dt in dates:
            probs = np.array([proportions_by_date[dt][topic] for topic in topics], dtype=np.float32)
            mask = probs > 0
            if mask.any():
                entropy = float(-(probs[mask] * np.log(probs[mask])).sum())
            else:
                entropy = 0.0
            entropy_values.append(entropy)
        return dates, entropy_values

    nat_entropy_dates, nat_entropy = compute_entropy(nat_proportions_by_date)
    syn_entropy_dates, syn_entropy = compute_entropy(syn_proportions_by_date)

    fig_entropy, ax_entropy = plt.subplots(figsize=(5, 3.5))
    ax_entropy.set_title("Entropy of weekly topic distribution", fontsize=9, fontweight="bold")
    if nat_entropy_dates:
        ax_entropy.plot(nat_entropy_dates, nat_entropy, label="Natural target papers", color="#0072B2", linewidth=1.5, alpha=0.9)
    if syn_entropy_dates:
        ax_entropy.plot(syn_entropy_dates, syn_entropy, label="Synthetic target papers", color="#E69F00", linewidth=1.5, alpha=0.9)
    ax_entropy.set_xlabel("Week starting", fontsize=8)
    ax_entropy.set_ylabel("Entropy (nats)", fontsize=8)
    ax_entropy.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax_entropy.legend(fontsize=6, framealpha=0.9)
    if nat_entropy_dates or syn_entropy_dates:
        dates_for_ticks = sorted(set(nat_entropy_dates + syn_entropy_dates))
        step = max(len(dates_for_ticks) // 14, 1)
        ax_entropy.set_xticks(dates_for_ticks[::step])
        ax_entropy.tick_params(axis="x", rotation=45, labelsize=7)
        ax_entropy.tick_params(axis="y", labelsize=7)
        ax_entropy.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    entropy_output_path = args.output_path.replace(".png", "_entropy.png")
    print(f"Saving figure to {entropy_output_path}")
    fig_entropy.tight_layout(pad=0.5)
    fig_entropy.savefig(entropy_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig_entropy)


if __name__ == "__main__":
    main()
