"""Plot cumulative author diversity comparing natural vs synthetic corpora."""
import os
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import utils

NATURAL_COLOR = "#0072B2"
SYNTHETIC_COLOR = "#E69F00"


def main():
    parser = argparse.ArgumentParser(description="Plot cumulative author diversity.")
    parser.add_argument("--natural_path", type=str, required=True, help="Path to natural author diversity JSON")
    parser.add_argument("--synthetic_paths", type=str, nargs="+", required=True, help="Paths to synthetic author diversity JSONs")
    parser.add_argument("--output_path", type=str, default="figures/multiturn/author_diversity.png", help="Output figure path")
    parser.add_argument("--metric", type=str, default="effective", choices=["effective", "unique"], help="Metric to plot")
    args = parser.parse_args()

    utils.log("Loading natural data")
    natural_data, _ = utils.load_json(args.natural_path)
    if isinstance(natural_data, dict) and "data" in natural_data:
        natural_data = natural_data["data"]

    utils.log("Loading synthetic data")
    synthetic_by_bucket = defaultdict(list)
    for path in args.synthetic_paths:
        utils.log(f"Processing {path}")
        data, _ = utils.load_json(path)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        for record in data:
            bucket = record.get("bucket") or record.get("week_end")
            synthetic_by_bucket[bucket].append(record[f"{args.metric}_count"])

    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 2.8))
    plt.rcParams.update({"font.size": 8})

    # Plot natural
    nat_buckets = [r.get("bucket") or r.get("week_end") for r in natural_data]
    nat_dates = [datetime.strptime(b, "%Y-%m-%d") for b in nat_buckets]
    nat_values = [r[f"{args.metric}_count"] for r in natural_data]
    plt.plot(nat_dates, nat_values, label="Natural", color=NATURAL_COLOR, linewidth=1.5)

    # Plot synthetic (mean across seeds)
    synth_buckets = sorted(synthetic_by_bucket.keys())
    synth_dates = [datetime.strptime(b, "%Y-%m-%d") for b in synth_buckets]
    synth_means = [np.mean(synthetic_by_bucket[b]) for b in synth_buckets]
    plt.plot(synth_dates, synth_means, label="Synthetic", color=SYNTHETIC_COLOR, linewidth=1.5)

    metric_label = "Effective # of Authors" if args.metric == "effective" else "Unique Authors"
    plt.xlabel("Date", fontsize=8)
    plt.ylabel(metric_label, fontsize=8)
    plt.title("Cumulative Author Diversity", fontsize=9, fontweight="bold")
    plt.legend(loc="best", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
