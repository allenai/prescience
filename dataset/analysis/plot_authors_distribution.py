"""Distribution of authors per target paper."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def main():
    parser = argparse.ArgumentParser(description="Plot distribution of authors per target paper")
    parser.add_argument("--train_dir", type=str, default="data/corpus/train", help="Train corpus directory")
    parser.add_argument("--test_dir", type=str, default="data/corpus/test", help="Test corpus directory")
    parser.add_argument("--output_dir", type=str, default="data/corpus/analysis", help="Output directory")
    parser.add_argument("--max_authors", type=int, default=15, help="Cap author count for visualization")
    args = parser.parse_args()

    utils.log(f"Loading train corpus from {args.train_dir}")
    train_all_papers, _ = utils.load_json(os.path.join(args.train_dir, "all_papers.json"))
    train_target_papers = [p for p in train_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(train_target_papers)} train target papers")

    utils.log(f"Loading test corpus from {args.test_dir}")
    test_all_papers, _ = utils.load_json(os.path.join(args.test_dir, "all_papers.json"))
    test_target_papers = [p for p in test_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(test_target_papers)} test target papers")

    train_counts_raw = [len(p["authors"]) if "authors" in p else 0 for p in train_target_papers]
    test_counts_raw = [len(p["authors"]) if "authors" in p else 0 for p in test_target_papers]
    train_counts = [c for c in train_counts_raw if c <= args.max_authors]
    test_counts = [c for c in test_counts_raw if c <= args.max_authors]
    utils.log(f"Filtered to {len(train_counts)} train papers (<= {args.max_authors} authors), excluded {len(train_counts_raw) - len(train_counts)}")
    utils.log(f"Filtered to {len(test_counts)} test papers (<= {args.max_authors} authors), excluded {len(test_counts_raw) - len(test_counts)}")

    bins = range(0, args.max_authors + 2)

    utils.log("Creating plot")
    plt.figure(figsize=(4.5, 3.5))
    plt.rcParams.update({"font.size": 8})

    plt.hist(train_counts, bins=bins, alpha=0.5, label="Train", color="orange")
    plt.hist(test_counts, bins=bins, alpha=0.5, label="Test", color="blue")

    plt.xlabel("Number of Authors", fontsize=8)
    plt.ylabel("Number of Papers", fontsize=8)
    plt.title("Distribution of Authors in Target Papers", fontsize=9, fontweight="bold")
    plt.xticks(list(bins), fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(fontsize=7, framealpha=0.9)
    plt.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "authors_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    summary = {
        "train_target_papers": len(train_target_papers),
        "test_target_papers": len(test_target_papers),
        "max_authors_cap": args.max_authors,
        "train_stats": {
            "mean": float(np.mean(train_counts_raw)),
            "median": float(np.median(train_counts_raw)),
            "min": int(np.min(train_counts_raw)),
            "max": int(np.max(train_counts_raw)),
        },
        "test_stats": {
            "mean": float(np.mean(test_counts_raw)),
            "median": float(np.median(test_counts_raw)),
            "min": int(np.min(test_counts_raw)),
            "max": int(np.max(test_counts_raw)),
        },
    }
    summary_path = os.path.join(args.output_dir, "authors_distribution_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
