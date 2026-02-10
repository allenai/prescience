"""Distribution of author publication history lengths in target papers."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def get_author_history_lengths(target_papers):
    """Get max publication history length per unique author across all their appearances."""
    author_lengths = {}
    for paper in target_papers:
        for author in paper.get("authors", []):
            author_id = author["author_id"]
            history_len = len(author.get("publication_history", []))
            if author_id not in author_lengths:
                author_lengths[author_id] = history_len
            else:
                author_lengths[author_id] = max(author_lengths[author_id], history_len)
    return author_lengths


def main():
    parser = argparse.ArgumentParser(description="Plot distribution of author publication history lengths")
    parser.add_argument("--train_dir", type=str, default="data/corpus/train", help="Train corpus directory")
    parser.add_argument("--test_dir", type=str, default="data/corpus/test", help="Test corpus directory")
    parser.add_argument("--output_dir", type=str, default="data/corpus/analysis", help="Output directory")
    parser.add_argument("--max_length", type=int, default=20, help="Cap history length for visualization")
    args = parser.parse_args()

    utils.log(f"Loading train corpus from {args.train_dir}")
    train_all_papers, _ = utils.load_json(os.path.join(args.train_dir, "all_papers.json"))
    train_target_papers = [p for p in train_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(train_target_papers)} train target papers")

    utils.log(f"Loading test corpus from {args.test_dir}")
    test_all_papers, _ = utils.load_json(os.path.join(args.test_dir, "all_papers.json"))
    test_target_papers = [p for p in test_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(test_target_papers)} test target papers")

    train_author_lengths = get_author_history_lengths(train_target_papers)
    test_author_lengths = get_author_history_lengths(test_target_papers)
    utils.log(f"Found {len(train_author_lengths)} unique train authors, {len(test_author_lengths)} unique test authors")

    train_lengths = list(train_author_lengths.values())
    test_lengths = list(test_author_lengths.values())
    train_lengths_filtered = [l for l in train_lengths if l <= args.max_length]
    test_lengths_filtered = [l for l in test_lengths if l <= args.max_length]
    utils.log(f"Filtered to {len(train_lengths_filtered)} train authors (<= {args.max_length}), excluded {len(train_lengths) - len(train_lengths_filtered)}")
    utils.log(f"Filtered to {len(test_lengths_filtered)} test authors (<= {args.max_length}), excluded {len(test_lengths) - len(test_lengths_filtered)}")

    bins = range(0, args.max_length + 2)

    utils.log("Creating plot")
    plt.figure(figsize=(4.5, 3.5))
    plt.rcParams.update({"font.size": 8})

    plt.hist(train_lengths_filtered, bins=bins, alpha=0.5, label="Train", color="orange")
    plt.hist(test_lengths_filtered, bins=bins, alpha=0.5, label="Test", color="blue")

    plt.xlabel("Author Publication History Length", fontsize=8)
    plt.ylabel("Number of Authors", fontsize=8)
    plt.title("Distribution of Author History Length in Target Papers", fontsize=9, fontweight="bold")
    plt.xticks(list(bins), fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(fontsize=7, framealpha=0.9)
    plt.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "author_history_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    summary = {
        "train_unique_authors": len(train_author_lengths),
        "test_unique_authors": len(test_author_lengths),
        "max_length_cap": args.max_length,
        "train_stats": {
            "mean": float(np.mean(train_lengths)),
            "median": float(np.median(train_lengths)),
            "min": int(np.min(train_lengths)),
            "max": int(np.max(train_lengths)),
        },
        "test_stats": {
            "mean": float(np.mean(test_lengths)),
            "median": float(np.median(test_lengths)),
            "min": int(np.min(test_lengths)),
            "max": int(np.max(test_lengths)),
        },
    }
    summary_path = os.path.join(args.output_dir, "author_history_distribution_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
