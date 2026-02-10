"""Characterize over and underpredictions in impact prediction."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def compute_bucket_error_rates(instances, papers_dict, bucket_fn, bucket_labels):
    """Compute over/underprediction rates per bucket."""
    buckets = {label: {"over": 0, "under": 0, "total": 0} for label in bucket_labels}

    for inst in instances:
        corpus_id = inst["corpus_id"]
        if corpus_id not in papers_dict:
            continue
        paper = papers_dict[corpus_id]
        label = bucket_fn(paper, inst)
        if label is None:
            continue

        buckets[label]["total"] += 1
        if inst["predicted"] > inst["gt"]:
            buckets[label]["over"] += 1
        elif inst["predicted"] < inst["gt"]:
            buckets[label]["under"] += 1

    return buckets


def main():
    parser = argparse.ArgumentParser(description="Characterize over/under predictions in impact prediction")
    parser.add_argument("--eval_path", type=str, default="data/task_impact_prediction/test/scored/predictions.xgboost_regressor_grit_author_numbers_author_papers_prior_work_papers_prior_work_numbers_followup_work_paper.eval.json", help="Path to evaluation results")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading evaluation results from {args.eval_path}")
    eval_file, _ = utils.load_json(args.eval_path)
    data = eval_file["data"] if "data" in eval_file else eval_file
    instances = data["per_instance"]
    utils.log(f"Loaded {len(instances)} instances")

    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers")

    # Define bucket functions
    def gt_citations_bucket(paper, inst):
        gt = inst["gt"]
        if gt == 0:
            return "0"
        elif gt <= 2:
            return "1-2"
        elif gt <= 5:
            return "3-5"
        elif gt <= 10:
            return "6-10"
        elif gt <= 25:
            return "11-25"
        else:
            return "26+"

    def num_authors_bucket(paper, inst):
        n = len(paper.get("authors", []))
        if n <= 2:
            return "1-2"
        elif n <= 4:
            return "3-4"
        elif n <= 6:
            return "5-6"
        else:
            return "7+"

    def max_hindex_bucket(paper, inst):
        authors = paper.get("authors", [])
        if not authors:
            return None
        max_h = max(a.get("h_index", 0) for a in authors)
        if max_h <= 5:
            return "0-5"
        elif max_h <= 15:
            return "6-15"
        elif max_h <= 30:
            return "16-30"
        else:
            return "31+"

    def num_refs_bucket(paper, inst):
        n = len(paper.get("key_references", []))
        if n == 0:
            return "0"
        elif n <= 2:
            return "1-2"
        elif n <= 5:
            return "3-5"
        else:
            return "6+"

    # Compute bucket stats for each characteristic
    analyses = [
        ("Actual Citations", gt_citations_bucket, ["0", "1-2", "3-5", "6-10", "11-25", "26+"]),
        ("# Authors", num_authors_bucket, ["1-2", "3-4", "5-6", "7+"]),
        ("Max Author H-Index", max_hindex_bucket, ["0-5", "6-15", "16-30", "31+"]),
        ("# Key References", num_refs_bucket, ["0", "1-2", "3-5", "6+"]),
    ]

    results = {}
    for name, bucket_fn, labels in analyses:
        utils.log(f"\nAnalyzing: {name}")
        buckets = compute_bucket_error_rates(instances, papers_dict, bucket_fn, labels)
        results[name] = {"labels": labels, "buckets": buckets}

        for label in labels:
            b = buckets[label]
            if b["total"] > 0:
                over_pct = 100 * b["over"] / b["total"]
                under_pct = 100 * b["under"] / b["total"]
                utils.log(f"  {label}: n={b['total']}, over={over_pct:.1f}%, under={under_pct:.1f}%")

    # Create multi-panel plot
    utils.log("\nCreating plot")
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    plt.rcParams.update({"font.size": 8})

    for idx, (name, bucket_fn, labels) in enumerate(analyses):
        ax = axes[idx // 2, idx % 2]
        buckets = results[name]["buckets"]

        x = np.arange(len(labels))
        over_rates = [100 * buckets[l]["over"] / buckets[l]["total"] if buckets[l]["total"] > 0 else 0 for l in labels]
        under_rates = [100 * buckets[l]["under"] / buckets[l]["total"] if buckets[l]["total"] > 0 else 0 for l in labels]

        width = 0.35
        ax.bar(x - width/2, over_rates, width, label="Over", color="#D55E00", alpha=0.8)
        ax.bar(x + width/2, under_rates, width, label="Under", color="#0072B2", alpha=0.8)

        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("% of Predictions", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, axis="y")

    plt.suptitle("Prediction Error Characterization", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prediction_error_characterization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    # Save detailed results
    output_json = {}
    for name, bucket_fn, labels in analyses:
        output_json[name] = {l: results[name]["buckets"][l] for l in labels}
    summary_path = os.path.join(args.output_dir, "prediction_error_characterization.json")
    utils.save_json(output_json, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved results to {summary_path}")


if __name__ == "__main__":
    main()
