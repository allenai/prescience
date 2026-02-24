"""Plot local clustering coefficient of target papers over time."""
import argparse
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import utils

NATURAL_COLOR = "#0072B2"
SYNTHETIC_COLOR = "#E69F00"


def partition_by_date(papers):
    date2papers = defaultdict(list)
    for paper in papers.values():
        date2papers[paper["date"]].append(paper)
    return dict(sorted(date2papers.items()))


def bucket_daily_to_weekly(partitioned):
    weekly = defaultdict(list)
    for date_str, papers in partitioned.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        bucket_start = datetime.strptime(f"{iso_year}-W{iso_week}-1", "%G-W%V-%u")
        weekly[bucket_start].extend(papers)
    return dict(sorted(weekly.items()))


def target_paper_has_triangular_references(paper, reference_lookup):
    key_refs_field = paper["key_references"] if "key_references" in paper and paper["key_references"] else []
    key_ref_ids = [ref["corpus_id"] for ref in key_refs_field if "corpus_id" in ref and ref["corpus_id"] in reference_lookup]
    if len(set(key_ref_ids)) < 2:
        return False
    key_ref_set = set(key_ref_ids)
    for ref_id in key_ref_set:
        ref_paper = reference_lookup[ref_id]
        nested_list = ref_paper["key_references"] if "key_references" in ref_paper and ref_paper["key_references"] else []
        for nested_ref in nested_list:
            if "corpus_id" not in nested_ref:
                continue
            nested_id = nested_ref["corpus_id"]
            if nested_id in key_ref_set and nested_id != ref_id:
                return True
    return False


def compute_local_clustering_coefficient(partitioned, reference_lookup):
    dates = []
    values = []
    for dt, papers in partitioned.items():
        if not papers:
            continue
        total = len(papers)
        count = sum(1 for paper in papers if target_paper_has_triangular_references(paper, reference_lookup))
        dates.append(dt)
        values.append(count / total)
    return dates, values


def main():
    parser = argparse.ArgumentParser(description="Plot local clustering coefficient of target papers over time.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic experiment directory under data/multiturn/")
    parser.add_argument("--output_path", type=str, default="figures/multiturn/local_clustering.png", help="Path to save the figure")
    args = parser.parse_args()

    utils.log("Loading natural corpus")
    all_papers_nat, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)

    utils.log("Loading synthetic corpus")
    all_papers_syn = utils.load_json(f"data/multiturn/{args.synthetic_dir}/all_papers.json")[0]

    all_papers_nat_dict = {paper["corpus_id"]: paper for paper in all_papers_nat}
    all_papers_syn_dict = {paper["corpus_id"]: paper for paper in all_papers_syn}

    combined = dict(all_papers_nat_dict)
    combined.update(all_papers_syn_dict)

    target_nat = {cid: paper for cid, paper in all_papers_nat_dict.items() if "target" in paper["roles"]}
    target_syn = {cid: paper for cid, paper in all_papers_syn_dict.items() if "synthetic" in paper["roles"]}

    utils.log(f"Computing local clustering for {len(target_nat)} natural and {len(target_syn)} synthetic targets")
    partitioned_nat = bucket_daily_to_weekly(partition_by_date(target_nat))
    partitioned_syn = bucket_daily_to_weekly(partition_by_date(target_syn))

    nat_dates, nat_values = compute_local_clustering_coefficient(partitioned_nat, combined)
    syn_dates, syn_values = compute_local_clustering_coefficient(partitioned_syn, combined)

    utils.log("Creating plot")
    plt.rcParams.update({"font.size": 8})
    plt.figure(figsize=(3.5, 2.8))

    plt.plot(nat_dates, nat_values, label="Natural", color=NATURAL_COLOR, linewidth=1.5)
    plt.plot(syn_dates, syn_values, label="Synthetic", color=SYNTHETIC_COLOR, linewidth=1.5)

    plt.title("Local clustering coefficient", fontsize=9, fontweight="bold")
    plt.xlabel("Week starting", fontsize=8)
    plt.ylabel("Fraction with citation triangles", fontsize=8)
    plt.ylim(0, 1)
    plt.legend(loc="best", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    all_dates = sorted(set(nat_dates + syn_dates))
    if all_dates:
        step = max(len(all_dates) // 14, 1)
        plt.gca().set_xticks(all_dates[::step])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
