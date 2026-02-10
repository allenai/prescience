"""Compute cumulative key_references diversity over time for a single corpus."""
import os
import math
import random
import argparse

from tqdm import tqdm

import utils
from multiturn.analysis.utils import filter_by_role, partition_by_time_bucket

random.seed(42)


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


def main():
    parser = argparse.ArgumentParser(description="Compute cumulative key_references diversity over time.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic corpus directory under data/multiturn/")
    parser.add_argument("--compute_on", type=str, required=True, choices=["natural", "synthetic"], help="Which corpus to compute diversity on")
    parser.add_argument("--role", type=str, required=True, help="Role to filter papers by (e.g., target, synthetic)")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--time_bucket", type=int, default=30, help="Bucket size in days (default: 30 for monthly)")
    args = parser.parse_args()

    if args.compute_on == "synthetic":
        utils.log(f"Loading synthetic corpus from data/multiturn/{args.synthetic_dir}")
        synthetic_path = os.path.join("data/multiturn", args.synthetic_dir, "all_papers.json")
        all_papers, _ = utils.load_json(synthetic_path)
        all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    else:
        utils.log(f"Loading natural corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
        all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
        all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers")

    utils.log(f"Filtering papers by role: {args.role}")
    target_papers_dict = filter_by_role(all_papers_dict, args.role)
    utils.log(f"Found {len(target_papers_dict)} papers with role '{args.role}'")

    ref_counts_per_bucket = None
    if args.compute_on == "natural":
        utils.log(f"Loading synthetic reference corpus from data/multiturn/{args.synthetic_dir}")
        synthetic_path = os.path.join("data/multiturn", args.synthetic_dir, "all_papers.json")
        ref_papers, _ = utils.load_json(synthetic_path)
        ref_papers_dict = {p["corpus_id"]: p for p in ref_papers}
        ref_targets = filter_by_role(ref_papers_dict, "synthetic")
        ref_buckets = partition_by_time_bucket(ref_targets, args.time_bucket)
        ref_counts_per_bucket = {b: len(papers) for b, papers in ref_buckets.items()}
        utils.log(f"Reference has {len(ref_buckets)} buckets with counts: {list(ref_counts_per_bucket.values())}")

    # Partition into time buckets
    target_buckets = partition_by_time_bucket(target_papers_dict, args.time_bucket)
    utils.log(f"Partitioned into {len(target_buckets)} buckets")

    # Process each bucket
    utils.log("Computing cumulative diversity metrics")
    cumulative_counts = {}
    total_papers = 0
    results = []

    for bucket in tqdm(sorted(target_buckets.keys()), desc="Processing buckets"):
        bucket_papers = target_buckets[bucket]

        # Subsample to match reference bucket sizes
        if ref_counts_per_bucket and bucket in ref_counts_per_bucket:
            target_count = ref_counts_per_bucket[bucket]
            if len(bucket_papers) > target_count:
                bucket_papers = random.sample(bucket_papers, target_count)

        # Update cumulative key_references counts
        for paper in bucket_papers:
            for ref in paper.get("key_references", []):
                ref_id = ref["corpus_id"]
                if ref_id not in cumulative_counts:
                    cumulative_counts[ref_id] = 0
                cumulative_counts[ref_id] += 1
            total_papers += 1

        unique_count = len(cumulative_counts)
        effective_count = math.exp(compute_entropy(cumulative_counts))
        total_citations = sum(cumulative_counts.values())

        results.append({
            "bucket": bucket,
            "unique_count": unique_count,
            "effective_count": effective_count,
            "total_citations": total_citations,
            "num_papers": total_papers,
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    utils.save_json(results, args.output_path, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(results)} records to {args.output_path}")


if __name__ == "__main__":
    main()
