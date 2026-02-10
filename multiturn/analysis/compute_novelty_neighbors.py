"""Compute k nearest neighbors in prior corpus for novelty analysis."""
import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import utils
from multiturn.analysis.utils import load_natural_corpus_and_embeddings, load_synthetic_corpus_and_embeddings, filter_by_role, partition_by_time_bucket, partition_by_date


def main():
    parser = argparse.ArgumentParser(description="Compute k-NN in prior corpus for novelty analysis.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--synthetic_dir", type=str, default="simulated", help="Synthetic corpus directory under data/multiturn/")
    parser.add_argument("--compute_on", type=str, required=True, choices=["natural", "synthetic"], help="Which corpus to compute neighbors on")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory with embedding files")
    parser.add_argument("--role", type=str, required=True, help="Role to filter papers by (e.g., target, synthetic)")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["gtr", "grit", "specter2"], help="Embedding type")
    parser.add_argument("--time_bucket", type=int, default=14, choices=[1, 7, 14, 30], help="Time bucket size in days")
    parser.add_argument("--n", type=int, default=50, help="Papers to sample per bucket")
    parser.add_argument("--k", type=int, default=5, help="Nearest neighbors per paper")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fixed_prior", action="store_true", help="Keep prior corpus fixed to pre-simulation papers only (do not add new papers as simulation progresses)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.compute_on == "synthetic":
        utils.log(f"Loading synthetic corpus from data/multiturn/{args.synthetic_dir}")
        all_papers_dict, embeddings_dict = load_synthetic_corpus_and_embeddings(args.synthetic_dir, args.embeddings_dir, args.embedding_type)
    else:
        utils.log(f"Loading natural corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
        all_papers_dict, embeddings_dict = load_natural_corpus_and_embeddings(args.hf_repo_id, args.split, args.embeddings_dir, args.embedding_type)

    utils.log(f"Filtering target papers by role: {args.role}")
    target_papers = filter_by_role(all_papers_dict, args.role)
    utils.log(f"Found {len(target_papers)} papers with role '{args.role}'")

    ref_prior_size = None
    ref_growth_per_bucket = None
    if args.compute_on == "natural":
        utils.log(f"Loading synthetic reference corpus from data/multiturn/{args.synthetic_dir}")
        synthetic_path = os.path.join("data/multiturn", args.synthetic_dir, "all_papers.json")
        ref_papers, _ = utils.load_json(synthetic_path)
        ref_papers_dict = {p["corpus_id"]: p for p in ref_papers}
        ref_targets = filter_by_role(ref_papers_dict, "synthetic")
        ref_buckets = partition_by_time_bucket(ref_targets, args.time_bucket)
        ref_first_bucket = min(ref_buckets.keys())
        ref_prior_size = sum(1 for p in ref_papers_dict.values() if p["date"] < ref_first_bucket)
        ref_growth_per_bucket = {b: len(papers) for b, papers in ref_buckets.items()}
        utils.log(f"Reference prior size: {ref_prior_size}")
        utils.log(f"Reference buckets: {len(ref_buckets)}, growth range: {min(ref_growth_per_bucket.values())}-{max(ref_growth_per_bucket.values())}")

    utils.log(f"Partitioning target papers into {args.time_bucket}-day buckets")
    buckets = partition_by_time_bucket(target_papers, args.time_bucket)
    utils.log(f"Found {len(buckets)} buckets")

    for bucket_key, bucket_papers in buckets.items():
        if len(bucket_papers) < args.n:
            raise ValueError(f"Bucket {bucket_key} has only {len(bucket_papers)} papers, need {args.n}")

    utils.log("Partitioning all papers by date for temporal index building")
    all_papers_by_date = partition_by_date(all_papers_dict)

    first_bucket = min(buckets.keys())
    prior_dates = [d for d in sorted(all_papers_by_date.keys()) if d < first_bucket]
    utils.log(f"Building initial prior index from {len(prior_dates)} dates before {first_bucket}")

    prior_embeddings = {}
    for date_key in tqdm(prior_dates, desc="Collecting prior embeddings"):
        for paper in all_papers_by_date[date_key]:
            cid = paper["corpus_id"]
            if cid in embeddings_dict:
                prior_embeddings[cid] = embeddings_dict[cid]

    utils.log(f"Initial prior corpus has {len(prior_embeddings)} papers")

    if ref_prior_size and len(prior_embeddings) > ref_prior_size:
        sampled_ids = random.sample(list(prior_embeddings.keys()), ref_prior_size)
        prior_embeddings = {cid: prior_embeddings[cid] for cid in sampled_ids}
        utils.log(f"Subsampled initial prior to {ref_prior_size} papers (matching reference)")

    if len(prior_embeddings) < args.k:
        raise ValueError(f"Prior corpus has only {len(prior_embeddings)} papers, need at least {args.k}")

    utils.log("Building FAISS index for prior corpus...")
    prior_index = utils.create_index(prior_embeddings, "cosine")
    utils.log("Index built")

    if args.fixed_prior:
        utils.log("Using fixed prior corpus (will NOT add papers as simulation progresses)")

    all_results = []
    sorted_buckets = sorted(buckets.keys())

    for bucket_key in tqdm(sorted_buckets, desc="Processing buckets"):
        bucket_papers = buckets[bucket_key]
        sampled_papers = random.sample(bucket_papers, args.n)

        query_ids = [p["corpus_id"] for p in sampled_papers if p["corpus_id"] in embeddings_dict]
        query_vecs = [embeddings_dict[cid] for cid in query_ids]

        if len(query_vecs) == 0:
            utils.log(f"Warning: No embeddings for sampled papers in bucket {bucket_key}")
            continue

        retrieved_ids, similarities = utils.query_index(prior_index, query_vecs, args.k)

        for qid, neighbor_ids, sims in zip(query_ids, retrieved_ids, similarities):
            neighbors = []
            for nid, sim in zip(neighbor_ids, sims):
                neighbors.append({"corpus_id": nid, "distance": float(1.0 - sim)})
            all_results.append({"corpus_id": qid, "bucket": bucket_key, "neighbors": neighbors})

        if not args.fixed_prior:
            papers_to_add = []
            for date_key in sorted(all_papers_by_date.keys()):
                if date_key < bucket_key:
                    continue
                next_bucket_idx = sorted_buckets.index(bucket_key) + 1
                if next_bucket_idx < len(sorted_buckets) and date_key >= sorted_buckets[next_bucket_idx]:
                    break
                for paper in all_papers_by_date[date_key]:
                    cid = paper["corpus_id"]
                    if cid in embeddings_dict and cid not in prior_embeddings:
                        papers_to_add.append(cid)

            if ref_growth_per_bucket and bucket_key in ref_growth_per_bucket:
                target_growth = ref_growth_per_bucket[bucket_key]
                if len(papers_to_add) > target_growth:
                    papers_to_add = random.sample(papers_to_add, target_growth)

            for cid in papers_to_add:
                prior_embeddings[cid] = embeddings_dict[cid]
                prior_index = utils.add_vector_to_index(prior_index, cid, embeddings_dict[cid])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    utils.save_json(all_results, args.output_path, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(all_results)} paper results to {args.output_path}")


if __name__ == "__main__":
    main()
