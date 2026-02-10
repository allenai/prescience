"""Compute k nearest neighbors within time buckets for diversity analysis."""
import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import utils
from multiturn.analysis.utils import load_natural_corpus_and_embeddings, load_synthetic_corpus_and_embeddings, filter_by_role, partition_by_time_bucket


def compute_bucket_neighbors(query_papers, pool_papers, embeddings_dict, k):
    """Compute k nearest neighbors for query papers within a pool.

    Args:
        query_papers: Papers to query (list of dicts with corpus_id)
        pool_papers: Papers to build index from (list of dicts with corpus_id)
        embeddings_dict: Dict of corpus_id -> embedding
        k: Number of neighbors to retrieve

    Returns list of dicts with corpus_id and neighbors (id + distance).
    """
    pool_ids = [p["corpus_id"] for p in pool_papers]
    pool_embeddings = {cid: embeddings_dict[cid] for cid in pool_ids if cid in embeddings_dict}

    if len(pool_embeddings) < k + 1:
        raise ValueError(f"Pool has only {len(pool_embeddings)} papers with embeddings, need at least {k + 1}")

    index = utils.create_index(pool_embeddings, "cosine")

    query_ids = [p["corpus_id"] for p in query_papers if p["corpus_id"] in embeddings_dict]
    query_vecs = [embeddings_dict[cid] for cid in query_ids]
    retrieved_ids, similarities = utils.query_index(index, query_vecs, k + 1)

    results = []
    for qid, neighbor_ids, sims in zip(query_ids, retrieved_ids, similarities):
        neighbors = []
        for nid, sim in zip(neighbor_ids, sims):
            if nid != qid:
                neighbors.append({"corpus_id": nid, "distance": float(1.0 - sim)})
            if len(neighbors) >= k:
                break
        results.append({"corpus_id": qid, "neighbors": neighbors})

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute k-NN within time buckets for diversity analysis.")
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.compute_on == "synthetic":
        utils.log(f"Loading synthetic corpus from data/multiturn/{args.synthetic_dir}")
        all_papers_dict, embeddings_dict = load_synthetic_corpus_and_embeddings(args.synthetic_dir, args.embeddings_dir, args.embedding_type)
    else:
        utils.log(f"Loading natural corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
        all_papers_dict, embeddings_dict = load_natural_corpus_and_embeddings(args.hf_repo_id, args.split, args.embeddings_dir, args.embedding_type)

    utils.log(f"Filtering papers by role: {args.role}")
    filtered_papers = filter_by_role(all_papers_dict, args.role)
    utils.log(f"Found {len(filtered_papers)} papers with role '{args.role}'")

    ref_bucket_sizes = None
    if args.compute_on == "natural":
        utils.log(f"Loading synthetic reference corpus from data/multiturn/{args.synthetic_dir}")
        synthetic_path = os.path.join("data/multiturn", args.synthetic_dir, "all_papers.json")
        ref_papers, _ = utils.load_json(synthetic_path)
        ref_papers_dict = {p["corpus_id"]: p for p in ref_papers}
        ref_targets = filter_by_role(ref_papers_dict, "synthetic")
        ref_buckets = partition_by_time_bucket(ref_targets, args.time_bucket)
        ref_bucket_sizes = {b: len(papers) for b, papers in ref_buckets.items()}
        utils.log(f"Reference buckets: {len(ref_buckets)}, size range: {min(ref_bucket_sizes.values())}-{max(ref_bucket_sizes.values())}")

    utils.log(f"Partitioning into {args.time_bucket}-day buckets")
    buckets = partition_by_time_bucket(filtered_papers, args.time_bucket)
    utils.log(f"Found {len(buckets)} buckets")

    all_results = []
    for bucket_key in tqdm(sorted(buckets.keys()), desc="Processing buckets"):
        bucket_papers = buckets[bucket_key]

        if ref_bucket_sizes and bucket_key in ref_bucket_sizes:
            target_size = ref_bucket_sizes[bucket_key]
            if len(bucket_papers) > target_size:
                pool_papers = random.sample(bucket_papers, target_size)
            else:
                pool_papers = bucket_papers
        else:
            pool_papers = bucket_papers

        if len(pool_papers) < args.n:
            raise ValueError(f"Bucket {bucket_key} pool has only {len(pool_papers)} papers, need {args.n}")

        sampled_papers = random.sample(pool_papers, args.n)
        bucket_results = compute_bucket_neighbors(sampled_papers, pool_papers, embeddings_dict, args.k)

        for result in bucket_results:
            result["bucket"] = bucket_key
            all_results.append(result)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    utils.save_json(all_results, args.output_path, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(all_results)} paper results to {args.output_path}")


if __name__ == "__main__":
    main()
