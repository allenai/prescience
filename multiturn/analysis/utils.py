"""Shared utilities for multiturn simulation analysis scripts."""
import os
from datetime import datetime

import numpy as np

import utils


def partition_by_date(papers):
    """Partition papers dict by date string."""
    date2papers = {}
    for paper in papers.values():
        date = paper["date"]
        if date not in date2papers:
            date2papers[date] = []
        date2papers[date].append(paper)
    return dict(sorted(date2papers.items()))


def get_bucket_start(date_str, time_bucket):
    """Get the start date of the bucket containing the given date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if time_bucket == 1:
        return date_str
    elif time_bucket == 7:
        iso_year, iso_week, _ = dt.isocalendar()
        bucket_start = datetime.strptime(f"{iso_year}-W{iso_week:02d}-1", "%G-W%V-%u")
        return bucket_start.strftime("%Y-%m-%d")
    elif time_bucket == 14:
        iso_year, iso_week, _ = dt.isocalendar()
        biweek = (iso_week - 1) // 2 * 2 + 1
        bucket_start = datetime.strptime(f"{iso_year}-W{biweek:02d}-1", "%G-W%V-%u")
        return bucket_start.strftime("%Y-%m-%d")
    elif time_bucket == 30:
        return f"{dt.year}-{dt.month:02d}-01"
    else:
        raise ValueError(f"Invalid time_bucket: {time_bucket}")


def partition_by_time_bucket(papers, time_bucket):
    """Partition papers into time buckets (1=daily, 7=weekly, 30=monthly)."""
    bucket2papers = {}
    for paper in papers.values():
        bucket_start = get_bucket_start(paper["date"], time_bucket)
        if bucket_start not in bucket2papers:
            bucket2papers[bucket_start] = []
        bucket2papers[bucket_start].append(paper)
    return dict(sorted(bucket2papers.items()))


def extract_key_embeddings(raw_embeddings):
    """Extract and reshape key embeddings from raw embedding dicts."""
    embeddings_dict = {}
    for cid, emb in raw_embeddings.items():
        embeddings_dict[cid] = np.asarray(emb["key"], dtype=np.float32).reshape(-1)
    return embeddings_dict


def load_natural_corpus_and_embeddings(hf_repo_id, split, embeddings_dir, embedding_type):
    """Load natural corpus from HuggingFace and embeddings from local storage."""
    all_papers, _, _ = utils.load_corpus(hf_repo_id=hf_repo_id, split=split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    embeddings_path = os.path.join(embeddings_dir, f"all_papers.{embedding_type}_embeddings.pkl")
    raw_embeddings, _ = utils.load_pkl(embeddings_path)
    embeddings_dict = extract_key_embeddings(raw_embeddings)
    return all_papers_dict, embeddings_dict


def load_synthetic_corpus_and_embeddings(synthetic_dir, embeddings_dir, embedding_type):
    """Load synthetic corpus and embeddings from local storage."""
    synthetic_path = os.path.join("data/multiturn", synthetic_dir, "all_papers.json")
    all_papers, _ = utils.load_json(synthetic_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    embeddings_path = os.path.join(embeddings_dir, f"all_papers.{embedding_type}_embeddings.pkl")
    raw_embeddings, _ = utils.load_pkl(embeddings_path)
    embeddings_dict = extract_key_embeddings(raw_embeddings)
    return all_papers_dict, embeddings_dict


def filter_by_role(all_papers_dict, role):
    """Filter papers that have the specified role."""
    return {cid: p for cid, p in all_papers_dict.items() if role in p["roles"]}


def sample_matched_pools(papers1, papers2, sample_size, rng):
    """Sample equal-sized pools from two paper lists for fair comparison."""
    matched_size = min(sample_size, len(papers1), len(papers2))
    if matched_size < 2:
        return [], []
    sampled1 = list(papers1)
    sampled2 = list(papers2)
    rng.shuffle(sampled1)
    rng.shuffle(sampled2)
    return sampled1[:matched_size], sampled2[:matched_size]


def bootstrap_statistic(values, n_bootstrap, rng):
    """Compute bootstrapped mean with 95% confidence interval."""
    if len(values) == 0:
        return {"mean": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan")}
    values = np.asarray(values, dtype=np.float32)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_indices = rng.choices(range(len(values)), k=len(values))
        sample = values[sample_indices]
        bootstrap_means.append(float(np.mean(sample)))
    return {
        "mean": float(np.mean(values)),
        "lower_ci": float(np.percentile(bootstrap_means, 2.5)),
        "upper_ci": float(np.percentile(bootstrap_means, 97.5)),
    }


def prepare_time_series(bucket_results):
    """Convert bucket results dict to arrays for plotting."""
    items = sorted(
        (datetime.strptime(bucket, "%Y-%m-%d"), stats)
        for bucket, stats in bucket_results.items()
    )
    dates = [d for d, _ in items]
    means = [s["mean"] for _, s in items]
    lowers = [s["lower_ci"] for _, s in items]
    uppers = [s["upper_ci"] for _, s in items]
    return dates, means, lowers, uppers
