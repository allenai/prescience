"""Compute the topic-pair-growth benchmark definition from LLM-labeled topic data.

Reads `topic_labels.vX.gpt-5.4.full.jsonl` (one record per paper: {corpus_id, topics}),
joins with paper dates from the HF corpus, enumerates all unordered topic pairs that
co-occur on at least one labeled paper, and produces `topic_pairs.vX.json` with a
schema interoperable with task_topic_growth_prediction baselines.

Each pair instance contains history size + monthly breakdown + ground-truth growth.
Pairs with fewer than --min_count_filter occurrences in either period are dropped.
"""

import argparse
import json
import os

import utils
from dataset.corpus.assign_topics import load_topics
from utils import enumerate_months, month_key


SEP = "||"


def make_pair_id(t_a, t_b):
    """Canonical lex-sorted pair id."""
    a, b = sorted([t_a, t_b])
    return f"{a}{SEP}{b}"


def load_topic_labels(path):
    """Return {corpus_id: list[topic_name]} from a labels JSONL file."""
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["corpus_id"]] = r["topics"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build the topic-pair-growth benchmark file from LLM topic labels.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--labels_path", type=str, required=True, help="JSONL produced by assign_topics.py (one record per paper)")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list_v5.txt", help="Text file with one topic per line (defines benchmark topic set)")
    parser.add_argument("--history_start", type=str, default="2023-10-01", help="History period start (inclusive)")
    parser.add_argument("--history_end", type=str, default="2024-10-01", help="History period end / forecast start")
    parser.add_argument("--forecast_end", type=str, default="2025-07-01", help="Forecast period end (exclusive)")
    parser.add_argument("--min_count_filter", type=int, default=10, help="Keep pair iff history_count >= K OR gt_growth >= K")
    parser.add_argument("--paper_subset_path", type=str, default=None, help="Optional JSONL of {corpus_id: ...} records; if given, restrict the benchmark to these corpus_ids (used for sub-sampled cross-labeler comparison)")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
    for t in topics:
        assert SEP not in t, f"Pair-id separator '{SEP}' collides with topic name: {t}"
    utils.log(f"Loaded {len(topics)} topics from {args.topics_path}")

    utils.log(f"Loading topic labels from {args.labels_path}")
    labels = load_topic_labels(args.labels_path)
    utils.log(f"Loaded labels for {len(labels)} papers")

    utils.log(f"Loading corpus from {args.hf_repo_id}")
    train_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="train", embedding_type=None, load_sd2publications=False)
    test_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="test", embedding_type=None, load_sd2publications=False)
    papers_by_id = {}
    for p in train_papers + test_papers:
        if p.get("corpus_id") in labels and p["corpus_id"] not in papers_by_id and "target" in (p.get("roles") or []):
            papers_by_id[p["corpus_id"]] = p
    utils.log(f"Resolved {len(papers_by_id)} target papers with labels")

    history_months = enumerate_months(args.history_start, args.history_end)
    history_len = len(history_months)
    forecast_len = len(enumerate_months(args.history_end, args.forecast_end))
    month_idx = {mk: i for i, mk in enumerate(history_months)}

    pair_history_ids = {}
    pair_history_monthly = {}
    pair_gt_growth = {}

    n_excluded_empty = n_history_papers = n_forecast_papers = n_singleton = 0
    for cid, paper in papers_by_id.items():
        paper_topics_set = set(t for t in labels.get(cid, []) if t in topic_set)
        if not paper_topics_set:
            n_excluded_empty += 1
            continue
        date = paper["date"]
        in_history = args.history_start <= date < args.history_end
        in_forecast = args.history_end <= date < args.forecast_end
        if not (in_history or in_forecast):
            continue
        if in_history:
            n_history_papers += 1
        else:
            n_forecast_papers += 1
        if len(paper_topics_set) < 2:
            n_singleton += 1
            continue
        paper_topics_list = sorted(paper_topics_set)
        mi = month_idx[month_key(date)] if in_history else None
        for i in range(len(paper_topics_list)):
            for j in range(i + 1, len(paper_topics_list)):
                pid = make_pair_id(paper_topics_list[i], paper_topics_list[j])
                if in_history:
                    pair_history_ids.setdefault(pid, []).append(cid)
                    monthly = pair_history_monthly.setdefault(pid, [0] * history_len)
                    monthly[mi] += 1
                else:
                    pair_gt_growth[pid] = pair_gt_growth.get(pid, 0) + 1

    utils.log(f"Papers in history: {n_history_papers}; in forecast: {n_forecast_papers}; excluded (empty labels): {n_excluded_empty}; singleton-label papers (contribute no pair): {n_singleton}")

    all_pairs = set(pair_history_ids.keys()) | set(pair_gt_growth.keys())
    utils.log(f"Total raw co-occurring pairs: {len(all_pairs)}")

    K = args.min_count_filter
    filtered_pairs = sorted(p for p in all_pairs if len(pair_history_ids.get(p, [])) >= K or pair_gt_growth.get(p, 0) >= K)
    utils.log(f"Pairs after min_count_filter (>= {K} in either period): {len(filtered_pairs)}")

    instances, membership, clusters = [], {}, {}
    for i, pid in enumerate(filtered_pairs):
        topic_a, topic_b = pid.split(SEP)
        member_ids = pair_history_ids.get(pid, [])
        monthly = pair_history_monthly.get(pid, [0] * history_len)
        instance = {"cluster_id": pid, "topic_a": topic_a, "topic_b": topic_b, "history_size": len(member_ids), "history_months": history_months, "history_monthly_counts": monthly, "gt_growth": pair_gt_growth.get(pid, 0)}
        instances.append((pid, instance))
        membership[pid] = member_ids
        clusters[pid] = {"index": i, "topic_a": topic_a, "topic_b": topic_b}

    config = {"source": "llm_topic_labels_pairs", "labels_path": args.labels_path, "topics_path": args.topics_path, "history_start": args.history_start, "history_end": args.history_end, "forecast_end": args.forecast_end, "history_months": history_len, "forecast_months": forecast_len, "min_count_filter": K, "num_topics": len(topics), "num_pairs": len(filtered_pairs), "num_excluded_empty": n_excluded_empty, "num_history_papers": n_history_papers, "num_forecast_papers": n_forecast_papers}
    payload = {"config": config, "instances": instances, "membership": membership, "clusters": clusters}

    utils.log(f"Saving topic-pair benchmark to {args.output_path}")
    utils.save_json(payload, args.output_path, metadata=utils.update_metadata([{"source": "huggingface", "repo_id": args.hf_repo_id}], args), overwrite=True)

    sum_history_size = sum(inst["history_size"] for _, inst in instances)
    sum_gt_growth = sum(inst["gt_growth"] for _, inst in instances)
    utils.log(f"sum(history_size) across pairs: {sum_history_size}")
    utils.log(f"sum(gt_growth)    across pairs: {sum_gt_growth}")


if __name__ == "__main__":
    main()
