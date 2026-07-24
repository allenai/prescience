"""Compute the topic-growth benchmark definition from LLM-labeled topic data.

Reads `topic_labels.vX.gpt-5.4.full.jsonl` (one record per paper: {corpus_id, topics}),
joins it with paper dates from the HF corpus, and produces `topics.vX.json` with the
same schema consumed by the task_topic_growth_prediction baselines.

Each topic instance contains history size + monthly breakdown + ground-truth growth.
Papers with empty topic lists (the "Other" set) are excluded from the benchmark.
"""

import argparse
import json
import os

import utils
from dataset.corpus.assign_topics import load_topics
from utils import enumerate_months, month_key


def load_topic_labels(path):
    """Return {corpus_id: list[topic_name]} from a labels JSONL file."""
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["corpus_id"]] = r["topics"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build the topic-growth benchmark file from LLM topic labels.")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--labels_path", type=str, required=True, help="JSONL produced by assign_topics.py (one record per paper)")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list_v5.txt", help="Text file with one topic per line (defines benchmark topic set)")
    parser.add_argument("--history_start", type=str, default="2023-10-01", help="History period start (inclusive)")
    parser.add_argument("--history_end", type=str, default="2024-10-01", help="History period end / forecast start (exclusive of history, inclusive of forecast)")
    parser.add_argument("--forecast_end", type=str, default="2025-10-01", help="Forecast period end (exclusive)")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path (e.g. data/task_topic_growth_prediction/topics.v5.json)")
    args = parser.parse_args()

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
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

    per_topic_history_ids = {t: [] for t in topics}
    per_topic_history_monthly = {t: [0] * history_len for t in topics}
    per_topic_gt_growth = {t: 0 for t in topics}
    month_idx = {mk: i for i, mk in enumerate(history_months)}

    n_excluded_empty = n_history = n_forecast = 0
    for cid, paper in papers_by_id.items():
        paper_topics = [t for t in labels.get(cid, []) if t in topic_set]
        if not paper_topics:
            n_excluded_empty += 1
            continue
        date = paper["date"]
        if args.history_start <= date < args.history_end:
            n_history += 1
            mk = month_key(date)
            mi = month_idx[mk]
            for t in paper_topics:
                per_topic_history_ids[t].append(cid)
                per_topic_history_monthly[t][mi] += 1
        elif args.history_end <= date < args.forecast_end:
            n_forecast += 1
            for t in paper_topics:
                per_topic_gt_growth[t] += 1
        # else: outside both windows — ignored

    utils.log(f"Papers in history: {n_history}; in forecast: {n_forecast}; excluded (empty labels): {n_excluded_empty}")

    instances, membership, clusters = [], {}, {}
    for i, topic in enumerate(topics):
        member_ids = per_topic_history_ids[topic]
        instance = {"cluster_id": topic, "history_size": len(member_ids), "history_months": history_months, "history_monthly_counts": per_topic_history_monthly[topic], "gt_growth": per_topic_gt_growth[topic]}
        instances.append((topic, instance))
        membership[topic] = member_ids
        clusters[topic] = {"index": i}

    config = {"source": "llm_topic_labels", "labels_path": args.labels_path, "topics_path": args.topics_path, "history_start": args.history_start, "history_end": args.history_end, "forecast_end": args.forecast_end, "history_months": history_len, "forecast_months": len(enumerate_months(args.history_end, args.forecast_end)), "num_topics": len(topics), "num_excluded_empty": n_excluded_empty, "num_history_papers": n_history, "num_forecast_papers": n_forecast}
    payload = {"config": config, "instances": instances, "membership": membership, "clusters": clusters}

    utils.log(f"Saving topics benchmark to {args.output_path}")
    utils.save_json(payload, args.output_path, metadata=utils.update_metadata([{"source": "huggingface", "repo_id": args.hf_repo_id}], args), overwrite=True)

    sum_history_size = sum(inst["history_size"] for _, inst in instances)
    sum_gt_growth = sum(inst["gt_growth"] for _, inst in instances)
    utils.log(f"sum(history_size) across topics: {sum_history_size} (papers × avg labels = {sum_history_size/max(n_history,1):.2f})")
    utils.log(f"sum(gt_growth)    across topics: {sum_gt_growth} (papers × avg labels = {sum_gt_growth/max(n_forecast,1):.2f})")


if __name__ == "__main__":
    main()
