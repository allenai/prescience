"""Author-momentum baseline for topic growth prediction (causally safe, inner-split calibrated).

For each paper in the history period, sums a per-author metric (num_papers or h_index at paper date)
across its authors. Per-topic score = sum across labeled papers.

Calibration: rate = Σ_t N_t(F_inner) / (Σ_t Score_t(H_inner) × |F_inner|)
Prediction: predicted(t) = Score_t(H) × rate × |F|
"""

import os
import argparse

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark


def author_score(paper, metric):
    """Sum the chosen author metric across paper.authors. Supported: num_papers, h_index, num_citations."""
    total = 0
    for author in paper.get("authors", []) or []:
        total += int(author.get(metric, 0) or 0)
    return total


def compute_topic_scores(membership, papers_by_id, period_start, period_end, metric):
    """Sum author scores across papers in [period_start, period_end) per topic."""
    scores = {t: 0 for t in membership}
    for topic, corpus_ids in membership.items():
        for cid in corpus_ids:
            p = papers_by_id.get(cid)
            if p is None:
                continue
            if not (period_start <= p["date"] < period_end):
                continue
            scores[topic] += author_score(p, metric)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Author-momentum baseline (inner-split calibrated)")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--inner_history_months", type=int, default=6, help="First N months of history used as inner-history")
    parser.add_argument("--author_metric", type=str, default="num_papers", choices=["num_papers", "h_index", "num_citations"], help="Per-author metric to sum")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, membership, _, metadata = load_topics_benchmark(args.topics_path)
    H, F, N = config["history_months"], config["forecast_months"], args.inner_history_months
    assert 1 <= N < H, f"--inner_history_months must be in [1, {H-1}]"
    F_inner = H - N
    history_start, history_end = config["history_start"], config["history_end"]

    ys, ms = int(history_start[:4]), int(history_start[5:7])
    inner_total = ms - 1 + N
    inner_history_end = f"{ys + inner_total // 12:04d}-{inner_total % 12 + 1:02d}-01"
    utils.log(f"inner_history_end = {inner_history_end}  (N={N}, |F_inner|={F_inner}); metric={args.author_metric}")

    utils.log(f"Loading corpus from {args.hf_repo_id}")
    train_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="train", embedding_type=None, load_sd2publications=False)
    test_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="test", embedding_type=None, load_sd2publications=False)
    papers_by_id = {}
    for p in train_papers + test_papers:
        if p["corpus_id"] not in papers_by_id:
            papers_by_id[p["corpus_id"]] = p
    utils.log(f"Loaded {len(papers_by_id)} unique papers")

    inner_scores = compute_topic_scores(membership, papers_by_id, history_start, inner_history_end, args.author_metric)
    full_scores = compute_topic_scores(membership, papers_by_id, history_start, history_end, args.author_metric)

    inner_forecast_total = sum(sum(inst["history_monthly_counts"][N:H]) for _, inst in instances)
    sum_inner_scores = sum(inner_scores.values())
    rate = (inner_forecast_total / (sum_inner_scores * F_inner)) if (sum_inner_scores > 0 and F_inner > 0) else 0.0
    utils.log(f"Σ N_t(F_inner) = {inner_forecast_total}; Σ Score_t(H_inner) = {sum_inner_scores}; rate = {rate:.6g}")

    predictions = []
    for cid, inst in instances:
        score = full_scores.get(cid, 0)
        predicted = score * rate * F
        predictions.append({"cluster_id": cid, "predicted_growth": float(predicted), "gt_growth": inst["gt_growth"]})

    output_path = os.path.join(args.output_dir, f"predictions.author_momentum_{args.author_metric}.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
