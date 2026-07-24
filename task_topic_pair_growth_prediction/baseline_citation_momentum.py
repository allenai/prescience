"""Citation-momentum baseline for topic-pair growth prediction (causally safe, inner-split calibrated).

For each paper labeled with both topics in the pair, computes observable citations at the
relevant cutoff (inner_history_end for inner-split rate, history_end for outer prediction).
Scores aggregate per pair via the membership list (paper labeled with both topics).

Calibration: rate = Σ_p N_p(F_inner) / (Σ_p Score_p(H_inner, inner_history_end) × |F_inner|)
Prediction: predicted(p) = Score_p(H, history_end) × rate × |F|
"""

import os
import argparse

import utils
from task_topic_pair_growth_prediction.dataset import load_topics_benchmark


def months_between(date_start, date_end):
    """Return integer month count date_end - date_start for 'YYYY-MM-DD' strings."""
    ys, ms = int(date_start[:4]), int(date_start[5:7])
    ye, me = int(date_end[:4]), int(date_end[5:7])
    return (ye - ys) * 12 + (me - ms)


def observable_citations(paper, cutoff_date):
    """Cumulative citations of `paper` observable as of cutoff_date (via citation_trajectory)."""
    traj = paper.get("citation_trajectory") or []
    if not traj:
        return 0
    age = months_between(paper["date"], cutoff_date)
    if age < 1:
        return 0
    return int(traj[min(age, len(traj)) - 1])


def compute_pair_scores(membership, papers_by_id, cutoff_date, period_start, period_end):
    """Sum observable citations (at cutoff_date) across papers in [period_start, period_end) per pair."""
    scores = {p: 0 for p in membership}
    for pair_id, corpus_ids in membership.items():
        for cid in corpus_ids:
            p = papers_by_id.get(cid)
            if p is None:
                continue
            if not (period_start <= p["date"] < period_end):
                continue
            scores[pair_id] += observable_citations(p, cutoff_date)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Citation-momentum baseline (inner-split calibrated)")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topic_pairs.vX.json")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--inner_history_months", type=int, default=6, help="First N months of history used as inner-history")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_pair_growth_prediction/predictions", help="Output directory")
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
    utils.log(f"inner_history_end = {inner_history_end}  (N={N}, |F_inner|={F_inner})")

    utils.log(f"Loading corpus from {args.hf_repo_id}")
    train_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="train", embedding_type=None, load_sd2publications=False)
    test_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="test", embedding_type=None, load_sd2publications=False)
    papers_by_id = {}
    for p in train_papers + test_papers:
        if p["corpus_id"] not in papers_by_id:
            papers_by_id[p["corpus_id"]] = p
    utils.log(f"Loaded {len(papers_by_id)} unique papers")

    inner_scores = compute_pair_scores(membership, papers_by_id, inner_history_end, history_start, inner_history_end)
    full_scores = compute_pair_scores(membership, papers_by_id, history_end, history_start, history_end)

    inner_forecast_total = sum(sum(inst["history_monthly_counts"][N:H]) for _, inst in instances)
    sum_inner_scores = sum(inner_scores.values())
    rate = (inner_forecast_total / (sum_inner_scores * F_inner)) if (sum_inner_scores > 0 and F_inner > 0) else 0.0
    utils.log(f"Σ N_p(F_inner) = {inner_forecast_total}; Σ Score_p(H_inner) = {sum_inner_scores}; rate = {rate:.6g}")

    predictions = []
    for cid, inst in instances:
        score = full_scores.get(cid, 0)
        predicted = score * rate * F
        predictions.append({"cluster_id": cid, "predicted_growth": float(predicted), "gt_growth": inst["gt_growth"]})

    output_path = os.path.join(args.output_dir, "predictions.citation_momentum.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
