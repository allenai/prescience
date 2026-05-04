"""Evaluation script for co-citation prediction task."""

import os
import argparse
import numpy as np
from tqdm import tqdm

import utils


NDCG_CUTOFFS = [10, 100, 1000]


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Compute nDCG@k with binary relevance. Handles |relevant| > k gracefully (unlike utils.calculate_ndcg)."""
    if len(retrieved_ids) == 0 or len(relevant_ids) == 0:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_topk = retrieved_ids[:k]
    dcg = 0.0
    for i, rid in enumerate(retrieved_topk):
        if rid in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = 0.0
    for i in range(min(k, len(relevant_set))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_predictions(predictions):
    """Evaluate predictions and compute metrics for each instance."""
    results = []
    for pred in tqdm(predictions, desc="Evaluating predictions"):
        corpus_id = pred["corpus_id"]
        gt_cocited_ids = pred["gt_cocited_ids"]
        predicted_cocited_ids = pred["predicted_cocited_ids"]

        ndcgs = {f"ndcg@{k}": ndcg_at_k(predicted_cocited_ids, gt_cocited_ids, k) for k in NDCG_CUTOFFS}
        k_eval = min(len(gt_cocited_ids), len(predicted_cocited_ids))
        top_k_predictions = predicted_cocited_ids[:k_eval]
        precision, recall, f1 = utils.calculate_precision_recall_f1(top_k_predictions, gt_cocited_ids)

        results.append({
            "corpus_id": corpus_id,
            **ndcgs,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_gt_cocited": len(gt_cocited_ids),
            "gt_truncated_at_1000": len(gt_cocited_ids) > 1000,
        })

    return results


def compute_aggregate_metrics(results):
    """Compute aggregate metrics across all evaluation instances."""
    n = len(results)
    aggregates = {"num_instances": n}
    for k in NDCG_CUTOFFS:
        aggregates[f"avg_ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in results) / n
    aggregates["avg_precision"] = sum(r["precision"] for r in results) / n
    aggregates["avg_recall"] = sum(r["recall"] for r in results) / n
    aggregates["avg_f1"] = sum(r["f1"] for r in results) / n
    aggregates["num_gt_truncated_at_1000"] = sum(1 for r in results if r["gt_truncated_at_1000"])
    return aggregates


def main():
    parser = argparse.ArgumentParser(description="Evaluate co-citation predictions")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--output_dir", type=str, default="data/task_cocitation_prediction/test/scored")
    args = parser.parse_args()

    utils.log(f"Loading predictions from {args.predictions_path}")
    predictions, predictions_metadata = utils.load_json(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions")

    utils.log("Evaluating predictions")
    results = evaluate_predictions(predictions)

    aggregates = compute_aggregate_metrics(results)
    utils.log(f"Evaluation complete:")
    utils.log(f"  Instances: {aggregates['num_instances']}")
    for k in NDCG_CUTOFFS:
        utils.log(f"  Avg nDCG@{k}: {aggregates[f'avg_ndcg@{k}']:.4f}")
    utils.log(f"  Avg Precision: {aggregates['avg_precision']:.4f}")
    utils.log(f"  Avg Recall: {aggregates['avg_recall']:.4f}")
    utils.log(f"  Avg F1: {aggregates['avg_f1']:.4f}")
    utils.log(f"  Instances with |GT|>1000: {aggregates['num_gt_truncated_at_1000']}")

    output = {"aggregates": aggregates, "per_instance": results}

    predictions_filename = os.path.basename(args.predictions_path)
    base, ext = os.path.splitext(predictions_filename)
    output_filename = f"{base}.eval{ext}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving evaluation results to {output_path}")
    utils.save_json(output, output_path, metadata=utils.update_metadata(predictions_metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
