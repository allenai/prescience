"""Evaluation script for topic-pair-growth prediction.

Computes four metric blocks:

1. **absolute**: per-pair predicted_growth vs gt_growth as raw counts.
2. **share**: normalize each side by its sum across pairs; removes uniform corpus-pipeline-lag confounds and calibration effects.
3. **delta_share**: subtract history_share from both predicted_share and gt_share so the trivial "no change" baseline gets R² ≈ 0; R² > 0 means the baseline predicts real pair dynamics beyond the size prior.
4. **ratio**: divide both sides by history_share; multiplicative analog to delta_share that emphasizes proportional growth (more sensitive to small pairs that double in share than to large pairs that grow by 5%). Filters out pairs absent from history.

The latter two require `--topics_path` so the evaluator can compute history_share. If
omitted, only absolute and share are produced (backward compatible).
"""

import os
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm

import utils
from task_topic_pair_growth_prediction.dataset import load_topics_benchmark


def safe_corrcoef(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).correlation)


def metric_block(pred, gt):
    """Compute MAE, R², Pearson, Spearman, log_mae, log_pearson on a pair of numpy arrays."""
    mae = float(np.mean(np.abs(pred - gt)))
    r2 = float(r2_score(gt, pred)) if len(pred) > 1 else None
    pearson = safe_corrcoef(pred, gt)
    spearman = safe_spearman(pred, gt)
    log_pred = np.log1p(np.maximum(pred, 0))
    log_gt = np.log1p(np.maximum(gt, 0))
    log_mae = float(np.mean(np.abs(log_pred - log_gt)))
    log_pearson = safe_corrcoef(log_pred, log_gt)
    return {"mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman, "log_mae": log_mae, "log_pearson": log_pearson}


def delta_metric_block(pred_delta, gt_delta):
    """Metric block for signed deltas — log_* skipped since deltas can be negative."""
    mae = float(np.mean(np.abs(pred_delta - gt_delta)))
    r2 = float(r2_score(gt_delta, pred_delta)) if len(pred_delta) > 1 else None
    return {"mae": mae, "r2": r2, "pearson": safe_corrcoef(pred_delta, gt_delta), "spearman": safe_spearman(pred_delta, gt_delta), "num_instances": len(pred_delta)}


def normalize_to_share(values, num_topics):
    """Clip negatives, normalize to a probability distribution. Falls back to uniform if sum ≤ 0."""
    clipped = np.maximum(values, 0.0)
    s = clipped.sum()
    if s <= 0:
        return np.full_like(clipped, 1.0 / num_topics)
    return clipped / s


def evaluate_predictions(predictions):
    results = []
    for pred in tqdm(predictions, desc="Evaluating"):
        results.append({"cluster_id": pred["cluster_id"], "predicted": float(pred["predicted_growth"]), "gt": float(pred["gt_growth"]), "abs_error": abs(float(pred["predicted_growth"]) - float(pred["gt_growth"]))})
    return results


def compute_aggregate_metrics(results, history_share_by_id=None):
    if len(results) == 0:
        return {"num_instances": 0, "absolute": None, "share": None, "delta_share": None, "ratio": None}
    predicted = np.array([r["predicted"] for r in results], dtype=np.float64)
    gt = np.array([r["gt"] for r in results], dtype=np.float64)
    n = len(results)

    absolute_metrics = metric_block(predicted, gt)
    pred_share = normalize_to_share(predicted, n)
    gt_share = normalize_to_share(gt, n)
    share_metrics = metric_block(pred_share, gt_share)

    delta_share_metrics = None
    ratio_metrics = None
    if history_share_by_id is not None:
        history_share = np.array([history_share_by_id.get(r["cluster_id"], 0.0) for r in results], dtype=np.float64)
        pred_delta = pred_share - history_share
        gt_delta = gt_share - history_share
        delta_share_metrics = delta_metric_block(pred_delta, gt_delta)
        mask = history_share > 0
        if int(mask.sum()) > 1:
            pred_ratio = pred_share[mask] / history_share[mask]
            gt_ratio = gt_share[mask] / history_share[mask]
            ratio_metrics = metric_block(pred_ratio, gt_ratio)
            ratio_metrics["num_instances"] = int(mask.sum())
        else:
            ratio_metrics = {"mae": None, "r2": None, "pearson": None, "spearman": None, "log_mae": None, "log_pearson": None, "num_instances": int(mask.sum())}

    return {"num_instances": n, "absolute": absolute_metrics, "share": share_metrics, "delta_share": delta_share_metrics, "ratio": ratio_metrics}


def history_share_from_topics(topics_path):
    """Load topic-pair file, compute history_share per pair."""
    _, instances, _, _, _ = load_topics_benchmark(topics_path)
    sizes = {cid: float(inst["history_size"]) for cid, inst in instances}
    total = sum(sizes.values())
    if total <= 0:
        return {cid: 0.0 for cid in sizes}
    return {cid: sz / total for cid, sz in sizes.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate topic-pair growth predictions (absolute, share, delta_share, ratio)")
    parser.add_argument("--predictions_path", type=str, required=True, help="Predictions JSON path")
    parser.add_argument("--topics_path", type=str, default=None, help="Topic-pair benchmark JSON path; required for delta_share and ratio metrics")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_pair_growth_prediction/scored", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading predictions from {args.predictions_path}")
    predictions, predictions_metadata = utils.load_json(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions")

    history_share_by_id = None
    if args.topics_path:
        utils.log(f"Loading pair benchmark from {args.topics_path} for history_share lookup")
        history_share_by_id = history_share_from_topics(args.topics_path)
        utils.log(f"history_share computed for {len(history_share_by_id)} pairs; sum={sum(history_share_by_id.values()):.4f}")

    results = evaluate_predictions(predictions)
    aggregates = compute_aggregate_metrics(results, history_share_by_id=history_share_by_id)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    utils.log("Evaluation complete:")
    utils.log(f"  Instances: {aggregates['num_instances']}")
    for space in ["absolute", "share", "delta_share", "ratio"]:
        m = aggregates.get(space)
        if m is None:
            utils.log(f"  [{space}] (not computed)")
            continue
        utils.log(f"  [{space}] MAE={fmt(m.get('mae'))}  R2={fmt(m.get('r2'))}  Pearson={fmt(m.get('pearson'))}  Spearman={fmt(m.get('spearman'))}  logMAE={fmt(m.get('log_mae'))}  logPearson={fmt(m.get('log_pearson'))}  n={m.get('num_instances', aggregates['num_instances'])}")

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
