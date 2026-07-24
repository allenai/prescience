"""Temporal extrapolation baseline for topic growth prediction (causally safe).

Per-topic linear fit on history_monthly_counts; extrapolates forward for |F| months and sums.
Optionally fits in log1p space (multiplicative). Uses only history-period data.
"""

import os
import argparse
import numpy as np

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark


def extrapolate_linear(monthly_counts, num_forecast_months, log_space=False):
    """Fit y = a·t + b on monthly counts and return total predicted growth over num_forecast_months."""
    y = np.asarray(monthly_counts, dtype=np.float64)
    t = np.arange(1, len(y) + 1, dtype=np.float64)
    y_fit = np.log1p(y) if log_space else y
    a, b = np.polyfit(t, y_fit, 1)
    future_t = np.arange(len(y) + 1, len(y) + 1 + num_forecast_months, dtype=np.float64)
    pred = a * future_t + b
    if log_space:
        pred = np.expm1(pred)
    return float(np.sum(np.maximum(pred, 0.0)))


def main():
    parser = argparse.ArgumentParser(description="Linear-trend extrapolation baseline")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--log_space", action="store_true", help="Fit the trend in log1p space (multiplicative)")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    F = config["forecast_months"]
    utils.log(f"Extrapolating {F} forecast months (log_space={args.log_space})")

    predictions = []
    for cid, inst in instances:
        predicted = extrapolate_linear(inst["history_monthly_counts"], F, log_space=args.log_space)
        predictions.append({"cluster_id": cid, "predicted_growth": predicted, "gt_growth": inst["gt_growth"]})

    suffix = "log" if args.log_space else "linear"
    output_path = os.path.join(args.output_dir, f"predictions.extrapolation_{suffix}.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
