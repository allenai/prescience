"""Mean baseline for topic growth prediction (causally safe, inner-split calibrated).

Calibrates a constant prediction using only inner-history-period information:
  c = mean_t(N_t(F_inner)) × (|F| / |F_inner|)
where F_inner is the last (|H| - N) months of history, |F| is the outer forecast length in months.
"""

import os
import argparse

import utils
from task_topic_growth_prediction.dataset import load_topics_benchmark


def main():
    parser = argparse.ArgumentParser(description="Mean baseline (inner-split calibrated)")
    parser.add_argument("--topics_path", type=str, required=True, help="Path to topics.vX.json")
    parser.add_argument("--inner_history_months", type=int, default=6, help="First N months of history used as inner-history; remaining months are inner-forecast")
    parser.add_argument("--output_dir", type=str, default="data/task_topic_growth_prediction/predictions", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading benchmark from {args.topics_path}")
    config, instances, _, _, metadata = load_topics_benchmark(args.topics_path)
    H = config["history_months"]
    F = config["forecast_months"]
    N = args.inner_history_months
    assert 1 <= N < H, f"--inner_history_months must be in [1, {H-1}]"
    F_inner = H - N
    utils.log(f"|H|={H}, |F|={F}, inner_history_months={N}, |H_inner|={N}, |F_inner|={F_inner}")

    # mean over topics of their inner-forecast size
    inner_forecast_sizes = [sum(inst["history_monthly_counts"][N:H]) for _, inst in instances]
    mean_inner = sum(inner_forecast_sizes) / len(instances) if instances else 0.0
    scale = F / F_inner
    constant_prediction = mean_inner * scale
    utils.log(f"mean(inner_forecast_size) = {mean_inner:.4f}; scale = {scale:.4f}; predicted constant = {constant_prediction:.4f}")

    predictions = [{"cluster_id": cid, "predicted_growth": float(constant_prediction), "gt_growth": inst["gt_growth"]} for cid, inst in instances]
    output_path = os.path.join(args.output_dir, "predictions.mean.json")
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(metadata, args), overwrite=True)


if __name__ == "__main__":
    main()
