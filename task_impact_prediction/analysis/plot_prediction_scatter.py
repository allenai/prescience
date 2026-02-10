"""Log-log scatter plot of predicted vs actual citations for impact prediction."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs actual citations on log-log scale")
    parser.add_argument("--eval_path", type=str, default="data/task_impact_prediction/test/scored/predictions.xgboost_regressor_grit_author_numbers_author_papers_prior_work_papers_prior_work_numbers_followup_work_paper.eval.json", help="Path to evaluation results file")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading evaluation results from {args.eval_path}")
    eval_file, _ = utils.load_json(args.eval_path)
    data = eval_file["data"] if "data" in eval_file else eval_file
    instances = data["per_instance"]
    utils.log(f"Loaded {len(instances)} instances")

    # Extract predicted and ground truth values
    predicted = np.array([inst["predicted"] for inst in instances])
    gt = np.array([inst["gt"] for inst in instances])

    # Add small offset for log scale (to handle zeros)
    offset = 0.5
    predicted_log = predicted + offset
    gt_log = gt + offset

    # Classify as over/under prediction
    overpredicted = predicted > gt
    underpredicted = predicted < gt
    accurate = predicted == gt

    utils.log(f"Overpredictions: {np.sum(overpredicted)} ({100*np.mean(overpredicted):.1f}%)")
    utils.log(f"Underpredictions: {np.sum(underpredicted)} ({100*np.mean(underpredicted):.1f}%)")
    utils.log(f"Exact: {np.sum(accurate)} ({100*np.mean(accurate):.1f}%)")

    # Create plot
    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 3.5))
    plt.rcParams.update({"font.size": 8})

    # Plot underpredictions (blue) and overpredictions (red)
    plt.scatter(gt_log[underpredicted], predicted_log[underpredicted],
                alpha=0.3, s=8, c="#0072B2", label="Underpredicted", rasterized=True)
    plt.scatter(gt_log[overpredicted], predicted_log[overpredicted],
                alpha=0.3, s=8, c="#D55E00", label="Overpredicted", rasterized=True)

    # Add diagonal line (perfect prediction)
    min_val = offset
    max_val = max(np.max(gt_log), np.max(predicted_log))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.7, label="Perfect")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Citations (12 months)", fontsize=8)
    plt.ylabel("Predicted Citations", fontsize=8)
    plt.title("Impact Prediction: Predicted vs Actual", fontsize=9, fontweight="bold")
    plt.legend(loc="upper left", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, which="both")
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prediction_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    # Save summary stats
    summary = {
        "total_instances": len(instances),
        "overpredictions": int(np.sum(overpredicted)),
        "underpredictions": int(np.sum(underpredicted)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_gt": float(np.mean(gt)),
        "median_predicted": float(np.median(predicted)),
        "median_gt": float(np.median(gt)),
    }
    summary_path = os.path.join(args.output_dir, "prediction_scatter_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
