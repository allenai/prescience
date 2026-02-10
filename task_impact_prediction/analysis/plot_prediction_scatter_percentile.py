"""Percentile-percentile scatter plot of predicted vs actual citations for impact prediction."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

import utils


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs actual citations on percentile scale")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation results file")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading evaluation results from {args.eval_path}")
    eval_file, _ = utils.load_json(args.eval_path)
    data = eval_file["data"] if "data" in eval_file else eval_file
    instances = data["per_instance"]
    utils.log(f"Loaded {len(instances)} instances")

    predicted = np.array([inst["predicted"] for inst in instances])
    gt = np.array([inst["gt"] for inst in instances])

    # Compute percentile ranks (0-100)
    predicted_pct = 100 * rankdata(predicted, method='average') / len(predicted)
    gt_pct = 100 * rankdata(gt, method='average') / len(gt)

    # Classify as over/under prediction based on percentile
    overpredicted = predicted_pct > gt_pct
    underpredicted = predicted_pct < gt_pct

    # Compute Spearman correlation
    spearman_corr, spearman_p = spearmanr(predicted, gt)
    utils.log(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")

    utils.log(f"Overpredicted (by percentile): {np.sum(overpredicted)} ({100*np.mean(overpredicted):.1f}%)")
    utils.log(f"Underpredicted (by percentile): {np.sum(underpredicted)} ({100*np.mean(underpredicted):.1f}%)")

    utils.log("Creating plot")
    plt.figure(figsize=(3.5, 3.5))
    plt.rcParams.update({"font.size": 8})

    plt.scatter(gt_pct[underpredicted], predicted_pct[underpredicted],
                alpha=0.3, s=8, c="#0072B2", label="Underpredicted", rasterized=True)
    plt.scatter(gt_pct[overpredicted], predicted_pct[overpredicted],
                alpha=0.3, s=8, c="#D55E00", label="Overpredicted", rasterized=True)

    plt.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.7, label="Perfect")

    plt.xlabel("Actual Citation Percentile", fontsize=8)
    plt.ylabel("Predicted Citation Percentile", fontsize=8)
    plt.title(f"P-P Plot (Spearman r={spearman_corr:.3f})", fontsize=9, fontweight="bold")
    plt.legend(loc="upper left", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "prediction_scatter_percentile.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    summary = {
        "total_instances": len(instances),
        "spearman_correlation": float(spearman_corr),
        "spearman_pvalue": float(spearman_p),
        "overpredictions_pct": int(np.sum(overpredicted)),
        "underpredictions_pct": int(np.sum(underpredicted)),
        "predicted_pct_std": float(np.std(predicted_pct)),
        "gt_pct_std": float(np.std(gt_pct)),
    }
    summary_path = os.path.join(args.output_dir, "prediction_scatter_percentile_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
