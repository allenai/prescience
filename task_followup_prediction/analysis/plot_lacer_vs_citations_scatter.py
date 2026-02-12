"""Scatter plot of LACER score vs 12-month citations for followup prediction."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import utils


def main():
    parser = argparse.ArgumentParser(description="Scatter plot of LACER score vs future citations")
    parser.add_argument("--input_path", type=str, default="data/task_followup_prediction/test/lacer_scored/generations.gpt-5.2-2025-12-11.lacer_scored.json", help="Path to LACER-scored file")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/analysis", help="Output directory")
    parser.add_argument("--months", type=int, default=12, help="Number of months for citation count")
    args = parser.parse_args()

    utils.log(f"Loading data from {args.input_path}")
    data_file, _ = utils.load_json(args.input_path)
    data = data_file["data"] if "data" in data_file else data_file

    # Extract citation counts and LACER scores
    citations = []
    lacer_scores = []
    for record in data:
        if "lacer_score" not in record:
            continue
        trajectory = record.get("citation_trajectory", [])
        if len(trajectory) < args.months:
            continue
        citations.append(trajectory[args.months - 1])
        lacer_scores.append(record["lacer_score"])

    citations = np.array(citations)
    lacer_scores = np.array(lacer_scores)
    utils.log(f"Extracted {len(citations)} instances with both LACER scores and {args.months}-month citations")

    # Compute Spearman correlation
    corr, p_value = stats.spearmanr(citations, lacer_scores)
    p_str = f"p={p_value:.0e}" if p_value < 0.01 else f"p={p_value:.2f}"
    utils.log(f"Spearman ρ={corr:.3f}, {p_str}")

    # Fit line in log space
    log_citations = np.log10(citations + 1)
    slope, intercept = np.polyfit(log_citations, lacer_scores, 1)
    x_fit = np.linspace(log_citations.min(), log_citations.max(), 100)
    y_fit = slope * x_fit + intercept

    # Plot
    plt.figure(figsize=(3.5, 3.5))
    plt.rcParams.update({"font.size": 8})

    plt.scatter(citations + 1, lacer_scores, alpha=0.3, s=8, c="#0072B2", rasterized=True)
    plt.plot(10 ** x_fit, y_fit, "k--", linewidth=1, alpha=0.7,
             label=f"Best fit (Spearman ρ={corr:.2f}, {p_str})")

    plt.xscale("log")
    plt.xlabel(f"Citations @ {args.months} Months", fontsize=8)
    plt.ylabel("LACER Score", fontsize=8)
    plt.title("LACER Score vs Future Citations", fontsize=9, fontweight="bold")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, 10)
    plt.legend(loc="upper left", fontsize=6, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "lacer_vs_citations_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
