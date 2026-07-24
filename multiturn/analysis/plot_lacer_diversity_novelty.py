"""Plot combined LACER diversity and novelty time series as a single shared-y figure."""
import os
import random
import argparse
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import utils
from multiturn.analysis.plot_lacer_diversity import bootstrap_ci, load_raw_scores

NATURAL_COLOR = "#0072B2"
SYNTHETIC_COLOR = "#E69F00"

PANELS = [
    ("Diversity (within month)", "lacer_diversity_natural_monthly_matched.json", "lacer_diversity_synthetic_{}_monthly.json"),
    (r"Novelty vs. $\mathbf{H}^{<t}$", "lacer_novelty_natural_monthly_matched.json", "lacer_novelty_synthetic_{}_monthly.json"),
    (r"Novelty vs. $\mathbf{H}^{<t_0}$", "lacer_novelty_natural_monthly_fixed_matched.json", "lacer_novelty_synthetic_{}_monthly_fixed.json"),
]


def panel_series(natural_path, synthetic_paths, n_bootstrap, rng):
    """Return bootstrapped (mean, lower, upper) per bucket for the natural and pooled-synthetic corpora."""
    natural = {b: bootstrap_ci(s, n_bootstrap, rng) for b, s in sorted(load_raw_scores(natural_path).items())}
    pooled = defaultdict(list)
    for path in synthetic_paths:
        for bucket, scores in load_raw_scores(path).items():
            pooled[bucket].extend(scores)
    synthetic = {b: bootstrap_ci(pooled[b], n_bootstrap, rng) for b in sorted(pooled)}
    return natural, synthetic


def draw_series(ax, results, color, label):
    """Plot a mean line with a shaded confidence band on the given axis."""
    buckets = sorted(results)
    dates = [datetime.strptime(b, "%Y-%m-%d") for b in buckets]
    ax.plot(dates, [results[b][0] for b in buckets], label=label, color=color, linewidth=1.5)
    ax.fill_between(dates, [results[b][1] for b in buckets], [results[b][2] for b in buckets], color=color, alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description="Plot combined LACER diversity and novelty figure with a shared y-axis.")
    parser.add_argument("--analysis_dir", type=str, default="data/multiturn/analysis", help="Directory with computed LACER diversity/novelty JSONs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 52, 57, 59], help="Synthetic rollout seeds to pool")
    parser.add_argument("--output_path", type=str, default="figures/multiturn/lacer_diversity_novelty.png", help="Output figure path")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    plt.rcParams.update({"font.size": 9.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    years = set()
    for ax, (title, natural_file, synthetic_template) in zip(axes, PANELS):
        utils.log(f"Processing panel: {title}")
        natural_path = os.path.join(args.analysis_dir, natural_file)
        synthetic_paths = [os.path.join(args.analysis_dir, synthetic_template.format(s)) for s in args.seeds]
        natural, synthetic = panel_series(natural_path, synthetic_paths, args.n_bootstrap, rng)
        years.update(b[:4] for b in natural)
        draw_series(ax, natural, NATURAL_COLOR, "Natural")
        draw_series(ax, synthetic, SYNTHETIC_COLOR, "Synthetic")
        ax.set_title(title, fontsize=9.6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8.8)

    axes[0].set_ylabel("(10 - Mean LACER)", fontsize=9.6)
    axes[0].set_yticks([4, 5, 6, 7])
    axes[0].set_yticklabels(["4.0", "5.0", "6.0", "7.0"])
    axes[2].legend(loc="upper right", fontsize=8.8, framealpha=0.9)
    year_range = sorted(years)
    fig.supxlabel(f"Date ({year_range[0]}–{year_range[-1]})" if len(year_range) > 1 else f"Date ({year_range[0]})", fontsize=9.6)
    fig.tight_layout(pad=0.5)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
