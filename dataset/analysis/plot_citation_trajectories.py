"""Average citation trajectories for target papers."""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def main():
    parser = argparse.ArgumentParser(description="Plot average citation trajectories for target papers")
    parser.add_argument("--train_dir", type=str, default="data/corpus/train", help="Train corpus directory")
    parser.add_argument("--test_dir", type=str, default="data/corpus/test", help="Test corpus directory")
    parser.add_argument("--output_dir", type=str, default="data/corpus/analysis", help="Output directory")
    args = parser.parse_args()

    utils.log(f"Loading train corpus from {args.train_dir}")
    train_all_papers, _ = utils.load_json(os.path.join(args.train_dir, "all_papers.json"))
    train_target_papers = [p for p in train_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(train_target_papers)} train target papers")

    utils.log(f"Loading test corpus from {args.test_dir}")
    test_all_papers, _ = utils.load_json(os.path.join(args.test_dir, "all_papers.json"))
    test_target_papers = [p for p in test_all_papers if "target" in p.get("roles", [])]
    utils.log(f"Loaded {len(test_target_papers)} test target papers")

    train_trajectories = {}
    for paper in train_target_papers:
        traj = paper.get("citation_trajectory", [])
        if len(traj) > 0:
            length = len(traj)
            if length not in train_trajectories:
                train_trajectories[length] = []
            train_trajectories[length].append(traj)

    test_trajectories = {}
    for paper in test_target_papers:
        traj = paper.get("citation_trajectory", [])
        if len(traj) > 0:
            length = len(traj)
            if length not in test_trajectories:
                test_trajectories[length] = []
            test_trajectories[length].append(traj)

    train_avg = {length: np.mean(trajs, axis=0) for length, trajs in train_trajectories.items()}
    test_avg = {length: np.mean(trajs, axis=0) for length, trajs in test_trajectories.items()}

    utils.log("Train trajectory lengths and counts:")
    for length in sorted(train_trajectories.keys()):
        utils.log(f"  {length} months: {len(train_trajectories[length])} papers")
    utils.log("Test trajectory lengths and counts:")
    for length in sorted(test_trajectories.keys()):
        utils.log(f"  {length} months: {len(test_trajectories[length])} papers")

    utils.log("Creating plot")
    plt.figure(figsize=(5, 3.5))
    plt.rcParams.update({"font.size": 8})

    for length, avg_traj in sorted(train_avg.items()):
        label = "Train" if length == min(train_avg.keys()) else None
        plt.plot(range(1, length + 1), avg_traj, color="orange", alpha=0.7, label=label)

    for length, avg_traj in sorted(test_avg.items()):
        label = "Test" if length == min(test_avg.keys()) else None
        plt.plot(range(1, length + 1), avg_traj, color="blue", alpha=0.7, label=label)

    plt.xlabel("Months Since Publication", fontsize=8)
    plt.ylabel("Average Citations", fontsize=8)
    plt.title("Average Citation Trajectories for Target Papers", fontsize=9, fontweight="bold")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(fontsize=7, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "citation_trajectories.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved plot to {output_path}")

    summary = {
        "train_trajectory_counts": {str(k): len(v) for k, v in sorted(train_trajectories.items())},
        "test_trajectory_counts": {str(k): len(v) for k, v in sorted(test_trajectories.items())},
        "train_total_with_trajectory": sum(len(v) for v in train_trajectories.values()),
        "test_total_with_trajectory": sum(len(v) for v in test_trajectories.values()),
    }
    summary_path = os.path.join(args.output_dir, "citation_trajectories_summary.json")
    utils.save_json(summary, summary_path, utils.update_metadata([], args))
    utils.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
