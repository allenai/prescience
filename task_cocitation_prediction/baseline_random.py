"""Random baseline for co-citation prediction. Ranks past papers uniformly at random."""

import os
import random
import argparse
from bisect import bisect_left
from tqdm import tqdm

import utils
from task_cocitation_prediction.dataset import create_evaluation_instances

random.seed(42)


def predict_cocited(cutoff_date, k, sorted_ids, sorted_dates):
    """Sample k past papers uniformly at random from papers with date < cutoff_date."""
    cutoff_idx = bisect_left(sorted_dates, cutoff_date)
    candidates = sorted_ids[:cutoff_idx]
    sample = random.sample(candidates, min(k, len(candidates)))
    return sample, [0.0] * len(sample)


def main():
    parser = argparse.ArgumentParser(description="Random baseline for co-citation prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--eval_months", type=int, default=4, help="Size of evaluation cohort window in months")
    parser.add_argument("--lookahead_months", type=int, default=8, help="Size of future-citation observation window in months")
    parser.add_argument("--k", type=int, default=1000, help="Number of papers to predict per instance")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--eval_instances_cache", type=str, default=None, help="Path to cached evaluation instances JSON; compute and save to this path if it does not exist")
    parser.add_argument("--output_dir", type=str, default="data/task_cocitation_prediction/test/predictions")
    args = parser.parse_args()

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, "predictions.random.json")
    utils.log(f"Loaded {len(all_papers)} papers")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, all_papers_dict, eval_months=args.eval_months, lookahead_months=args.lookahead_months, cache_path=args.eval_instances_cache)

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    sorted_papers = sorted(all_papers, key=lambda p: p["date"])
    sorted_ids = [p["corpus_id"] for p in sorted_papers]
    sorted_dates = [p["date"] for p in sorted_papers]

    utils.log("Running random baseline")
    predictions = []
    for idx, (date, instance) in enumerate(tqdm(evaluation_instances, desc="Running random baseline")):
        if idx not in selected_indices:
            continue

        predicted_ids, predicted_scores = predict_cocited(date, args.k, sorted_ids, sorted_dates)

        predictions.append({
            "corpus_id": instance["corpus_id"],
            "gt_cocited_ids": instance["gt_cocited_ids"],
            "gt_cocited_counts": instance["gt_cocited_counts"],
            "predicted_cocited_ids": predicted_ids,
            "predicted_cocited_scores": predicted_scores,
        })

        if len(predictions) % args.save_every == 0:
            utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
