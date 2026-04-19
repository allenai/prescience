"""Historical popularity baseline for co-citation prediction. Ranks past papers by how often they appear as a key_reference in corpus papers predating P."""

import os
import random
import argparse
from bisect import bisect_left
from tqdm import tqdm

import utils
from task_cocitation_prediction.dataset import create_evaluation_instances

random.seed(42)


def predict_cocited(popularity_counter, cutoff_date, k, sorted_ids, sorted_dates):
    """Rank past papers by historical key_reference popularity; pad with random past papers if needed."""
    cutoff_idx = bisect_left(sorted_dates, cutoff_date)
    candidate_set = set(sorted_ids[:cutoff_idx])

    scored = [(cid, popularity_counter[cid]) for cid in popularity_counter if cid in candidate_set]
    scored.sort(key=lambda x: x[1], reverse=True)
    predicted_ids = [cid for cid, _ in scored]
    predicted_scores = [float(score) for _, score in scored]

    if len(predicted_ids) < k:
        excluded = set(predicted_ids)
        pool = [cid for cid in sorted_ids[:cutoff_idx] if cid not in excluded]
        num_needed = k - len(predicted_ids)
        random_papers = random.sample(pool, min(num_needed, len(pool)))
        predicted_ids.extend(random_papers)
        predicted_scores.extend([0.0] * len(random_papers))

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Historical popularity baseline for co-citation prediction")
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
    output_path = os.path.join(args.output_dir, "predictions.historical_popularity.json")
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
    citer_papers = [p for p in sorted_papers if p.get("key_references")]
    utils.log(f"Historical citer pool: {len(citer_papers)} papers with populated key_references")

    utils.log("Running historical popularity baseline")
    popularity_counter = {}
    citer_idx = 0
    predictions = []
    for idx, (date, instance) in enumerate(tqdm(evaluation_instances, desc="Running historical popularity baseline")):
        while citer_idx < len(citer_papers) and citer_papers[citer_idx]["date"] < date:
            for ref in citer_papers[citer_idx]["key_references"]:
                if ref["corpus_id"] not in popularity_counter:
                    popularity_counter[ref["corpus_id"]] = 0
                popularity_counter[ref["corpus_id"]] += 1
            citer_idx += 1

        if idx not in selected_indices:
            continue

        predicted_ids, predicted_scores = predict_cocited(popularity_counter, date, args.k, sorted_ids, sorted_dates)

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
