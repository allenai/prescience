"""Evaluate topic classifier accuracy on natural papers with known categories."""
import argparse
import os
import random
from collections import Counter

from tqdm import tqdm
import openai

import utils
from multiturn.analysis.classify_synthetic_primary_categories import (
    OTHER_LABEL, build_system_prompt, sample_fewshots, build_query_messages, classify_jobs
)

RNG = random.Random(42)


def main():
    parser = argparse.ArgumentParser(description="Evaluate topic classifier accuracy on natural papers")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo (natural corpus)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07", help="OpenAI chat model name")
    parser.add_argument("--max_workers", type=int, default=128, help="Number of parallel workers for OpenAI requests")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of natural papers to evaluate")
    parser.add_argument("--shots_per_category", type=int, default=5, help="Few-shot examples per natural category")
    parser.add_argument("--max_fewshot_categories", type=int, default=12, help="Maximum number of categories for few-shot")
    parser.add_argument("--output_path", type=str, default="data/corpus/analysis/topic_classifier_evaluation.json", help="Output path for results")
    args = parser.parse_args()

    utils.log(f"Loading natural papers from HuggingFace ({args.split} split)")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    natural_targets = [p for p in all_papers if "target" in p.get("roles", []) and p.get("categories")]
    utils.log(f"Found {len(natural_targets)} natural target papers with categories")

    label_set = set()
    category_counts = Counter()
    for paper in natural_targets:
        label = paper["categories"][0]
        label_set.add(label)
        category_counts[label] += 1
    labels = sorted(label_set)
    if OTHER_LABEL not in labels:
        labels.append(OTHER_LABEL)

    utils.log(f"Found {len(labels)} unique categories")
    utils.log("Top categories: " + ", ".join(f"{l}:{c}" for l, c in category_counts.most_common(10)))

    shot_labels = [label for label, _ in category_counts.most_common(args.max_fewshot_categories)]
    shots, used_ids = sample_fewshots(natural_targets, shot_labels, args.shots_per_category)
    utils.log(f"Using {len(shots)} few-shot examples from {len(used_ids)} papers")

    available_for_eval = [p for p in natural_targets if p["corpus_id"] not in used_ids]
    utils.log(f"Papers available for evaluation: {len(available_for_eval)}")

    RNG.shuffle(available_for_eval)
    eval_papers = available_for_eval[:args.num_samples]
    utils.log(f"Sampled {len(eval_papers)} papers for evaluation")

    system_prompt = build_system_prompt(labels)

    jobs = []
    ground_truth = {}
    for paper in tqdm(eval_papers, desc="Preparing evaluation jobs"):
        corpus_id = paper["corpus_id"]
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        messages = build_query_messages(title, abstract, system_prompt, shots)
        jobs.append((corpus_id, messages))
        ground_truth[corpus_id] = paper["categories"][0]

    utils.log("Running classification")
    client = openai.OpenAI()
    predictions = classify_jobs(jobs, client, args.model, labels, args.max_workers, desc="Evaluating classifier")

    correct = 0
    total = 0
    per_category_correct = Counter()
    per_category_total = Counter()
    confusion = Counter()

    for corpus_id, pred_label in predictions.items():
        gt_label = ground_truth[corpus_id]
        per_category_total[gt_label] += 1
        total += 1
        if pred_label == gt_label:
            correct += 1
            per_category_correct[gt_label] += 1
        confusion[(gt_label, pred_label)] += 1

    overall_accuracy = correct / total if total > 0 else 0.0
    utils.log(f"Overall accuracy: {correct}/{total} = {overall_accuracy:.2%}")

    utils.log("Per-category accuracy:")
    per_category_accuracy = {}
    for label in sorted(per_category_total.keys(), key=lambda l: per_category_total[l], reverse=True):
        cat_correct = per_category_correct[label]
        cat_total = per_category_total[label]
        cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0
        per_category_accuracy[label] = {"correct": cat_correct, "total": cat_total, "accuracy": cat_acc}
        utils.log(f"  {label}: {cat_correct}/{cat_total} = {cat_acc:.2%}")

    results = {
        "overall_accuracy": overall_accuracy,
        "correct": correct,
        "total": total,
        "per_category_accuracy": per_category_accuracy,
        "confusion_matrix": {f"{gt}->{pred}": count for (gt, pred), count in confusion.most_common(50)},
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    utils.save_json(results, args.output_path, utils.update_metadata([], args))
    utils.log(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
