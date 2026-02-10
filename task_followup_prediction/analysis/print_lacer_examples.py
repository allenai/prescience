"""Print example (ground truth, generated) abstract pairs for each LACER score."""

import os
import sys
import argparse
import random

sys.path.insert(0, os.getcwd())
import utils

random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="Print LACER score examples")
    parser.add_argument("--scored_path", type=str, default="data/task_followup_prediction/test/lacer_scored/generations.gpt-5-2025-08-07.lacer_scored.json")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--k", type=int, default=2, help="Examples per score")
    args = parser.parse_args()

    scored_data, _ = utils.load_json(args.scored_path)
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Group by rounded score
    by_score = {i: [] for i in range(1, 11)}
    for record in scored_data:
        score = int(round(record["lacer_score"]))
        if 1 <= score <= 10:
            by_score[score].append(record)

    # Print k examples per score
    for score in range(1, 11):
        instances = by_score[score]
        if len(instances) == 0:
            continue

        print(f"\n{'='*80}")
        print(f"LACER SCORE: {score} ({len(instances)} total instances)")
        print(f"{'='*80}")

        samples = random.sample(instances, min(args.k, len(instances)))
        for i, record in enumerate(samples):
            gt = all_papers_dict[record["corpus_id"]]
            print(f"\n--- Example {i+1} (corpus_id: {record['corpus_id']}) ---")
            print(f"\nGROUND TRUTH TITLE: {gt['title']}")
            print(f"\nGROUND TRUTH ABSTRACT:\n{gt['abstract']}")
            print(f"\nGENERATED TITLE: {record['title']}")
            print(f"\nGENERATED ABSTRACT:\n{record['abstract']}")


if __name__ == "__main__":
    main()
