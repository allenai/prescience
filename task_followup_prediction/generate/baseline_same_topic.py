"""Same-topic baseline for followup work prediction. Predicts by sampling a paper with the same primary category."""

import os
import random
import argparse
from tqdm import tqdm

import utils

random.seed(42)


def predict_from_same_topic(record, papers_by_category, all_papers_dict):
    """Randomly select a paper with the same primary category as the prediction."""
    primary_category = record["categories"][0]
    candidates = [cid for cid in papers_by_category.get(primary_category, []) if cid != record["corpus_id"]]
    if not candidates:
        candidates = [cid for cid in all_papers_dict.keys() if cid != record["corpus_id"]]
    selected_id = random.choice(candidates)
    selected_paper = all_papers_dict[selected_id]
    return {
        "title": selected_paper["title"],
        "abstract": selected_paper["abstract"],
        "reasoning": f"Randomly selected paper with same primary category ({primary_category}): {selected_id}"
    }


def main():
    """Generate followup work predictions by sampling papers with matching primary category."""
    parser = argparse.ArgumentParser(description="Same-topic baseline for followup work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--max_query_papers", type=int, default=5000)
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    papers_by_category = {}
    for p in all_papers:
        if p["categories"]:
            primary_cat = p["categories"][0]
            if primary_cat not in papers_by_category:
                papers_by_category[primary_cat] = []
            papers_by_category[primary_cat].append(p["corpus_id"])

    query_papers = utils.filter_by_roles(all_papers, ["target"])
    if len(query_papers) > args.max_query_papers:
        query_papers = random.sample(query_papers, args.max_query_papers)
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    utils.log("Generating predictions by sampling same-topic papers")
    for rec in tqdm(query_papers, desc="Generating predictions"):
        prediction = predict_from_same_topic(rec, papers_by_category, all_papers_dict)
        rec.update(prediction)

    output_path = os.path.join(args.output_dir, "generations.same_topic.json")
    utils.save_json(query_papers, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()
