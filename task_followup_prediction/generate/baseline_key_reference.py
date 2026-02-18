"""Key reference baseline for followup work prediction. Predicts by randomly sampling a key reference."""

import os
import random
import argparse
from tqdm import tqdm

import utils

random.seed(42)


def predict_from_key_reference(record, all_papers_dict):
    """Randomly select a key reference and use its title/abstract as the prediction."""
    key_refs = record["key_references"]
    selected_ref = random.choice(key_refs)
    ref_paper = all_papers_dict[selected_ref["corpus_id"]]

    return {
        "title": ref_paper["title"],
        "abstract": ref_paper["abstract"],
        "reasoning": f"Randomly selected key reference: {selected_ref['corpus_id']}"
    }


def main():
    """Generate followup work predictions by randomly sampling key references."""
    parser = argparse.ArgumentParser(description="Key reference baseline for followup work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations")
    parser.add_argument("--max_query_papers", type=int, default=5000)
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = utils.filter_by_roles(all_papers, ["target"])

    if len(query_papers) > args.max_query_papers:
        query_papers = random.sample(query_papers, args.max_query_papers)
    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    utils.log("Generating predictions from key references")
    for rec in tqdm(query_papers, desc="Generating predictions"):
        prediction = predict_from_key_reference(rec, all_papers_dict)
        rec.update(prediction)

    output_path = os.path.join(args.output_dir, "generations.key_reference.json")
    utils.save_json(query_papers, output_path, overwrite=True, metadata=utils.update_metadata([], args))
    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()
