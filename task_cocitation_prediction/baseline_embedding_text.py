"""Embedding-text baseline for co-citation prediction. Ranks past papers by similarity to P's paper embedding."""

import os
import random
import argparse
from tqdm import tqdm

import utils
from task_cocitation_prediction.dataset import create_evaluation_instances

random.seed(42)


def get_embedding(all_embeddings, corpus_id):
    """Get key embedding for a paper."""
    return all_embeddings[corpus_id]["key"].reshape(-1)


def create_paper_index(cutoff_date, all_papers_dict, all_embeddings, distance_metric):
    """Create FAISS index of paper embeddings for papers before cutoff_date."""
    paper_embeddings = {}
    for corpus_id, paper in all_papers_dict.items():
        if paper["date"] < cutoff_date and corpus_id in all_embeddings:
            paper_embeddings[corpus_id] = get_embedding(all_embeddings, corpus_id)
    utils.log(f"Created embeddings for {len(paper_embeddings)} papers")
    return utils.create_index(paper_embeddings, distance_metric)


def predict_cocited(corpus_id, k, index, all_embeddings):
    """Predict co-cited papers by querying FAISS with P's own paper embedding."""
    if corpus_id not in all_embeddings:
        return [], []
    query_embedding = get_embedding(all_embeddings, corpus_id)
    retrieved_papers_lists, distances_lists = utils.query_index(index, [query_embedding], k)
    predicted_ids = [cid for cid in retrieved_papers_lists[0] if cid is not None]
    predicted_scores = [float(d) for cid, d in zip(retrieved_papers_lists[0], distances_lists[0]) if cid is not None]
    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Embedding-text baseline for co-citation prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--eval_months", type=int, default=4, help="Size of evaluation cohort window in months")
    parser.add_argument("--lookahead_months", type=int, default=8, help="Size of future-citation observation window in months")
    parser.add_argument("--k", type=int, default=1000, help="Number of papers to predict per instance")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--eval_instances_cache", type=str, default=None, help="Path to cached evaluation instances JSON; compute and save to this path if it does not exist")
    parser.add_argument("--output_dir", type=str, default="data/task_cocitation_prediction/test/predictions")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, _, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=False)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.embedding_text.{args.embedding_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers and {len(all_embeddings)} embeddings")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, all_papers_dict, eval_months=args.eval_months, lookahead_months=args.lookahead_months, cache_path=args.eval_instances_cache)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    all_papers = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial paper index with cutoff date: {first_date}")
    index = create_paper_index(first_date, all_papers_dict, all_embeddings, distance_metric)
    postdated_papers = [p for p in all_papers if p["date"] >= first_date]

    utils.log(f"Running embedding-text baseline with embedding: {args.embedding_type}")
    predictions = []
    pending_adds = []
    for paper in tqdm(postdated_papers, desc="Running embedding-text baseline"):
        if len(pending_adds) > 0 and pending_adds[0]["date"] < paper["date"]:
            for p in pending_adds:
                if p["corpus_id"] in all_embeddings:
                    index = utils.add_vector_to_index(index, p["corpus_id"], get_embedding(all_embeddings, p["corpus_id"]))
            pending_adds = []

        corpus_id = paper["corpus_id"]
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                predicted_ids, predicted_scores = predict_cocited(corpus_id, args.k, index, all_embeddings)

                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_cocited_ids": instance["gt_cocited_ids"],
                    "gt_cocited_counts": instance["gt_cocited_counts"],
                    "predicted_cocited_ids": predicted_ids,
                    "predicted_cocited_scores": predicted_scores,
                })

                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        pending_adds.append(paper)

    for p in pending_adds:
        if p["corpus_id"] in all_embeddings:
            index = utils.add_vector_to_index(index, p["corpus_id"], get_embedding(all_embeddings, p["corpus_id"]))

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
