"""Compute paper embeddings for all papers in the corpus."""

import os
import argparse

import utils


def main():
    parser = argparse.ArgumentParser("Compute embeddings for all papers")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace dataset repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--embedding_type", type=str, choices=["gtr", "grit", "specter2"], required=True, help="Type of embedding to compute")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for embeddings")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_corpus_ids = set(paper["corpus_id"] for paper in all_papers)

    all_embeddings_path = os.path.join(args.output_dir, f"all_papers.{args.embedding_type}_embeddings.pkl")
    if os.path.exists(all_embeddings_path):
        corpus_id2embeddings, _ = utils.load_pkl(all_embeddings_path)
        existing_corpus_ids = set(corpus_id2embeddings.keys())
        missing_corpus_ids = [paper["corpus_id"] for paper in all_papers if paper["corpus_id"] not in existing_corpus_ids]
        papers_to_embed = [paper for paper in all_papers if paper["corpus_id"] not in existing_corpus_ids]
        utils.log(f"Found {len(missing_corpus_ids)} missing embeddings out of {len(all_corpus_ids)} papers")
    else:
        corpus_id2embeddings = {}
        missing_corpus_ids = [paper["corpus_id"] for paper in all_papers]
        papers_to_embed = all_papers
        utils.log(f"Computing embeddings for {len(missing_corpus_ids)} papers")

    title_abstracts = [utils.get_title_abstract_string(paper) for paper in papers_to_embed]
    all_embeddings_dicts = utils.embed_on_gpus_parallel(title_abstracts, embedding_type=args.embedding_type)
    for c_id, e_dict in zip(missing_corpus_ids, all_embeddings_dicts):
        corpus_id2embeddings[c_id] = e_dict

    # Remove embeddings for papers no longer in the corpus
    stale_ids = [c_id for c_id in corpus_id2embeddings if c_id not in all_corpus_ids]
    for c_id in stale_ids:
        del corpus_id2embeddings[c_id]
    utils.log(f"Removed {len(stale_ids)} stale embeddings")

    utils.log(f"Saving {len(corpus_id2embeddings)} embeddings to {all_embeddings_path}")
    utils.save_pkl(corpus_id2embeddings, all_embeddings_path)


if __name__ == "__main__":
    main()
