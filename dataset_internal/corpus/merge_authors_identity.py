"""No-S2AND author merge: assigns each raw S2 author ID its own singleton cluster, preserving the stage04 schema (og2sd/sd2og/sd2publications + sd-format author IDs) expected by downstream scripts."""
import os
import argparse
from tqdm import tqdm

import utils
from dataset_internal.corpus.merge_authors_in_corpus import (
    render_new_sd_author_id,
    prune_stopword_clusters,
    prune_papers_with_illegal_authors_from_dataset,
    download_missing_target_author_pub_history_papers_and_prune,
    download_missing_target_author_pub_history_key_references_papers_and_prune,
    download_missing_target_author_pub_history_authors_info_and_prune,
    refine_sd2publications_after_pruning,
    replace_og_author_ids_with_sd_author_ids,
    sort_all_papers_and_publication_histories_by_date,
)


def build_identity_author_mappings(all_papers_dict, author_id_to_corpus_ids):
    """Create og2sd/sd2og/sd2publications treating every original author ID as its own singleton cluster."""
    sd2og = {}
    og2sd = {}
    sd2publications = {}
    for paper in tqdm(all_papers_dict.values(), desc="Building identity author mappings"):
        if "authors" in paper:  # this will be true for "target" ∪ "target.author.publication_history" papers
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                if og_author_id not in og2sd:
                    sd_author_id = render_new_sd_author_id(author, sd2og)
                    og2sd[og_author_id] = sd_author_id
                    sd2og[sd_author_id] = {og_author_id}
                    if og_author_id in author_id_to_corpus_ids:  # target-paper author with a downloaded publication list
                        sd2publications[sd_author_id] = set(author_id_to_corpus_ids[og_author_id])
                    else:
                        sd2publications[sd_author_id] = set()
                sd2publications[og2sd[og_author_id]].add(paper["corpus_id"])

    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = prune_stopword_clusters(sd2og, og2sd, sd2publications)
    utils.log(f"Constructed identity mappings for {len(og2sd)} original authors across {len(sd2og)} singleton clusters")
    return sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og


def main():
    """Entry point for the identity (no-S2AND) merge workflow."""
    parser = argparse.ArgumentParser("Author merge pipeline using identity (singleton) clusters instead of S2AND.")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Directory containing all_papers.stage03.json and author_id_to_corpus_ids.json")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for Athena queries")
    parser.add_argument("--max_workers", type=int, default=100, help="Number of workers for Athena queries")
    parser.add_argument("--max_key_references", type=int, default=10, help="Max key references per publication history paper")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for merged all_papers")
    args = parser.parse_args()

    all_papers, metadata = utils.load_json(os.path.join(args.input_dir, "all_papers.stage03.json"))
    metadata = metadata if metadata is not None else []
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}

    author_id_to_corpus_ids, _ = utils.load_json(os.path.join(args.input_dir, "author_id_to_corpus_ids.json"))
    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = build_identity_author_mappings(all_papers_dict, author_id_to_corpus_ids)
    all_papers_dict = prune_papers_with_illegal_authors_from_dataset(all_papers_dict, pruned_authors_og)

    all_papers_dict, sd2publications = download_missing_target_author_pub_history_papers_and_prune(og2sd, sd2publications, all_papers_dict, args)
    all_papers_dict = download_missing_target_author_pub_history_key_references_papers_and_prune(all_papers_dict, args)
    all_papers_dict, sd2og, og2sd, sd2publications = download_missing_target_author_pub_history_authors_info_and_prune(all_papers_dict, sd2og, og2sd, sd2publications, args)

    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = prune_stopword_clusters(sd2og, og2sd, sd2publications)
    all_papers_dict = prune_papers_with_illegal_authors_from_dataset(all_papers_dict, pruned_authors_og)
    sd2publications = refine_sd2publications_after_pruning(sd2publications, all_papers_dict)

    all_papers_dict = replace_og_author_ids_with_sd_author_ids(all_papers_dict, og2sd)

    all_papers_dict, sd2publications = sort_all_papers_and_publication_histories_by_date(all_papers_dict, sd2publications)
    sd2og = {sd: list(ogs) for sd, ogs in sd2og.items()}  # convert sets to lists for JSON serialization
    sd2publications = {sd: (None if pubs is None else list(pubs)) for sd, pubs in sd2publications.items()}  # convert sets to lists for JSON serialization

    all_papers = list(all_papers_dict.values())
    updated_metadata = utils.update_metadata(metadata, args)
    utils.log(f"Saving {len(all_papers)} papers with identity author merges to stage04 output")
    utils.save_json(all_papers, os.path.join(args.output_dir, "all_papers.stage04.json"), metadata=updated_metadata)
    utils.save_json(sd2og, os.path.join(args.output_dir, "sd2og.json"), metadata=updated_metadata)
    utils.save_json(og2sd, os.path.join(args.output_dir, "og2sd.json"), metadata=updated_metadata)
    utils.save_json(sd2publications, os.path.join(args.output_dir, "sd2publications.json"), metadata=updated_metadata)


if __name__ == "__main__":
    main()
