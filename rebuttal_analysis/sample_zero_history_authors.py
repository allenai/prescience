"""Sample zero-prior-history authors for manual/web adjudication (Reviewer 4 Q3; Reviewer 1 W3).

A roster author with zero prior publications in the corpus at their paper's date is either a genuine
first-appearance author or a disambiguation/data error. We over-sample high-h-index authors, since a
senior author showing zero history is the informative error case. Emits candidates with the metadata
needed to look each one up (name, target paper, arXiv id, and S2 global counts at publication time)."""

import argparse
import json
import os
import random

import utils
from task_coauthor_prediction.dataset import get_preexisting_publications_for_author

random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="Sample zero-prior-history authors for adjudication")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--num_first_timers", type=int, default=6, help="Sampled authors with S2 num_papers <= 1")
    parser.add_argument("--num_suspicious", type=int, default=6, help="Sampled zero-history authors with the highest S2 h-index")
    parser.add_argument("--output_dir", default="rebuttal_outputs/C_author_robustness", help="Directory for outputs")
    args = parser.parse_args()

    all_papers, sd2publications, _ = utils.load_corpus(split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    targets = [p for p in all_papers if "target" in p["roles"]]

    zero_history = []
    for paper in targets:
        for author in paper["authors"]:
            if len(get_preexisting_publications_for_author(author["author_id"], paper["date"], sd2publications, all_papers_dict)) == 0:
                zero_history.append({
                    "author_id": author["author_id"], "name": author["name"],
                    "s2_num_papers": author["num_papers"], "s2_num_citations": author["num_citations"], "s2_h_index": author["h_index"],
                    "paper_corpus_id": paper["corpus_id"], "paper_title": paper["title"], "arxiv_id": paper.get("arxiv_id"), "date": paper["date"],
                })
    utils.log(f"Found {len(zero_history)} zero-history author-entries")

    first_timers = [z for z in zero_history if z["s2_num_papers"] <= 1]
    suspicious = sorted(zero_history, key=lambda z: z["s2_h_index"], reverse=True)
    sample = random.sample(first_timers, min(args.num_first_timers, len(first_timers))) + suspicious[:args.num_suspicious]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "zero_history_candidates.json"), "w") as f:
        json.dump({"total_zero_history_entries": len(zero_history), "candidates": sample}, f, indent=2)
    for candidate in sample:
        print(f"{candidate['s2_h_index']:>4} h-idx | {candidate['s2_num_papers']:>4} papers | {candidate['name']} | {candidate['arxiv_id']} | {candidate['paper_title'][:70]}")
    utils.log(f"Saved {len(sample)} candidates to {args.output_dir}/zero_history_candidates.json")


if __name__ == "__main__":
    main()
