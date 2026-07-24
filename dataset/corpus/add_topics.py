"""Attach topic labels to target papers in all_papers.json.

Reads a JSONL of `{"corpus_id": "...", "topics": [...]}` records produced by
`assign_topics.py` and adds a `topics` field to each target paper in
`<input_dir>/all_papers.json`. The new field is inserted immediately after
`categories` to keep the canonical schema order. Non-target papers are not
modified. Documented as the final step of corpus creation.
"""
import argparse
import json
import os

import utils


def parse_args():
    parser = argparse.ArgumentParser("Attach topic labels to target papers in all_papers.json.")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Directory containing all_papers.json")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Directory where the topic-augmented all_papers.json will be written")
    parser.add_argument("--topics_path", type=str, default="data/task_topic_growth_prediction/topic_labels.v5.gpt-5.4.full.jsonl", help="JSONL file with corpus_id -> topics mapping")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    return parser.parse_args()


def load_topics(topics_path):
    """Read the JSONL of corpus_id -> list[str] topic labels."""
    utils.log(f"Loading topic labels from {topics_path}")
    topics_by_id = {}
    with open(topics_path) as f:
        for line in f:
            record = json.loads(line)
            topics_by_id[record["corpus_id"]] = record["topics"]
    utils.log(f"Loaded topic labels for {len(topics_by_id)} corpus_ids")
    return topics_by_id


def insert_topics_after_categories(paper, topics):
    """Return a new dict with `topics` placed immediately after `categories`."""
    out = {}
    for key, value in paper.items():
        out[key] = value
        if key == "categories":
            out["topics"] = topics
    return out


def main():
    args = parse_args()
    input_path = os.path.join(args.input_dir, "all_papers.json")
    output_path = os.path.join(args.output_dir, "all_papers.json")

    topics_by_id = load_topics(args.topics_path)

    papers, metadata = utils.load_json(input_path)
    metadata = metadata if metadata is not None else []
    utils.log(f"Loaded {len(papers)} papers from {input_path}")

    n_targets = 0
    n_with_topics = 0
    n_missing_topics = 0
    updated_papers = []
    for paper in papers:
        if "target" in paper["roles"]:
            n_targets += 1
            if paper["corpus_id"] in topics_by_id:
                topics = topics_by_id[paper["corpus_id"]]
                updated_papers.append(insert_topics_after_categories(paper, topics))
                n_with_topics += 1
            else:
                utils.log(f"Warning: no topics found for target {paper['corpus_id']}; leaving topics field absent")
                updated_papers.append(paper)
                n_missing_topics += 1
        else:
            updated_papers.append(paper)
    utils.log(f"Attached topics to {n_with_topics}/{n_targets} target papers ({n_missing_topics} missing)")

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.islink(output_path):
        utils.log(f"Removing existing symlink at {output_path} before writing real file")
        os.unlink(output_path)
    utils.save_json(updated_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Wrote topic-augmented corpus to {output_path}")


if __name__ == "__main__":
    main()
