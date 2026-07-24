"""Prepare PreScience corpus for HuggingFace upload."""
import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

import utils

def convert_json_to_parquet(input_path, output_path):
    """Convert JSON to Parquet with explicit nullable schema."""
    utils.log(f"Loading papers from {input_path}")
    papers, _ = utils.load_json(input_path)

    utils.log(f"Defining Parquet schema with nullable fields")
    schema = pa.schema([
        ('corpus_id', pa.string()),
        ('arxiv_id', pa.string()),
        ('date', pa.string()),
        ('title', pa.string()),
        ('abstract', pa.string()),
        ('categories', pa.list_(pa.string())),
        ('topics', pa.list_(pa.string())),
        ('roles', pa.list_(pa.string())),
        ('key_references', pa.list_(pa.struct([
            ('corpus_id', pa.string()),
            ('num_citations', pa.int64()),
        ]))),
        ('authors', pa.list_(pa.struct([
            ('author_id', pa.string()),
            ('name', pa.string()),
            ('publication_history', pa.list_(pa.string())),
            ('h_index', pa.int64()),
            ('num_papers', pa.int64()),
            ('num_citations', pa.int64()),
        ]))),
        ('citation_trajectory', pa.list_(pa.int64())),
    ])

    utils.log(f"Converting {len(papers)} papers to Arrow table")
    table = pa.Table.from_pylist(papers, schema=schema)

    utils.log(f"Writing Parquet file to {output_path}")
    pq.write_table(table, output_path, row_group_size=50000)

def convert_dict_to_jsonl(data_dict, output_path):
    """Convert dict to JSONL format (one entry per line)."""
    with open(output_path, 'w') as f:
        for key, value in data_dict.items():
            entry = {'key': key, 'value': value}
            f.write(json.dumps(entry) + '\n')

def union_per_key(train_map, test_map):
    """Per-key order-preserving union of two dicts whose values are list-or-None."""
    merged = {}
    for sd in set(train_map) | set(test_map):
        train_v = train_map.get(sd)
        test_v = test_map.get(sd)
        if train_v is None and test_v is None:
            merged[sd] = None
        elif train_v is None:
            merged[sd] = test_v
        elif test_v is None:
            merged[sd] = train_v
        else:
            seen, out = set(), []
            for x in train_v + test_v:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            merged[sd] = out
    return merged

def main():
    parser = argparse.ArgumentParser("Prepare PreScience corpus for HuggingFace upload")
    parser.add_argument("--train_dir", type=str, default="data/corpus/train")
    parser.add_argument("--test_dir", type=str, default="data/corpus/test")
    parser.add_argument("--output_dir", type=str, default="data/huggingface")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    utils.log("Converting train corpus to Parquet")
    convert_json_to_parquet(
        os.path.join(args.train_dir, "all_papers.json"),
        os.path.join(args.output_dir, "train.parquet")
    )

    utils.log("Converting test corpus to Parquet")
    convert_json_to_parquet(
        os.path.join(args.test_dir, "all_papers.json"),
        os.path.join(args.output_dir, "test.parquet")
    )

    utils.log("Merging and converting author mappings to JSONL")
    train_sd2og, _ = utils.load_json(os.path.join(args.train_dir, "sd2og.json"))
    test_sd2og, _ = utils.load_json(os.path.join(args.test_dir, "sd2og.json"))
    train_sd2pubs, _ = utils.load_json(os.path.join(args.train_dir, "sd2publications.json"))
    test_sd2pubs, _ = utils.load_json(os.path.join(args.test_dir, "sd2publications.json"))

    all_sd2og = union_per_key(train_sd2og, test_sd2og)
    all_sd2pubs = union_per_key(train_sd2pubs, test_sd2pubs)

    utils.log(f"Merged author mappings: {len(all_sd2og)} unique authors (per-key union across splits)")

    convert_dict_to_jsonl(all_sd2og, os.path.join(args.output_dir, "author_disambiguation.jsonl"))
    convert_dict_to_jsonl(all_sd2pubs, os.path.join(args.output_dir, "author_publications.jsonl"))

    utils.log("Preparation complete!")
    utils.log(f"Files ready in {args.output_dir}:")
    utils.log("  - train.parquet (373K papers)")
    utils.log("  - test.parquet (465K papers)")
    utils.log("  - author_disambiguation.jsonl (S2AND ID → original IDs)")
    utils.log("  - author_publications.jsonl (S2AND ID → corpus IDs)")

if __name__ == "__main__":
    main()