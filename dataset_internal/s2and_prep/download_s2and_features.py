import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import utils
from dataset_internal.corpus.add_key_references import key_references_tuples_to_dict as references_tuples_to_dict

def authors_query_func(author_ids):
    query = f"""
        SELECT * FROM content_ext.authors WHERE ai2_id IN ({', '.join(map(str, author_ids))});
    """
    return query

def papers_query_func(corpus_ids):
    query = f"""
        SELECT * FROM content_ext.papers WHERE corpus_paper_id IN ({', '.join(map(str, corpus_ids))});
    """
    return query

def specter1_query_func(corpus_ids):
    query = f"""
        SELECT * FROM paper_embeddings.paper_embeddings_prod WHERE paper_id IN ({', '.join(map(str, corpus_ids))});
    """
    return query

def get_block(author_record: dict):
    first_name_first_letter = "_" if ("first_name" not in author_record) or (author_record["first_name"] is None) or (len(author_record["first_name"]) == 0) else author_record["first_name"].lower()[0]
    last_name = "_" if ("last_name" not in author_record) or (author_record["last_name"] is None) or (len(author_record["last_name"]) == 0) else author_record["last_name"].lower()
    return first_name_first_letter + " " + last_name

def retrieve_references_corpus_ids_query_func(citing_corpusids: list):
    citing_corpus_ids_string = ", ".join(citing_corpusids)
    query = f"""
        SELECT DISTINCT 
            citingpaper.corpus_id AS citing_corpus_paperid, 
            citedpaper.corpus_id AS cited_corpus_paperid
        FROM "citations_rds_export"."citation"
        WHERE citingpaper.corpus_id IN ({citing_corpus_ids_string})
    """
    return query

def specter_records_to_tuple(specter_records):
    specter_dict = {}
    for record in tqdm(specter_records, desc="Converting SPECTER1 records to tuple"):
        paper_id = str(record["paper_id"])
        embedding = np.array(json.loads(record["embedding"]))
        specter_dict[paper_id] = embedding
    embeddings_array = np.array(list(specter_dict.values()))
    corpus_ids_array = np.array(list(specter_dict.keys()))
    return embeddings_array, corpus_ids_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S2AND features for target papers and associated papers.")
    parser.add_argument("--input_dirs", type=str, nargs="+", default=["data/corpus/train", "data/corpus/test"], help="Input directories for target papers and associated papers")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for Athena queries")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of workers for Athena queries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--output_dir", type=str, default="data/corpus/s2and_prescience", help="Output directory for target papers and associated papers")
    args = parser.parse_args()

    all_papers = []
    metadata_entries = []
    for input_dir in args.input_dirs:
        records, metadata = utils.load_json(os.path.join(input_dir, "all_papers.stage03.json"))
        all_papers.extend(records)
        metadata_entries.extend(metadata)
    all_papers = [p for p in all_papers if "authors" in p]
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers} # deduplicates

    author_ids = set()
    for p in all_papers:
        if "authors" in p:
            author_ids.update(a["author_id"] for a in p["authors"])
    corpus_ids = list(set(all_papers_dict.keys()))
    author_ids = list(author_ids)
    
    utils.log(f"Found {len(corpus_ids)} unique corpus IDs in target papers.")
    utils.log(f"Found {len(author_ids)} unique author IDs in target papers.")

    paper_records = utils.submit_athena_queries_batched(
        corpus_ids, 
        papers_query_func,
        "content_ext",
        args.batch_size,
        args.max_workers,
        "Retrieving S2AND papers from Athena"
    )
    paper_records = [p for p in paper_records if p is not None]
    
    author_records = utils.submit_athena_queries_batched(
        author_ids, 
        authors_query_func, 
        "content_ext",
        args.batch_size,
        args.max_workers,
        "Retrieving S2AND authors from Athena"
    )
    author_records = [a for a in author_records if a is not None]
    author_records_dict = {str(a["ai2_id"]): a for a in author_records if a["ai2_id"] is not None}
    
    citation_records = utils.submit_athena_queries_batched(
        corpus_ids,
        retrieve_references_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers,
        "Retrieving citation records from Athena"
    )
    references_dict = references_tuples_to_dict(citation_records)
    
    specter1_records = utils.submit_athena_queries_batched(
        corpus_ids,
        specter1_query_func,
        "paper_embeddings",
        args.batch_size,
        args.max_workers,
        "Retrieving SPECTER1 embeddings from Athena"
    )
    specter1_tuple = specter_records_to_tuple(specter1_records)
    
    papers_json_dict = {}
    signatures_json_dict = {}
    for paper in tqdm(paper_records, desc="Creating S2AND features dicts"):
        papers_json_dict[str(paper["corpus_paper_id"])] = {
            "paper_id": paper["corpus_paper_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "journal_name": paper["journal_name"],
            "venue": paper["venue"],
            "year": paper["year"],
            "references": references_dict[str(paper["corpus_paper_id"])] if str(paper["corpus_paper_id"]) in references_dict else [],
            "authors": [ 
                {
                    "position": idx,
                    "author_name": a["name"]
                } for idx, a in enumerate(all_papers_dict[str(paper["corpus_paper_id"])]["authors"])
            ] if "authors" in all_papers_dict[str(paper["corpus_paper_id"])] else []
        }
    
        for idx, a in enumerate(all_papers_dict[str(paper["corpus_paper_id"])]["authors"]):
            signature_id = f"{paper['corpus_paper_id']}_{a['author_id']}"
            if a["author_id"] in author_records_dict:
                author_record = author_records_dict[a["author_id"]]
                block = get_block(author_record)
                signatures_json_dict[signature_id] = {
                    "author_id": int(a["author_id"]),
                    "paper_id": paper["corpus_paper_id"],
                    "signature_id": signature_id,
                    "author_info": {
                        "given_block": block,
                        "block": block,
                        "position": idx,
                        "first": author_record["first_name"],
                        "middle": None if author_record["middle_names"] is None else json.loads(author_record["middle_names"])[0],
                        "last": author_record["last_name"],
                        "suffix": None,
                        "affiliations": [] if author_record["affiliations"] is None else json.loads(author_record["affiliations"]),
                        "email": author_record["email"]
                    }
                }
            
    utils.save_json(papers_json_dict, os.path.join(args.output_dir, "papers.json"), overwrite=args.overwrite)
    utils.save_json(signatures_json_dict, os.path.join(args.output_dir, "signatures.json"), overwrite=args.overwrite)
    utils.save_pkl(specter1_tuple, os.path.join(args.output_dir, "specter.pickle"), overwrite=args.overwrite, protocol=4)
    