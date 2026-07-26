import os
import argparse
from tqdm import tqdm

import utils
from dataset_internal.corpus.add_key_references import (
    download_arxiv_papers_by_corpus_ids_query_func,
    retrieve_key_references_corpus_ids_query_func,
    key_references_tuples_to_dict,
)


def retrieve_paper_authors_query_func(corpus_ids):
    """Return Athena SQL for retrieving author rosters for the provided corpus IDs."""
    corpus_ids_string = ", ".join(map(str, corpus_ids))
    query = f"""
        SELECT DISTINCT pa.corpus_paper_id AS corpus_paper_id,
               a.ai2_id AS ai2_id,
               a.first_name AS first_name,
               a.last_name AS last_name,
               a.middle_names AS middle_names,
               pa.position AS author_position
        FROM content_ext.paper_authors pa
        JOIN content_ext.authors a
          ON pa.corpus_author_id = a.corpus_author_id
        WHERE pa.corpus_paper_id IN ({corpus_ids_string})
    """
    return query


def retrieve_author_publications_query_func(author_ids):
    """Return Athena SQL for retrieving publication corpus IDs for the provided author IDs."""
    author_ids_string = ", ".join(map(str, author_ids))
    query = f"""
        SELECT DISTINCT a.ai2_id AS ai2_id, pa.corpus_paper_id AS corpus_paper_id
        FROM content_ext.authors a
        JOIN content_ext.paper_authors pa
          ON a.corpus_author_id = pa.corpus_author_id
        WHERE a.ai2_id IN ({author_ids_string})
    """
    return query


def render_author_name(author_record):
    """Render full author name from Athena author record."""
    parts = []
    if author_record["first_name"]:
        parts.append(author_record["first_name"])
    if author_record["middle_names"]:
        parts.append(author_record["middle_names"])
    if author_record["last_name"]:
        parts.append(author_record["last_name"])
    return " ".join(parts)


def download_and_attach_authors_to_papers(all_papers_dict, role, args):
    """Download author rosters from Athena and attach to papers with the given role."""
    corpus_ids = [corpus_id for corpus_id, p in all_papers_dict.items() if role in p["roles"]]
    utils.log(f"Identified {len(corpus_ids)} {role} papers to download author rosters for")

    records = utils.submit_athena_queries_batched(
        corpus_ids,
        retrieve_paper_authors_query_func,
        "content_ext",
        args.batch_size,
        args.max_workers,
    )
    utils.log(f"Downloaded {len(records)} author records from Athena")

    # Filter out records with null IDs and convert to strings
    valid_records = []
    null_count = 0
    for record in records:
        if record["ai2_id"] is not None and record["corpus_paper_id"] is not None:
            record["ai2_id"] = str(record["ai2_id"])
            record["corpus_paper_id"] = str(record["corpus_paper_id"])
            valid_records.append(record)
        else:
            null_count += 1
    utils.log(f"Retained {len(valid_records)} author records with non-null IDs (dropped {null_count} with null IDs)")

    # Organize by corpus_id
    corpus_id_to_authors = {}
    for record in valid_records:
        corpus_id = record["corpus_paper_id"]
        author_entry = {
            "author_id": record["ai2_id"],
            "name": render_author_name(record),
            "author_position": record["author_position"],
        }
        if corpus_id not in corpus_id_to_authors:
            corpus_id_to_authors[corpus_id] = []
        corpus_id_to_authors[corpus_id].append(author_entry)

    # Sort authors by position and remove the temporary field
    for corpus_id in corpus_id_to_authors:
        corpus_id_to_authors[corpus_id] = sorted(
            corpus_id_to_authors[corpus_id],
            key=lambda x: x["author_position"] if x["author_position"] is not None else 9999
        )
        for author in corpus_id_to_authors[corpus_id]:
            del author["author_position"]

    distinct_author_ids = {author["author_id"] for authors in corpus_id_to_authors.values() for author in authors}
    utils.log(f"Found {len(distinct_author_ids)} distinct authors across {len(corpus_id_to_authors)} {role} papers ({len(corpus_ids) - len(corpus_id_to_authors)} papers have no authors)")

    # Attach to papers (skip if authors already exist to avoid overwriting publication_history)
    for corpus_id, authors in corpus_id_to_authors.items():
        if "authors" not in all_papers_dict[corpus_id]:
            all_papers_dict[corpus_id]["authors"] = authors

    return all_papers_dict


def download_publication_histories_for_target_authors(all_papers_dict, args):
    """Download publication history corpus IDs for all authors of target papers."""
    all_author_ids = set()
    for paper in all_papers_dict.values():
        if "target" in paper["roles"] and "authors" in paper:
            for author in paper["authors"]:
                all_author_ids.add(author["author_id"])

    utils.log(f"Identified {len(all_author_ids)} distinct target paper authors to download publication histories for")

    records = utils.submit_athena_queries_batched(
        list(all_author_ids),
        retrieve_author_publications_query_func,
        "content_ext",
        args.batch_size,
        args.max_workers,
    )
    utils.log(f"Downloaded {len(records)} author-publication records from Athena")

    # Filter out records with null IDs and organize by author_id
    author_id_to_corpus_ids = {}
    null_count = 0
    for record in records:
        if record["ai2_id"] is None or record["corpus_paper_id"] is None:
            null_count += 1
            continue
        author_id = str(record["ai2_id"])
        corpus_id = str(record["corpus_paper_id"])
        if author_id not in author_id_to_corpus_ids:
            author_id_to_corpus_ids[author_id] = []
        author_id_to_corpus_ids[author_id].append(corpus_id)

    total_pubs = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    utils.log(f"Found {total_pubs} publication corpus IDs for {len(author_id_to_corpus_ids)} authors ({len(all_author_ids) - len(author_id_to_corpus_ids)} authors have no publications, dropped {null_count} records with null IDs)")
    return author_id_to_corpus_ids


def download_publication_history_papers(author_id_to_corpus_ids, args):
    """Download publication history papers from Athena and refine author_id_to_corpus_ids."""
    all_pub_corpus_ids = {
        corpus_id
        for corpus_ids in author_id_to_corpus_ids.values()
        for corpus_id in corpus_ids
    }
    utils.log(f"Identified {len(all_pub_corpus_ids)} unique publication history corpus IDs to download")

    records = utils.submit_athena_queries_batched(
        list(all_pub_corpus_ids),
        download_arxiv_papers_by_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers,
    )
    utils.log(f"Downloaded {len(records)} publication history papers from Athena")
    num_before_filter = len(records)
    records = utils.retain_useful_fields_from_arxiv_records(records)
    records = utils.add_role(records, "target.author.publication_history")
    utils.log(f"Retained {len(records)} publication history papers with all relevant fields (dropped {num_before_filter - len(records)} missing fields, {len(all_pub_corpus_ids) - num_before_filter} not on arxiv)")

    pub_history_dict = {paper["corpus_id"]: paper for paper in records}

    # Refine author_id_to_corpus_ids to only include corpus_ids we found abstracts for
    total_before = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    for author_id in author_id_to_corpus_ids:
        author_id_to_corpus_ids[author_id] = [
            corpus_id for corpus_id in author_id_to_corpus_ids[author_id]
            if corpus_id in pub_history_dict
        ]
    total_after = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    utils.log(f"Refined author publication lists from {total_before} to {total_after} entries with available abstracts")

    return author_id_to_corpus_ids, pub_history_dict


def attach_and_insert_publication_histories(all_papers_dict, author_id_to_corpus_ids, pub_history_dict):
    """Attach publication histories to target authors and insert papers into corpus."""
    utils.log("Attaching publication histories to target paper authors")

    total_attached = 0
    authors_with_empty_history = 0
    for paper in tqdm(all_papers_dict.values(), desc="Attaching publication histories"):
        if "target" in paper["roles"]:
            for author in paper["authors"]:
                author_id = author["author_id"]
                corpus_ids = author_id_to_corpus_ids.get(author_id, [])

                # Keep only publications that predate the target
                valid_corpus_ids = [
                    corpus_id for corpus_id in corpus_ids
                    if pub_history_dict[corpus_id]["date"] < paper["date"]
                ]
                valid_corpus_ids.sort(key=lambda corpus_id: pub_history_dict[corpus_id]["date"])
                author["publication_history"] = valid_corpus_ids
                total_attached += len(valid_corpus_ids)
                if len(valid_corpus_ids) == 0:
                    authors_with_empty_history += 1

    utils.log(f"Attached {total_attached} publication history entries ({authors_with_empty_history} authors have empty histories due to date truncation or no prior publications)")

    # Insert publication history papers into corpus
    corpus_size_before = len(all_papers_dict)
    for corpus_id, paper in pub_history_dict.items():
        if corpus_id in all_papers_dict:
            if "target.author.publication_history" not in all_papers_dict[corpus_id]["roles"]:
                all_papers_dict[corpus_id]["roles"].append("target.author.publication_history")
        else:
            all_papers_dict[corpus_id] = paper

    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new publication history papers into corpus (corpus now contains {len(all_papers_dict)} papers)")
    return all_papers_dict


def download_and_attach_key_references_and_prune(all_papers_dict, args):
    """Download key references for publication history papers and attach them. Also prunes publication history papers exceeding max_key_references and cleans up the corpus."""
    pub_history_corpus_ids = {
        corpus_id for corpus_id, paper in all_papers_dict.items()
        if "target.author.publication_history" in paper["roles"]
    }
    utils.log(f"Identified {len(pub_history_corpus_ids)} publication history papers to download key references for")

    key_ref_tuples = utils.submit_athena_queries_batched(
        list(pub_history_corpus_ids),
        retrieve_key_references_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers,
    )
    utils.log(f"Downloaded {len(key_ref_tuples)} key reference tuples from Athena")

    key_ref_dict = key_references_tuples_to_dict(key_ref_tuples)
    all_key_ref_ids = {corpus_id for refs in key_ref_dict.values() for corpus_id in refs}
    utils.log(f"Found {len(all_key_ref_ids)} unique key reference corpus IDs across {len(key_ref_dict)} papers ({len(pub_history_corpus_ids) - len(key_ref_dict)} papers have no key references)")

    records = utils.submit_athena_queries_batched(
        list(all_key_ref_ids),
        download_arxiv_papers_by_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers,
    )
    utils.log(f"Downloaded {len(records)} key reference papers from Athena")
    num_before_filter = len(records)
    records = utils.retain_useful_fields_from_arxiv_records(records)
    records = utils.add_role(records, "target.author.publication_history.key_reference")
    utils.log(f"Retained {len(records)} key reference papers with all relevant fields (dropped {num_before_filter - len(records)} missing fields, {len(all_key_ref_ids) - num_before_filter} not on arxiv)")

    key_ref_papers_dict = {p["corpus_id"]: p for p in records}

    # Insert key reference papers into corpus
    corpus_size_before = len(all_papers_dict)
    for corpus_id, paper in key_ref_papers_dict.items():
        if corpus_id in all_papers_dict:
            if "target.author.publication_history.key_reference" not in all_papers_dict[corpus_id]["roles"]:
                all_papers_dict[corpus_id]["roles"].append("target.author.publication_history.key_reference")
        else:
            all_papers_dict[corpus_id] = paper

    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new key reference papers into corpus (corpus now contains {len(all_papers_dict)} papers)")

    # Attach key references to publication history papers (skip if already has key_references to preserve target key_refs)
    for corpus_id in tqdm(pub_history_corpus_ids, desc="Attaching key references"):
        paper = all_papers_dict[corpus_id]
        if "key_references" not in paper:
            ref_ids = key_ref_dict[corpus_id] if corpus_id in key_ref_dict else []
            valid_refs = [
                {"corpus_id": ref_id}
                for ref_id in ref_ids
                if ref_id in key_ref_papers_dict and key_ref_papers_dict[ref_id]["date"] < paper["date"]
            ]
            paper["key_references"] = valid_refs

    num_with_refs = len([c for c in pub_history_corpus_ids if all_papers_dict[c]["key_references"]])
    utils.log(f"{num_with_refs} out of {len(pub_history_corpus_ids)} publication history papers have at least one key reference")

    # Prune publication history papers exceeding max_key_references
    pub_history_before = len(pub_history_corpus_ids)
    exceeding_corpus_ids = {
        corpus_id for corpus_id in pub_history_corpus_ids
        if len(all_papers_dict[corpus_id]["key_references"]) > args.max_key_references
    }
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if corpus_id not in exceeding_corpus_ids
    }
    utils.log(f"Pruned {len(exceeding_corpus_ids)} publication history papers exceeding {args.max_key_references} key references ({pub_history_before - len(exceeding_corpus_ids)} remain)")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)

    return all_papers_dict


def prune_papers_without_authors_and_clean_references(all_papers_dict, role):
    """Remove papers of the given role that have no authors, then clean up references."""
    corpus_size_before = len(all_papers_dict)
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if role not in paper["roles"] or ("authors" in paper and paper["authors"])
    }
    utils.log(f"Pruned {corpus_size_before - len(all_papers_dict)} {role} papers without authors (corpus now contains {len(all_papers_dict)} papers)")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)

    return all_papers_dict


def main():
    """Add authors and publication histories to target papers."""
    parser = argparse.ArgumentParser(description="Add authors and publication histories to corpus papers")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test")
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--max_workers", type=int, default=50)
    parser.add_argument("--max_key_references", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage02.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage03.json")

    utils.log(f"Loading {input_path}")
    all_papers, metadata = utils.load_json(input_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Add authors to target papers
    all_papers_dict = download_and_attach_authors_to_papers(all_papers_dict, "target", args)
    all_papers_dict = prune_papers_without_authors_and_clean_references(all_papers_dict, "target")

    # Download and attach publication histories
    author_id_to_corpus_ids = download_publication_histories_for_target_authors(all_papers_dict, args)
    author_id_to_corpus_ids, pub_history_dict = download_publication_history_papers(author_id_to_corpus_ids, args)
    utils.save_json(author_id_to_corpus_ids, os.path.join(args.output_dir, "author_id_to_corpus_ids.json"), utils.update_metadata(metadata, args), overwrite=args.overwrite)  # sidecar for merge_authors_identity (no-S2AND path)
    all_papers_dict = attach_and_insert_publication_histories(all_papers_dict, author_id_to_corpus_ids, pub_history_dict)

    # Download and attach key references
    all_papers_dict = download_and_attach_key_references_and_prune(all_papers_dict, args)

    # Add authors to publication history papers
    all_papers_dict = download_and_attach_authors_to_papers(all_papers_dict, "target.author.publication_history", args)
    all_papers_dict = prune_papers_without_authors_and_clean_references(all_papers_dict, "target.author.publication_history")

    all_papers = list(all_papers_dict.values())
    utils.save_json(all_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved {len(all_papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
