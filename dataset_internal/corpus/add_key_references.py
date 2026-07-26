import os
import argparse
from tqdm import tqdm

import utils


def retrieve_key_references_corpus_ids_query_func(citing_corpusids):
    """Return Athena SQL for retrieving key reference corpus IDs for the provided citing corpus IDs."""
    citing_corpus_ids_string = ", ".join(citing_corpusids)
    query = f"""
    SELECT DISTINCT
        citingpaper.corpus_id AS citing_corpus_paperid,
        citedpaper.corpus_id AS cited_corpus_paperid
    FROM "citations_rds_export"."citation"
    WHERE citingpaper.corpus_id IN ({citing_corpus_ids_string})
    AND keycitationtypes IS NOT NULL
    """
    return query


def download_arxiv_papers_by_corpus_ids_query_func(corpus_ids):
    """Return Athena SQL for downloading arxiv papers by corpus IDs."""
    corpus_ids_string = ", ".join(corpus_ids)
    query = f"""
    WITH arxiv_ids AS (
        SELECT metadata_id,
            corpus_paper_id,
            categories,
            created
        FROM (
                content_ext.paper_sources
                INNER JOIN arxiv.metadata ON content_ext.paper_sources.source_id = arxiv.metadata.metadata_id
            )
        WHERE corpus_paper_id IN ({corpus_ids_string})
    )

    SELECT DISTINCT *
    FROM (
            s2orc_papers.oa_latest
            INNER JOIN arxiv_ids ON s2orc_papers.oa_latest.id = arxiv_ids.corpus_paper_id
        )
    """
    return query


def key_references_tuples_to_dict(key_reference_tuples):
    """Convert key reference tuples to a dict mapping citing corpus_id to list of cited corpus_ids."""
    key_reference_dict = {}
    for tup in tqdm(key_reference_tuples, desc="Converting key_reference_tuples to dict"):
        citing_corpus_id = str(tup["citing_corpus_paperid"])
        cited_corpus_id = str(tup["cited_corpus_paperid"])
        if cited_corpus_id != "None":
            if citing_corpus_id not in key_reference_dict:
                key_reference_dict[citing_corpus_id] = []
            key_reference_dict[citing_corpus_id].append(cited_corpus_id)
    # Remove duplicates
    for key in key_reference_dict:
        key_reference_dict[key] = list(set(key_reference_dict[key]))
    return key_reference_dict


def download_key_reference_tuples(all_papers_dict, args):
    """Download key reference tuples for target papers from Athena."""
    target_corpus_ids = [corpus_id for corpus_id, p in all_papers_dict.items() if "target" in p["roles"]]
    utils.log(f"Identified {len(target_corpus_ids)} target papers to download key references for")

    key_ref_tuples = utils.submit_athena_queries_batched(
        target_corpus_ids,
        retrieve_key_references_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers
    )
    utils.log(f"Downloaded {len(key_ref_tuples)} key reference tuples from Athena")

    key_ref_dict = key_references_tuples_to_dict(key_ref_tuples)
    all_cited_corpus_ids = {corpus_id for refs in key_ref_dict.values() for corpus_id in refs}
    utils.log(f"Found {len(all_cited_corpus_ids)} unique cited corpus IDs across {len(key_ref_dict)} target papers ({len(target_corpus_ids) - len(key_ref_dict)} targets have no key references)")

    return key_ref_dict


def download_key_reference_papers_and_refine_tuples(all_papers_dict, key_ref_dict, args):
    """Download key reference papers from Athena and refine key_ref_dict based on availability and date."""
    all_cited_corpus_ids = {corpus_id for refs in key_ref_dict.values() for corpus_id in refs}
    utils.log(f"Downloading {len(all_cited_corpus_ids)} key reference papers from Athena")

    records = utils.submit_athena_queries_batched(
        list(all_cited_corpus_ids),
        download_arxiv_papers_by_corpus_ids_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers
    )
    utils.log(f"Downloaded {len(records)} key reference papers from Athena")
    num_before_filter = len(records)
    records = utils.retain_useful_fields_from_arxiv_records(records)
    records = utils.add_role(records, "target.key_reference")
    utils.log(f"Retained {len(records)} key reference papers (dropped {num_before_filter - len(records)} missing fields, {len(all_cited_corpus_ids) - num_before_filter} not on arxiv)")

    key_ref_papers_dict = {paper["corpus_id"]: paper for paper in records}

    # Refine key_ref_dict: only keep refs that we found AND that predate the target
    for corpus_id in key_ref_dict:
        paper_date = all_papers_dict[corpus_id]["date"]
        key_ref_dict[corpus_id] = [
            ref_id for ref_id in key_ref_dict[corpus_id]
            if ref_id in key_ref_papers_dict and key_ref_papers_dict[ref_id]["date"] < paper_date
        ]
    key_ref_dict = {k: v for k, v in key_ref_dict.items() if len(v) > 0}
    total_refs = sum(len(refs) for refs in key_ref_dict.values())
    utils.log(f"Refined to {total_refs} key references across {len(key_ref_dict)} target papers (filtered by availability and date)")

    return key_ref_dict, key_ref_papers_dict


def attach_and_insert_key_references(all_papers_dict, key_ref_dict, key_ref_papers_dict, args):
    """Attach key references to target papers, filter by max count, and insert papers into corpus."""
    # Filter to targets with at most max_key_references and remove those that don't pass
    key_ref_dict = {k: v for k, v in key_ref_dict.items() if len(v) <= args.max_key_references}
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if "target" not in paper["roles"] or corpus_id in key_ref_dict
    }
    utils.log(f"Retained {len(key_ref_dict)} target papers with at most {args.max_key_references} key references")

    # Attach key references and insert key reference papers
    corpus_size_before = len(all_papers_dict)
    for corpus_id, ref_ids in key_ref_dict.items():
        all_papers_dict[corpus_id]["key_references"] = [{"corpus_id": ref_id} for ref_id in ref_ids]
        for ref_id in ref_ids:
            if ref_id not in all_papers_dict:
                all_papers_dict[ref_id] = key_ref_papers_dict[ref_id]
            elif "target.key_reference" not in all_papers_dict[ref_id]["roles"]:
                all_papers_dict[ref_id]["roles"].append("target.key_reference")

    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new key reference papers (corpus now contains {len(all_papers_dict)} papers)")
    return all_papers_dict


def main():
    """Add key references to target papers."""
    parser = argparse.ArgumentParser(description="Add key references to target papers")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test")
    parser.add_argument("--max_key_references", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage01.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage02.json")

    utils.log(f"Loading {input_path}")
    all_papers, metadata = utils.load_json(input_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Download key reference tuples
    key_ref_dict = download_key_reference_tuples(all_papers_dict, args)

    # Download key reference papers and refine
    key_ref_dict, key_ref_papers_dict = download_key_reference_papers_and_refine_tuples(all_papers_dict, key_ref_dict, args)

    # Attach key references to target papers and insert papers into corpus
    all_papers_dict = attach_and_insert_key_references(all_papers_dict, key_ref_dict, key_ref_papers_dict, args)

    # Remove unreachable papers
    all_papers_dict = utils.remove_unreachable_papers(all_papers_dict)
    utils.log(f"Corpus contains {len(all_papers_dict)} papers after removing unreachable papers")

    all_papers = list(all_papers_dict.values())
    utils.save_json(all_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved {len(all_papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
