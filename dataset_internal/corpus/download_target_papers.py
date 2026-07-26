import os
import argparse

import utils

def download_target_corpus_ids_query_func(start_date, end_date, categories):
    categories_string = "|".join(categories).lower()
    
    query = f"""
    WITH arxiv_ids AS (           -- Table that results from first merge.
        SELECT metadata_id,
            corpus_paper_id,
            categories,
            created
        FROM (
                content_ext.paper_sources  -- This merge associates a S2ORC ID with each row of arXiv metadata.
                INNER JOIN arxiv.metadata ON content_ext.paper_sources.source_id = arxiv.metadata.metadata_id
            )
        WHERE REGEXP_LIKE(lower(categories), '{categories_string}')   -- Keep arXiv papers from desired categories.
    )

    SELECT DISTINCT *
    FROM (
            s2orc_papers.oa_latest        -- Merge with s2orc papers.
            INNER JOIN arxiv_ids ON s2orc_papers.oa_latest.id = arxiv_ids.corpus_paper_id
        ) 
    WHERE created >= DATE('{start_date}')
    AND created < DATE('{end_date}')
    AND lower(content.source.pdf_src) IN ('arxiv')
    """
    return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download target papers' corpus_ids, titles, abstracts, and urls from Athena")
    parser.add_argument("--start_date", type=str, default="2024-10-01", help="Start date for target papers")
    parser.add_argument("--end_date", type=str, default="2025-10-01", help="End date for target papers")
    parser.add_argument("--categories",  nargs="+", default=["cs.CL", "cs.LG", "cs.AI", "cs.ML", "cs.CV", "cs.IR", "cs.NE"])
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for target papers and associated papers")
    args = parser.parse_args()
    
    download_target_papers_query = download_target_corpus_ids_query_func(args.start_date, args.end_date, args.categories)
    target_records = utils.submit_athena_query(download_target_papers_query, "s2orc_papers")
    utils.log(f"Downloaded {len(target_records)} target papers' records from Athena")

    target_records = utils.retain_useful_fields_from_arxiv_records(target_records)
    target_records = utils.add_role(target_records, "target")
    utils.log(f"Retained {len(target_records)} target papers' records with all relevant fields")

    target_records = [record for record in target_records if record["date"] >= args.start_date and record["date"] < args.end_date]
    utils.log(f"Retained {len(target_records)} target papers' records within the date range")

    utils.log(f"Sorting {len(target_records)} target papers' records by date")
    all_papers_records = sorted(target_records, key=lambda x: x["date"])

    # Convert to dict and convert back to deduplicate corpus ID
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers_records}
    all_papers_records = list(all_papers_dict.values())
    utils.log(f"Retained {len(all_papers_records)} unique target papers after deduplication by corpus ID")
    
    all_papers_path = os.path.join(args.output_dir, "all_papers.stage01.json")
    utils.save_json(all_papers_records, all_papers_path, utils.update_metadata([], args))
    