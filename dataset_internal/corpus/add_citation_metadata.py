import os
import datetime
import argparse
from tqdm import tqdm

import utils
from dataset_internal.corpus.add_authors import retrieve_author_publications_query_func


# Query functions

def retrieve_paper_dates_query_func(corpus_ids: list):
    """Return Athena SQL for retrieving publication dates for the provided corpus IDs."""
    corpus_ids_string = ", ".join(corpus_ids)
    query = f"""
        SELECT id, metadata.publication_date
        FROM s2orc_papers.latest
        WHERE id IN ({corpus_ids_string})
    """
    return query


def retrieve_citations_query_func(cited_corpusids: list):
    """Return Athena SQL for retrieving citations for the provided corpus IDs."""
    cited_corpus_ids_string = ", ".join(cited_corpusids)
    query = f"""
        SELECT DISTINCT citing_corpus_paperid, cited_corpus_paperid, inserted
        FROM content_ext.citations
        WHERE cited_corpus_paperid IN ({cited_corpus_ids_string})
    """
    return query


# Data processing

def build_citations_dict(records):
    """Build dict mapping cited_corpus_id -> list of (citing_id, date) tuples."""
    citations_dict = {}
    for record in tqdm(records, desc="Building citations dict"):
        citing_id = str(record["citing_corpus_paperid"])
        cited_id = str(record["cited_corpus_paperid"])
        date = str(record["inserted"])[:10]
        if (citing_id != "None") and (cited_id != "None"):
            if cited_id not in citations_dict:
                citations_dict[cited_id] = []
            citations_dict[cited_id].append((citing_id, date))
    return citations_dict


def build_sd2publications(records, og2sd):
    """Build dict mapping S2AND author_id -> set of corpus_ids."""
    og2publications = {}
    for record in records:
        og_id = str(record["ai2_id"])
        corpus_id = str(record["corpus_paper_id"])
        if og_id not in og2publications:
            og2publications[og_id] = set()
        og2publications[og_id].add(corpus_id)

    sd2publications = {}
    for og_id, corpus_ids in og2publications.items():
        sd_id = og2sd[og_id]
        if sd_id not in sd2publications:
            sd2publications[sd_id] = set()
        sd2publications[sd_id].update(corpus_ids)

    return sd2publications


def build_paper_dates_dict(records):
    """Build dict mapping corpus_id -> publication date (YYYY-MM-DD)."""
    paper_dates = {}
    for record in tqdm(records, desc="Building paper dates dict"):
        corpus_id = str(record["id"])
        pub_date_dict = record["publication_date"]
        if pub_date_dict is not None:
            year = int(pub_date_dict["year"]) if pub_date_dict["year"] is not None else 9999
            month = int(pub_date_dict["month"]) if pub_date_dict["month"] is not None else 1
            day = int(pub_date_dict["day"]) if pub_date_dict["day"] is not None else 1
            paper_dates[corpus_id] = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    return paper_dates


# Citation metrics

def get_num_citations_before(corpus_id, cutoff_date, citations_dict):
    """Count citations to a paper that occurred before cutoff_date."""
    citations = citations_dict[corpus_id] if corpus_id in citations_dict else []
    return sum(1 for _, date in citations if date < cutoff_date)


def get_h_index(pub_corpus_ids, cutoff_date, citations_dict):
    """Calculate h-index from publication citation counts before cutoff_date."""
    counts = sorted(
        [get_num_citations_before(pub_id, cutoff_date, citations_dict) for pub_id in pub_corpus_ids],
        reverse=True
    )
    h = 0
    for i, c in enumerate(counts):
        if c >= i + 1:
            h = i + 1
    return h


def get_citation_trajectory(corpus_id, pub_date, end_date, citations_dict, max_months=36):
    """Get monthly cumulative citation counts from pub_date to end_date (up to max_months)."""
    pub_dt = datetime.datetime.strptime(pub_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    months_available = min(max_months, int((end_dt - pub_dt).days / 30))

    trajectory = [
        get_num_citations_before(corpus_id, (pub_dt + datetime.timedelta(days=month * 30)).strftime("%Y-%m-%d"), citations_dict)
        for month in range(1, months_available + 1)
    ]

    return trajectory if trajectory else [0]


# Main pipeline

def attach_citation_metadata(all_papers_dict, sd2publications, citations_dict, paper_dates):
    """Attach citation metadata to target papers."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    target_corpus_ids = [corpus_id for corpus_id, paper in all_papers_dict.items() if "target" in paper["roles"]]

    for target_id in tqdm(target_corpus_ids, desc="Attaching citation metadata"):
        paper = all_papers_dict[target_id]
        pub_date = paper["date"]

        # Add num_citations to each key reference
        for ref in paper["key_references"]:
            ref["num_citations"] = get_num_citations_before(ref["corpus_id"], pub_date, citations_dict)

        # Add author metrics
        for author in paper["authors"]:
            author_id = author["author_id"]
            author_pubs = sd2publications[author_id] if author_id in sd2publications else set()
            author_pubs_before = [
                pub_id for pub_id in author_pubs
                if pub_id in paper_dates and paper_dates[pub_id] < pub_date
            ]
            # O(10) authors have len(author_pubs_before) < len(author["publication_history"]) due to missing paper dates, recent pubs, etc.
            author["num_papers"] = max(len(author_pubs_before), len(author["publication_history"]))
            author["num_citations"] = sum(get_num_citations_before(pub_id, pub_date, citations_dict) for pub_id in author_pubs_before)
            author["h_index"] = get_h_index(author_pubs_before, pub_date, citations_dict)

        # Add citation trajectory
        paper["citation_trajectory"] = get_citation_trajectory(target_id, pub_date, today, citations_dict)

    return all_papers_dict


def main():
    """Add citation metadata to corpus papers."""
    parser = argparse.ArgumentParser(description="Add citation metadata to corpus")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test")
    parser.add_argument("--batch_size", type=int, default=20000)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage04.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage05.json")

    # Load corpus and author mappings
    all_papers, metadata = utils.load_json(input_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    sd2og, _ = utils.load_json(os.path.join(args.input_dir, "sd2og.json"))
    og2sd, _ = utils.load_json(os.path.join(args.input_dir, "og2sd.json"))

    # Collect IDs needed for queries
    target_ids = set()
    key_ref_ids = set()
    og_author_ids = set()
    for corpus_id, paper in all_papers_dict.items():
        if "target" in paper["roles"]:
            target_ids.add(corpus_id)
            for ref in paper["key_references"]:
                key_ref_ids.add(ref["corpus_id"])
            for author in paper["authors"]:
                og_author_ids.update(sd2og[author["author_id"]])
    utils.log(f"Collected {len(target_ids)} targets, {len(key_ref_ids)} key refs, {len(og_author_ids)} OG authors")

    # Download author publication histories
    author_pub_records = utils.submit_athena_queries_batched(
        sorted(og_author_ids),
        retrieve_author_publications_query_func,
        "content_ext",
        args.batch_size,
        args.max_workers,
        "Retrieving author publications",
    )
    sd2publications = build_sd2publications(author_pub_records, og2sd)
    author_pub_ids = set(pub_id for pubs in sd2publications.values() for pub_id in pubs)
    utils.log(f"Downloaded {len(author_pub_ids)} author publications across {len(sd2publications)} authors")

    # Download publication dates for author publications
    paper_date_records = utils.submit_athena_queries_batched(
        author_pub_ids,
        retrieve_paper_dates_query_func,
        "s2orc_papers",
        args.batch_size,
        args.max_workers,
        "Retrieving paper dates",
    )
    paper_dates = build_paper_dates_dict(paper_date_records)
    utils.log(f"Downloaded publication dates for {len(paper_dates)} papers")

    # Overwrite paper_dates with corpus dates (prefer arxiv dates over s2orc_papers.latest)
    for corpus_id, paper in all_papers_dict.items():
        paper_dates[corpus_id] = paper["date"]
    utils.log(f"Augmented to {len(paper_dates)} papers with corpus dates")

    # Download citations
    papers_needing_citations = target_ids | key_ref_ids | author_pub_ids
    citation_records = utils.submit_athena_queries_batched(
        sorted(papers_needing_citations),
        retrieve_citations_query_func,
        "content_ext",
        args.batch_size,
        args.max_workers,
        "Retrieving citations",
    )
    citations_dict = build_citations_dict(citation_records)
    utils.log(f"Downloaded citations for {len(citations_dict)} papers")

    # Attach citation metadata to target papers
    all_papers_dict = attach_citation_metadata(all_papers_dict, sd2publications, citations_dict, paper_dates)

    # Save output
    all_papers = sorted(all_papers_dict.values(), key=lambda x: x["date"])
    updated_metadata = utils.update_metadata(metadata, args)
    utils.save_json(all_papers, output_path, metadata=updated_metadata, overwrite=args.overwrite)
    utils.log(f"Saved {len(all_papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
