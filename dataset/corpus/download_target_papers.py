"""Download target papers from local arXiv snapshot and S2 API."""
import os
import argparse
import datetime

import utils


def parse_arxiv_created_date(date_str):
    """Parse arXiv version created date (RFC 2822 format) to YYYY-MM-DD string."""
    dt = datetime.datetime.strptime(date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z")
    return dt.strftime("%Y-%m-%d")


def filter_arxiv_snapshot(arxiv_snapshot, start_date, end_date, categories):
    """Filter arXiv snapshot to papers in target categories and date range."""
    categories_lower = {c.lower() for c in categories}
    target_papers = []
    for arxiv_id, record in arxiv_snapshot.items():
        paper_categories = record["categories"].strip().split()
        if not any(cat.lower() in categories_lower for cat in paper_categories):
            continue
        date = parse_arxiv_created_date(record["versions"][0]["created"])
        if date < start_date or date >= end_date:
            continue
        target_papers.append({
            "arxiv_id": arxiv_id,
            "date": date,
            "categories": paper_categories,
            "title": record["title"].strip().replace("\n", " "),
            "abstract": record["abstract"].strip().replace("\n", " "),
        })
    utils.log(f"Filtered {len(target_papers)} target papers from arXiv snapshot")
    return target_papers


def lookup_corpus_ids(target_papers):
    """Look up S2 corpus IDs for arXiv papers via batch API."""
    s2_ids = [f"ARXIV:{p['arxiv_id']}" for p in target_papers]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "externalIds"],
        batch_size=500,
    )
    arxiv_to_corpus = {}
    for record in records:
        if record is None or record.get("corpusId") is None:
            continue
        external_ids = record.get("externalIds") or {}
        if external_ids.get("ArXiv") is None:
            continue
        arxiv_to_corpus[external_ids["ArXiv"]] = str(record["corpusId"])
    return arxiv_to_corpus


def main():
    parser = argparse.ArgumentParser("Download target papers from arXiv snapshot and S2 API")
    parser.add_argument("--arxiv_snapshot_path", type=str, default="data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json", help="Path to arXiv metadata snapshot")
    parser.add_argument("--start_date", type=str, default="2024-10-01", help="Start date for target papers (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-10-01", help="End date for target papers (YYYY-MM-DD)")
    parser.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG", "cs.AI", "cs.CV", "cs.IR", "cs.NE"], help="arXiv categories to include")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for target papers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    arxiv_snapshot = utils.load_arxiv_snapshot(args.arxiv_snapshot_path)
    target_papers = filter_arxiv_snapshot(arxiv_snapshot, args.start_date, args.end_date, args.categories)

    arxiv_to_corpus = lookup_corpus_ids(target_papers)
    utils.log(f"Found corpus IDs for {len(arxiv_to_corpus)} / {len(target_papers)} target papers")

    target_records = []
    for paper in target_papers:
        if paper["arxiv_id"] not in arxiv_to_corpus:
            continue
        paper["corpus_id"] = arxiv_to_corpus[paper["arxiv_id"]]
        paper["roles"] = ["target"]
        target_records.append(paper)

    target_records = sorted(target_records, key=lambda x: x["date"])
    all_papers_dict = {paper["corpus_id"]: paper for paper in target_records}
    all_papers_records = list(all_papers_dict.values())
    utils.log(f"Retained {len(all_papers_records)} unique target papers after deduplication")

    all_papers_path = os.path.join(args.output_dir, "all_papers.stage01.json")
    utils.save_json(all_papers_records, all_papers_path, utils.update_metadata([], args), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
