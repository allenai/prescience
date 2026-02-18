"""Dataset creation utilities for prior work prediction task."""

from bisect import bisect_left
from tqdm import tqdm


def get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get a single author's publications before cutoff_date using binary search."""
    if author_id not in sd2publications:
        return []

    author_pubs = sd2publications[author_id]
    if author_pubs is None:
        return []

    author_pubs = [p for p in author_pubs if p in all_papers_dict]
    pub_dates = [all_papers_dict[p]["date"] for p in author_pubs]
    idx = bisect_left(pub_dates, cutoff_date)
    return author_pubs[:idx]


def get_cited_references_for_author(author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict):
    """Get papers cited by an author in their recent publications (with key_references) before cutoff_date."""
    author_pubs = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
    pubs_with_refs = [cid for cid in author_pubs if len(all_papers_dict[cid].get("key_references") or []) > 0]
    recent_pubs = pubs_with_refs[-num_recent_papers:]

    cited_corpus_ids = []
    for corpus_id in recent_pubs:
        paper = all_papers_dict[corpus_id]
        for ref in paper["key_references"]:
            cited_corpus_ids.append(ref["corpus_id"])

    return cited_corpus_ids


def create_evaluation_instances(all_papers, sd2publications, all_papers_dict):
    """Create chronologically-sorted evaluation instances for prior work prediction."""

    def has_pub_with_refs(author_id, date):
        author_pubs = get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)
        for corpus_id in author_pubs:
            if len(all_papers_dict[corpus_id].get("key_references") or []) > 0:
                return True
        return False

    def has_author_with_cited_refs(paper):
        for author in paper["authors"]:
            if has_pub_with_refs(author["author_id"], paper["date"]):
                return True
        return False

    # Filter to target papers with at least one author with cited references
    candidate_papers = []
    for paper in all_papers:
        if "target" in paper["roles"] and has_author_with_cited_refs(paper):
            candidate_papers.append(paper)

    candidate_papers_sorted = sorted(candidate_papers, key=lambda p: p["date"])

    instances = []
    for paper in tqdm(candidate_papers_sorted, desc="Creating evaluation instances"):
        gt_reference_ids = [ref["corpus_id"] for ref in paper["key_references"]]
        instances.append((paper["date"], {
            "corpus_id": paper["corpus_id"],
            "author_ids": [a["author_id"] for a in paper["authors"]],
            "gt_reference_ids": gt_reference_ids,
        }))

    return instances
