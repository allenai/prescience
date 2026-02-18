"""Dataset creation utilities for coauthor prediction task."""

import random
from bisect import bisect_left
from tqdm import tqdm


def get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict):
    """Get a single author's publications before cutoff_date using binary search."""
    author_pubs = sd2publications[author_id]
    if author_pubs is None:
        return []

    author_pubs = [p for p in author_pubs if p in all_papers_dict]
    pub_dates = [all_papers_dict[p]["date"] for p in author_pubs]
    idx = bisect_left(pub_dates, cutoff_date)
    return author_pubs[:idx]


def get_preexisting_publications_for_corpus(cutoff_date, sd2publications, all_papers_dict):
    """Get all authors' publications before cutoff_date."""
    result = {}
    for author_id in tqdm(sd2publications, desc="Building publication histories"):
        result[author_id] = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
    return result


def select_seed_author(author_ids, paper, seed_author_type):
    """Select seed author based on strategy."""
    if seed_author_type == "first":
        return author_ids[0]
    elif seed_author_type == "last":
        return author_ids[-1]
    elif seed_author_type == "random":
        return random.choice(author_ids)
    elif seed_author_type == "highest_h_index":
        author_h_indices = {a["author_id"]: a["h_index"] for a in paper["authors"]}
        return max(author_ids, key=lambda aid: author_h_indices.get(aid, 0))
    else:
        raise ValueError(f"Unknown seed_author_type: {seed_author_type}")


def create_evaluation_instances(all_papers, sd2publications, all_papers_dict, seed_author_type="first"):
    """Create chronologically-sorted evaluation instances for coauthor prediction.

    seed_author_type: "first", "last", "random", "highest_h_index"
    """

    def has_prior_pubs(author_id, date):
        return len(get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict)) > 0

    def get_authors_with_prior_pubs(paper):
        return [a["author_id"] for a in paper["authors"] if has_prior_pubs(a["author_id"], paper["date"])]

    candidate_papers = [p for p in all_papers if "target" in p["roles"] and len(p["authors"]) >= 2]
    candidate_papers_sorted = sorted(candidate_papers, key=lambda p: p["date"])

    instances = []
    for paper in tqdm(candidate_papers_sorted, desc="Creating evaluation instances"):
        author_ids = get_authors_with_prior_pubs(paper)
        if len(author_ids) >= 2:
            seed_author_id = select_seed_author(author_ids, paper, seed_author_type)
            gt_coauthor_ids = [aid for aid in author_ids if aid != seed_author_id]
            instances.append((paper["date"], {
                "corpus_id": paper["corpus_id"],
                "first_author_id": seed_author_id,
                "gt_coauthor_ids": gt_coauthor_ids,
            }))

    return instances
