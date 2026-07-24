"""Autoresearch agent best baseline for coauthor prediction.

Frequency baseline + 2-hop and 3-hop expansion + citation-history pull + activity-decay boost.
"""

import os
import math
import random
import argparse
from collections import Counter
from datetime import datetime
from tqdm import tqdm

import utils
from task_coauthor_prediction.dataset import create_evaluation_instances, get_preexisting_publications_for_author

random.seed(42)


NUM_RECENT_PAPERS = 45
EXPAND_TOP_N = 150
EXPAND_NUM_RECENT = 240
EXPAND_DAMPING = 0.6
CITE_DAMPING = 0.05
CITE_NUM_RECENT_PAPERS = 10
ACTIVITY_HALF_LIFE_YEARS = 2.5
HOP3_TOP_N = 100
HOP3_NUM_RECENT = 160
HOP3_DAMPING = 0.3


def _get_cited_references_for_author(author_id, cutoff_date, num_recent_papers, sd2publications, all_papers_dict):
    """Mirror of priorwork.dataset.get_cited_references_for_author: refs of an author's most-recent N pre-cutoff papers."""
    author_pubs = get_preexisting_publications_for_author(author_id, cutoff_date, sd2publications, all_papers_dict)
    pubs_with_refs = [cid for cid in author_pubs if len(all_papers_dict[cid].get("key_references") or []) > 0]
    recent_pubs = pubs_with_refs[-num_recent_papers:]
    cited_corpus_ids = []
    for corpus_id in recent_pubs:
        for ref in all_papers_dict[corpus_id]["key_references"]:
            cited_corpus_ids.append(ref["corpus_id"])
    return cited_corpus_ids


def predict_coauthors(seed, cutoff_date, k, candidate_author_ids, sd2publications, all_papers_dict):
    seed_pubs = get_preexisting_publications_for_author(seed, cutoff_date, sd2publications, all_papers_dict)
    recent_pubs = seed_pubs[-NUM_RECENT_PAPERS:]
    n = len(recent_pubs)
    coauthor_scores = Counter()
    for i, cid in enumerate(recent_pubs):
        paper = all_papers_dict.get(cid)
        if paper is None:
            continue
        authors = paper.get("authors") or []
        team_factor = 1.0 / (max(len(authors), 1) ** 1.5)
        weight = ((i + 1) / n) * team_factor
        for author in authors:
            aid = author.get("author_id") if isinstance(author, dict) else None
            if aid is None or aid == seed:
                continue
            if aid in candidate_author_ids:
                coauthor_scores[aid] += weight

    top_l1 = [aid for aid, _ in coauthor_scores.most_common(EXPAND_TOP_N)]
    for l1 in top_l1:
        l1_score = coauthor_scores[l1]
        l1_pubs = get_preexisting_publications_for_author(l1, cutoff_date, sd2publications, all_papers_dict)[-EXPAND_NUM_RECENT:]
        m = len(l1_pubs)
        if m == 0:
            continue
        prolific_factor = 1.0 / math.log(1 + m + 1)
        for j, cid in enumerate(l1_pubs):
            paper = all_papers_dict.get(cid)
            if paper is None:
                continue
            authors = paper.get("authors") or []
            team_factor = 1.0 / (max(len(authors), 1) ** 1.5)
            cite_count = sum(paper.get("citation_trajectory") or [])
            cite_factor = math.log(1 + cite_count)
            weight = EXPAND_DAMPING * l1_score * ((j + 1) / m) * team_factor * prolific_factor * (1.0 + 0.3 * cite_factor)
            for author in authors:
                aid = author.get("author_id") if isinstance(author, dict) else None
                if aid is None or aid == seed or aid == l1:
                    continue
                if aid in candidate_author_ids:
                    coauthor_scores[aid] += weight

    seed_neighbors = set(top_l1) | {seed}
    hop3_seeds = [aid for aid, _ in coauthor_scores.most_common(EXPAND_TOP_N + HOP3_TOP_N)
                  if aid not in seed_neighbors][:HOP3_TOP_N]
    for h2 in hop3_seeds:
        h2_score = coauthor_scores[h2]
        h2_pubs = get_preexisting_publications_for_author(h2, cutoff_date, sd2publications, all_papers_dict)[-HOP3_NUM_RECENT:]
        m = len(h2_pubs)
        if m == 0:
            continue
        prolific_factor = 1.0 / math.log(1 + m + 1)
        for j, cid in enumerate(h2_pubs):
            paper = all_papers_dict.get(cid)
            if paper is None:
                continue
            authors = paper.get("authors") or []
            team_factor = 1.0 / (max(len(authors), 1) ** 1.5)
            weight = HOP3_DAMPING * h2_score * ((j + 1) / m) * team_factor * prolific_factor
            for author in authors:
                aid = author.get("author_id") if isinstance(author, dict) else None
                if aid is None or aid == seed or aid == h2:
                    continue
                if aid in candidate_author_ids:
                    coauthor_scores[aid] += weight

    cited_corpus_ids = _get_cited_references_for_author(seed, cutoff_date, CITE_NUM_RECENT_PAPERS, sd2publications, all_papers_dict)
    for cid in cited_corpus_ids:
        paper = all_papers_dict.get(cid)
        if paper is None:
            continue
        authors = paper.get("authors") or []
        team_factor = 1.0 / (max(len(authors), 1) ** 1.5)
        weight = CITE_DAMPING * team_factor
        for author in authors:
            aid = author.get("author_id") if isinstance(author, dict) else None
            if aid is None or aid == seed:
                continue
            if aid in candidate_author_ids:
                coauthor_scores[aid] += weight

    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    half_life_days = ACTIVITY_HALF_LIFE_YEARS * 365.25
    sorted_authors = coauthor_scores.most_common(k)
    boosted = []
    for aid, score in sorted_authors:
        pubs = get_preexisting_publications_for_author(aid, cutoff_date, sd2publications, all_papers_dict)
        if pubs:
            last_paper = all_papers_dict.get(pubs[-1])
            if last_paper is not None:
                last_dt = datetime.strptime(last_paper["date"], "%Y-%m-%d")
                days_ago = max((cutoff_dt - last_dt).days, 0)
                activity = math.pow(0.5, days_ago / half_life_days)
                score = score * (1.0 + 2.5 * activity)
        boosted.append((aid, score))
    boosted.sort(key=lambda x: x[1], reverse=True)
    predicted_ids = [aid for aid, _ in boosted]
    predicted_scores = [float(s) for _, s in boosted]

    if len(predicted_ids) < k:
        excluded = set(predicted_ids) | {seed}
        available = list(candidate_author_ids - excluded)
        rng = random.Random(42 + hash(seed))
        rng.shuffle(available)
        n_pad = min(k - len(predicted_ids), len(available))
        predicted_ids.extend(available[:n_pad])
        predicted_scores.extend([0.0] * n_pad)
    return predicted_ids[:k], predicted_scores[:k]


def _build_author_first_pub_date(sd2publications, all_papers_dict):
    """For each author, the date of their earliest in-corpus publication."""
    out = {}
    for aid, pubs in sd2publications.items():
        if pubs is None:
            continue
        in_corpus = [p for p in pubs if p in all_papers_dict]
        if not in_corpus:
            continue
        out[aid] = min(all_papers_dict[p]["date"] for p in in_corpus)
    return out


def main():
    parser = argparse.ArgumentParser(description="Autoresearch agent best baseline for coauthor prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace dataset repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--k", type=int, default=1000, help="Number of coauthors to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=1000, help="Save predictions every N instances")
    parser.add_argument("--seed_author_type", type=str, default="first", choices=["first", "last", "random", "highest_h_index"], help="Seed author selection strategy")
    parser.add_argument("--output_dir", type=str, default="data/task_coauthor_prediction/test/predictions")
    args = parser.parse_args()

    all_papers, sd2publications, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.autoresearch.{args.seed_author_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers and {len(sd2publications)} author publication histories")

    utils.log(f"Creating evaluation instances with seed_author_type={args.seed_author_type}")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict, args.seed_author_type)
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    utils.log("Building author-first-pub date index")
    author_first_pub = _build_author_first_pub_date(sd2publications, all_papers_dict)
    sorted_authors = sorted(author_first_pub.items(), key=lambda kv: kv[1])
    candidate_author_ids = set()
    author_idx = 0

    utils.log("Running autoresearch coauthor baseline")
    predictions = []
    for idx, (date, instance) in enumerate(tqdm(evaluation_instances, desc="Running autoresearch baseline")):
        while author_idx < len(sorted_authors) and sorted_authors[author_idx][1] < date:
            candidate_author_ids.add(sorted_authors[author_idx][0])
            author_idx += 1

        if idx in selected_indices:
            corpus_id = instance["corpus_id"]
            first_author_id = instance["first_author_id"]
            gt_coauthor_ids = instance["gt_coauthor_ids"]
            pred_ids, pred_scores = predict_coauthors(first_author_id, date, args.k, candidate_author_ids, sd2publications, all_papers_dict)
            predictions.append({
                "corpus_id": corpus_id,
                "first_author_id": first_author_id,
                "gt_coauthor_ids": gt_coauthor_ids,
                "predicted_coauthor_ids": pred_ids,
                "predicted_coauthor_scores": pred_scores,
            })
            if len(predictions) % args.save_every == 0:
                utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
