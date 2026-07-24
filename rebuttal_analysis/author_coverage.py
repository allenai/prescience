"""Author-profile coverage and missingness analysis for the rebuttal (Reviewer 1 W3/Q1, Reviewer 4 Q3).

Reports, on the test split loaded via the canonical utils.load_corpus path: how many roster authors
have profiles in sd2publications, the distribution of prior-publication depth at each paper's date, and
the effect of each author-dependent task's eligibility filter (how many targets are excluded and how the
excluded papers differ from the included ones)."""

import argparse
import json
import os
from collections import Counter

import numpy as np

import utils
from task_coauthor_prediction.dataset import get_preexisting_publications_for_author


def depth_bucket(n):
    """Bucket a prior-publication count for the distribution summary."""
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def describe_papers(papers):
    """Summary stats for a set of target papers (for included-vs-excluded comparison)."""
    if not papers:
        return {"n": 0}
    num_authors = [len(p["authors"]) for p in papers]
    max_hindex = [max((a["h_index"] for a in p["authors"]), default=0) for p in papers]
    return {
        "n": len(papers),
        "mean_num_authors": round(float(np.mean(num_authors)), 2),
        "mean_max_h_index": round(float(np.mean(max_hindex)), 2),
        "median_max_h_index": float(np.median(max_hindex)),
    }


def main():
    parser = argparse.ArgumentParser(description="Author-profile coverage and missingness analysis")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output_dir", default="rebuttal_outputs/C_author_robustness", help="Directory for outputs")
    args = parser.parse_args()

    all_papers, sd2publications, _ = utils.load_corpus(split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    targets = [p for p in all_papers if "target" in p["roles"]]
    utils.log(f"Loaded {len(all_papers)} papers, {len(targets)} targets, {len(sd2publications)} sd2 authors")

    # Coverage: roster authors present in sd2publications and prior-pub depth at each paper's date.
    roster_entries = [(p, a) for p in targets for a in p["authors"]]
    unique_authors = set(a["author_id"] for _, a in roster_entries)
    absent = [aid for aid in unique_authors if aid not in sd2publications]

    depth_counter = Counter()
    zero_prior = 0
    for paper, author in roster_entries:
        n_prior = len(get_preexisting_publications_for_author(author["author_id"], paper["date"], sd2publications, all_papers_dict))
        depth_counter[depth_bucket(n_prior)] += 1
        if n_prior == 0:
            zero_prior += 1

    coverage = {
        "unique_roster_authors": len(unique_authors),
        "roster_author_entries": len(roster_entries),
        "authors_absent_from_sd2publications": len(absent),
        "author_entries_with_zero_prior_pubs_at_paper_date": zero_prior,
        "author_entries_with_zero_prior_pubs_pct": round(100 * zero_prior / len(roster_entries), 2),
        "prior_pub_depth_distribution": {k: depth_counter[k] for k in ["0", "1-2", "3-5", "6-10", "11+"]},
    }

    # Eligibility-filter effect per task.
    def author_prior_count(author_id, date):
        return len(get_preexisting_publications_for_author(author_id, date, sd2publications, all_papers_dict))

    def has_refbearing_prior(paper):
        for author in paper["authors"]:
            for cid in get_preexisting_publications_for_author(author["author_id"], paper["date"], sd2publications, all_papers_dict):
                if len(all_papers_dict[cid].get("key_references") or []) > 0:
                    return True
        return False

    coauthor_incl, coauthor_excl, priorwork_incl, priorwork_excl = [], [], [], []
    for paper in targets:
        n_prior_authors = sum(1 for a in paper["authors"] if author_prior_count(a["author_id"], paper["date"]) > 0)
        (coauthor_incl if len(paper["authors"]) >= 2 and n_prior_authors >= 2 else coauthor_excl).append(paper)
        (priorwork_incl if has_refbearing_prior(paper) else priorwork_excl).append(paper)
    impact_incl = [p for p in targets if len(p.get("citation_trajectory") or []) >= 12]

    eligibility = {
        "total_targets": len(targets),
        "coauthor": {"eligible": len(coauthor_incl), "excluded": len(coauthor_excl), "excluded_pct": round(100 * len(coauthor_excl) / len(targets), 2), "included_profile": describe_papers(coauthor_incl), "excluded_profile": describe_papers(coauthor_excl)},
        "priorwork": {"eligible": len(priorwork_incl), "excluded": len(priorwork_excl), "excluded_pct": round(100 * len(priorwork_excl) / len(targets), 2), "included_profile": describe_papers(priorwork_incl), "excluded_profile": describe_papers(priorwork_excl)},
        "impact": {"eligible": len(impact_incl), "excluded": len(targets) - len(impact_incl), "excluded_pct": round(100 * (len(targets) - len(impact_incl)) / len(targets), 2)},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "coverage_missingness.json"), "w") as f:
        json.dump({"split": args.split, "coverage": coverage, "eligibility": eligibility}, f, indent=2)
    with open(os.path.join(args.output_dir, "coverage_missingness.md"), "w") as f:
        f.write(render_markdown(coverage, eligibility, args.split))
    utils.log(f"Saved coverage/missingness outputs to {args.output_dir}")


def render_markdown(coverage, eligibility, split):
    """Render a drop-in coverage/missingness summary."""
    lines = [f"# Author-profile coverage & missingness ({split} split)", ""]
    lines.append(f"- Unique roster authors: **{coverage['unique_roster_authors']:,}** across {coverage['roster_author_entries']:,} author-entries on target papers.")
    lines.append(f"- Authors absent from `sd2publications`: **{coverage['authors_absent_from_sd2publications']:,}** ({100*coverage['authors_absent_from_sd2publications']/coverage['unique_roster_authors']:.2f}%).")
    lines.append(f"- Author-entries with **zero** prior publications at their paper's date: **{coverage['author_entries_with_zero_prior_pubs_at_paper_date']:,}** ({coverage['author_entries_with_zero_prior_pubs_pct']}%).")
    lines.append("")
    lines.append("## Prior-publication depth (per author-entry, at paper date)")
    lines.append("")
    lines.append("| Depth | Author-entries |")
    lines.append("|---|---|")
    for k, v in coverage["prior_pub_depth_distribution"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append("## Eligibility-filter effect (fraction of target papers usable per task)")
    lines.append("")
    lines.append("| Task | Eligible | Excluded | Excluded % |")
    lines.append("|---|---|---|---|")
    for task in ["coauthor", "priorwork", "impact"]:
        e = eligibility[task]
        lines.append(f"| {task} | {e['eligible']:,} | {e['excluded']:,} | {e['excluded_pct']}% |")
    lines.append("")
    lines.append("## Included vs. excluded papers (coauthor & priorwork)")
    lines.append("")
    lines.append("| Task | Set | n | mean #authors | mean max h-index |")
    lines.append("|---|---|---|---|---|")
    for task in ["coauthor", "priorwork"]:
        for setname in ["included_profile", "excluded_profile"]:
            pr = eligibility[task][setname]
            lines.append(f"| {task} | {setname.split('_')[0]} | {pr['n']:,} | {pr.get('mean_num_authors','-')} | {pr.get('mean_max_h_index','-')} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
