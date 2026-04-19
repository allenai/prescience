"""Dataset creation utilities for co-citation prediction task."""

import os
from bisect import bisect_right
from datetime import datetime, timedelta
from tqdm import tqdm

import utils


def add_months(date_str, months):
    """Approximate month-offset on an ISO date string (days = months * 30, matches citation_trajectory convention)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=months * 30)
    return dt.strftime("%Y-%m-%d")


def create_evaluation_instances(all_papers, all_papers_dict, eval_months=4, lookahead_months=8, cache_path=None):
    """Create chronologically-sorted evaluation instances for co-citation prediction.

    For each target paper P in the first `eval_months` of the test period, compute a ground-truth ranking
    of past papers q (with q.date < P.date) by the number of future test-period target papers T with
    P.date < T.date <= P.date + lookahead_months that co-key-reference q with P.
    """
    if cache_path is not None and os.path.exists(cache_path):
        utils.log(f"Loading cached evaluation instances from {cache_path}")
        data, _ = utils.load_json(cache_path)
        return [(inst["date"], inst) for inst in data]

    target_dates = [p["date"] for p in all_papers if "target" in p["roles"]]
    test_start = min(target_dates)
    test_end = max(target_dates)
    utils.log(f"Test period: {test_start} to {test_end}")

    eval_cutoff = add_months(test_start, eval_months)
    required_end = add_months(test_start, eval_months + lookahead_months)
    if test_end < required_end:
        utils.log(f"Warning: latest target date {test_end} is earlier than required {required_end} "
                  f"(test_start + eval_months + lookahead_months); P's near the eval cutoff will have truncated lookahead windows")

    targets_sorted = sorted([p for p in all_papers if "target" in p["roles"]], key=lambda p: p["date"])
    target_dates_sorted = [p["date"] for p in targets_sorted]
    eval_targets = [p for p in targets_sorted if p["date"] < eval_cutoff]
    utils.log(f"Eval cohort: {len(eval_targets)} targets with date < {eval_cutoff}")

    instances = []
    num_dropped_empty = 0
    for paper in tqdm(eval_targets, desc="Creating evaluation instances"):
        corpus_id = paper["corpus_id"]
        paper_date = paper["date"]
        window_end = add_months(paper_date, lookahead_months)
        window_start_idx = bisect_right(target_dates_sorted, paper_date)
        window_end_idx = bisect_right(target_dates_sorted, window_end)

        cocitation_counts = {}
        for future in targets_sorted[window_start_idx:window_end_idx]:
            future_ref_ids = [r["corpus_id"] for r in future["key_references"]]
            if corpus_id in future_ref_ids:
                for ref_id in future_ref_ids:
                    if ref_id != corpus_id and ref_id in all_papers_dict and all_papers_dict[ref_id]["date"] < paper_date:
                        if ref_id not in cocitation_counts:
                            cocitation_counts[ref_id] = 0
                        cocitation_counts[ref_id] += 1

        if len(cocitation_counts) == 0:
            num_dropped_empty += 1
        else:
            sorted_cocited = sorted(cocitation_counts.items(), key=lambda x: (-x[1], -int(all_papers_dict[x[0]]["date"].replace("-", ""))))
            instances.append((paper_date, {
                "corpus_id": corpus_id,
                "date": paper_date,
                "key_reference_ids": [r["corpus_id"] for r in paper["key_references"]],
                "gt_cocited_ids": [cid for cid, _ in sorted_cocited],
                "gt_cocited_counts": [cnt for _, cnt in sorted_cocited],
            }))

    utils.log(f"Created {len(instances)} evaluation instances; dropped {num_dropped_empty} with empty ground truth")

    if cache_path is not None:
        utils.log(f"Saving evaluation instances cache to {cache_path}")
        utils.save_json([inst for _, inst in instances], cache_path, metadata=[{"eval_months": eval_months, "lookahead_months": lookahead_months}], overwrite=True)

    return instances
