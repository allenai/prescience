"""Print mean LACER scores for models in the month before and after their cutoff dates."""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.getcwd())
import utils


MODELS = [
    {
        "name": "claude-sonnet-4-5-20250929",
        "file": "data/task_followup_prediction/test/lacer_scored/generations.claude-sonnet-4-5-20250929.lacer_scored.json",
        "cutoff": "2025-07-31",
    },
    {
        "name": "claude-opus-4-5-20251101",
        "file": "data/task_followup_prediction/test/lacer_scored/generations.claude-opus-4-5-20251101.lacer_scored.json",
        "cutoff": "2025-08-31",
    },
    {
        "name": "gpt-5.2",
        "file": "data/task_followup_prediction/test/lacer_scored/generations.gpt-5.2-2025-12-11.lacer_scored.json",
        "cutoff": "2025-08-31",
    },
]


def get_month_range(cutoff_date, before=True):
    """Get the start and end dates for the month before or after the cutoff."""
    if before:
        end = cutoff_date
        start = cutoff_date.replace(day=1)
    else:
        start = cutoff_date + timedelta(days=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    return start, end


def compute_mean_lacer(records, start_date, end_date):
    """Compute mean LACER score for records within the date range."""
    scores = []
    for record in records:
        if "lacer_score" not in record or "date" not in record:
            continue
        record_date = datetime.strptime(record["date"], "%Y-%m-%d")
        if start_date <= record_date <= end_date:
            scores.append(record["lacer_score"])
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def main():
    print("=" * 80)
    print("Mean LACER Scores Before/After Model Cutoff Dates")
    print("=" * 80)
    print()

    for model in MODELS:
        if not os.path.exists(model["file"]):
            print(f"[SKIP] {model['name']}: File not found ({model['file']})")
            continue

        data, _ = utils.load_json(model["file"])
        cutoff = datetime.strptime(model["cutoff"], "%Y-%m-%d")

        pre_start, pre_end = get_month_range(cutoff, before=True)
        post_start, post_end = get_month_range(cutoff, before=False)

        pre_mean, pre_count = compute_mean_lacer(data, pre_start, pre_end)
        post_mean, post_count = compute_mean_lacer(data, post_start, post_end)

        print(f"Model: {model['name']}")
        print(f"  Cutoff date: {model['cutoff']}")
        print(f"  Pre-cutoff  ({pre_start.strftime('%Y-%m-%d')} to {pre_end.strftime('%Y-%m-%d')}): "
              f"mean={pre_mean:.3f} (n={pre_count})" if pre_mean else f"  Pre-cutoff: No data")
        print(f"  Post-cutoff ({post_start.strftime('%Y-%m-%d')} to {post_end.strftime('%Y-%m-%d')}): "
              f"mean={post_mean:.3f} (n={post_count})" if post_mean else f"  Post-cutoff: No data")
        if pre_mean and post_mean:
            diff = post_mean - pre_mean
            print(f"  Difference (post - pre): {diff:+.3f}")
        print()


if __name__ == "__main__":
    main()
