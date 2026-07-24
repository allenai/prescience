"""Dataset utilities for the topic-growth-prediction task.

Topics here are LLM-applied labels from `dataset/corpus/topics_list_v5.txt` (202 topics).
Each paper can carry multiple topic labels; papers with empty label lists ("Other") are
excluded from the benchmark.

History period = 2023-10-01 → 2024-10-01 (train split, 12 months).
Forecast period = 2024-10-01 → 2025-10-01 (test split, 12 months).
"""

import utils
from utils import enumerate_months, filter_by_date_range, month_key


def load_topics_benchmark(clusters_path):
    """Load a topics.*.json produced by compute_topics.py.

    Returns (config, instances, membership, clusters, metadata).
    """
    payload, metadata = utils.load_json(clusters_path)
    instances = [(cid, inst) for cid, inst in payload["instances"]]
    return payload["config"], instances, payload["membership"], payload["clusters"], metadata
