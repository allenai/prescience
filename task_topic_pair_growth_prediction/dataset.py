"""Dataset utilities for the topic-pair-growth-prediction task.

Each instance is an unordered pair of topics from `dataset/corpus/topics_list_v5.txt`
(canonicalized as f"{a}||{b}" with a < b lexicographically). A paper contributes to a pair
if both topics appear in its label list. Pairs with fewer than the configured minimum count
in either history or forecast period are filtered out.
"""

import utils


def load_topics_benchmark(clusters_path):
    """Load a topic_pairs.*.json produced by compute_topic_pairs.py.

    Function name and return signature mirror task_topic_growth_prediction.dataset so
    evaluate.py and the copied baselines work unchanged at the pair level.
    """
    payload, metadata = utils.load_json(clusters_path)
    instances = [(cid, inst) for cid, inst in payload["instances"]]
    return payload["config"], instances, payload["membership"], payload["clusters"], metadata
