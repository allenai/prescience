"""Embedding cluster alignment validation for the topic-labeling pipeline.

For each topic with ≥min_topic_size labeled papers, sample paper pairs from inside that
topic and compute mean cosine similarity in the chosen embedding space (GRIT by default).
Compare against a global null built by sampling random paper pairs across the entire labeled
corpus. Report per-topic intra-similarity, the null, the per-topic gap and z-score, plus
aggregate quantiles.

Free, fast (~5 min compute on 97k papers + GRIT embeddings).
"""

import argparse
import json
import os
import random
import numpy as np

import utils
from dataset.corpus.assign_topics import load_topics


def load_topic_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["corpus_id"]] = r["topics"]
    return labels


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def sample_pair_similarities(unit_matrix, indices, n_pairs, rng):
    """Sample n_pairs random pairs (with replacement) from `indices`, return cosine-similarity array."""
    if len(indices) < 2:
        return np.array([], dtype=np.float32)
    a = rng.choices(indices, k=n_pairs)
    b = rng.choices(indices, k=n_pairs)
    keep = np.array([(ai != bi) for ai, bi in zip(a, b)], dtype=bool)
    a = np.array(a)[keep]
    b = np.array(b)[keep]
    if len(a) == 0:
        return np.array([], dtype=np.float32)
    return np.einsum("ij,ij->i", unit_matrix[a], unit_matrix[b]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Embedding cluster alignment for topic labels (GRIT default)")
    parser.add_argument("--labels_path", type=str, default="data/task_topic_growth_prediction/topic_labels.v5.gpt-5.4.full.jsonl")
    parser.add_argument("--topics_path", type=str, default="dataset/corpus/topics_list_v5.txt")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience")
    parser.add_argument("--embeddings_dir_train", type=str, default="data/corpus/train")
    parser.add_argument("--embeddings_dir_test", type=str, default="data/corpus/test")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["grit", "gtr", "specter2"])
    parser.add_argument("--min_topic_size", type=int, default=20, help="Skip topics with fewer than this many labeled papers")
    parser.add_argument("--pairs_per_topic", type=int, default=2000, help="How many intra-topic pairs to sample per topic")
    parser.add_argument("--null_pairs", type=int, default=50000, help="How many random global pairs for the null")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="data/task_topic_growth_prediction/scored/label_coherence.v5.grit.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    topics = load_topics(args.topics_path)
    topic_set = set(topics)
    utils.log(f"Loaded {len(topics)} topics")

    utils.log(f"Loading labels from {args.labels_path}")
    labels = load_topic_labels(args.labels_path)
    utils.log(f"Loaded labels for {len(labels)} papers")

    utils.log(f"Loading corpus + {args.embedding_type} embeddings (train, test)")
    train_papers, _, train_emb = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="train", embeddings_dir=args.embeddings_dir_train, embedding_type=args.embedding_type, load_sd2publications=False)
    test_papers, _, test_emb = utils.load_corpus(hf_repo_id=args.hf_repo_id, split="test", embeddings_dir=args.embeddings_dir_test, embedding_type=args.embedding_type, load_sd2publications=False)
    embeddings = {}
    embeddings.update(train_emb)
    for cid, e in test_emb.items():
        if cid not in embeddings:
            embeddings[cid] = e
    utils.log(f"Embeddings loaded for {len(embeddings)} papers")

    eligible_ids = []
    for cid, topic_list in labels.items():
        if cid not in embeddings:
            continue
        if not any(t in topic_set for t in topic_list):
            continue
        eligible_ids.append(cid)
    utils.log(f"Eligible labeled papers with {args.embedding_type} embedding: {len(eligible_ids)}")

    def to_vec(emb):
        # GRIT stores {'key': (1,d), 'query': (1,d)}; GTR/SPECTER2 store (d,) or (1,d)
        if isinstance(emb, dict):
            arr = emb.get("key", next(iter(emb.values())))
        else:
            arr = emb
        arr = np.asarray(arr, dtype=np.float32)
        return arr.reshape(-1)
    id_to_idx = {cid: i for i, cid in enumerate(eligible_ids)}
    matrix = np.stack([to_vec(embeddings[cid]) for cid in eligible_ids], axis=0)
    utils.log(f"Embedding matrix: {matrix.shape}, dtype={matrix.dtype}")
    unit = normalize_rows(matrix)

    indices_by_topic = {t: [] for t in topics}
    for cid in eligible_ids:
        for t in labels[cid]:
            if t in topic_set:
                indices_by_topic[t].append(id_to_idx[cid])

    utils.log(f"Sampling {args.null_pairs} global random pairs for null")
    all_idx = list(range(len(eligible_ids)))
    null_sims = sample_pair_similarities(unit, all_idx, args.null_pairs, rng)
    null_mean = float(null_sims.mean())
    null_std = float(null_sims.std())
    utils.log(f"null cosine: mean={null_mean:.4f}  std={null_std:.4f}  n_pairs={len(null_sims)}")

    per_topic = []
    for t in topics:
        idx_list = indices_by_topic[t]
        n_papers = len(idx_list)
        if n_papers < args.min_topic_size:
            per_topic.append({"topic": t, "n_papers": n_papers, "intra_mean": None, "intra_std": None, "gap_vs_null": None, "z_vs_null": None, "skipped": True})
            continue
        sims = sample_pair_similarities(unit, idx_list, args.pairs_per_topic, rng)
        intra_mean = float(sims.mean())
        intra_std = float(sims.std())
        gap = intra_mean - null_mean
        z = gap / null_std if null_std > 0 else None
        per_topic.append({"topic": t, "n_papers": n_papers, "intra_mean": intra_mean, "intra_std": intra_std, "gap_vs_null": gap, "z_vs_null": z, "skipped": False})

    kept = [r for r in per_topic if not r["skipped"]]
    skipped = [r for r in per_topic if r["skipped"]]
    gaps = np.array([r["gap_vs_null"] for r in kept], dtype=np.float64)
    zs = np.array([r["z_vs_null"] for r in kept], dtype=np.float64)
    intra = np.array([r["intra_mean"] for r in kept], dtype=np.float64)

    summary = {"embedding_type": args.embedding_type, "n_topics_evaluated": len(kept), "n_topics_skipped_too_small": len(skipped), "min_topic_size": args.min_topic_size, "null_cosine_mean": null_mean, "null_cosine_std": null_std, "intra_mean_median": float(np.median(intra)), "intra_mean_q25": float(np.quantile(intra, 0.25)), "intra_mean_q75": float(np.quantile(intra, 0.75)), "gap_vs_null_median": float(np.median(gaps)), "gap_vs_null_min": float(gaps.min()), "gap_vs_null_max": float(gaps.max()), "z_vs_null_median": float(np.median(zs)), "z_vs_null_min": float(zs.min()), "z_vs_null_max": float(zs.max()), "frac_topics_with_positive_gap": float((gaps > 0).mean()), "frac_topics_with_z_above_2": float((zs > 2.0).mean())}

    utils.log("Per-topic embedding-coherence summary:")
    for k, v in summary.items():
        utils.log(f"  {k}: {v}")

    per_topic_sorted = sorted(kept, key=lambda r: r["gap_vs_null"], reverse=True)
    utils.log("Top 10 most-coherent topics (by intra−null gap):")
    for r in per_topic_sorted[:10]:
        utils.log(f"  {r['topic']:60s}  n={r['n_papers']:5d}  intra={r['intra_mean']:.4f}  gap={r['gap_vs_null']:+.4f}  z={r['z_vs_null']:.2f}")
    utils.log("Bottom 10 least-coherent topics:")
    for r in per_topic_sorted[-10:]:
        utils.log(f"  {r['topic']:60s}  n={r['n_papers']:5d}  intra={r['intra_mean']:.4f}  gap={r['gap_vs_null']:+.4f}  z={r['z_vs_null']:.2f}")

    output = {"config": {"labels_path": args.labels_path, "topics_path": args.topics_path, "embedding_type": args.embedding_type, "min_topic_size": args.min_topic_size, "pairs_per_topic": args.pairs_per_topic, "null_pairs": args.null_pairs, "seed": args.seed}, "summary": summary, "per_topic": per_topic}
    utils.log(f"Saving to {args.output_path}")
    utils.save_json(output, args.output_path, metadata=utils.update_metadata([], args), overwrite=True)


if __name__ == "__main__":
    main()
