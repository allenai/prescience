"""Autoresearch agent best baseline for prior work prediction.

Three-tier ranker: (1) cited-reference frequency across team's recent papers, (2) 2-hop co-cited references, (3) embedding similarity reranked with team-centroid + author-centroid + cited-ref-similarity boosts.
"""

import os
import random
import argparse
from collections import Counter
from tqdm import tqdm

import numpy as np

random.seed(42)

import utils
from task_priorwork_prediction.dataset import (
    create_evaluation_instances,
    get_preexisting_publications_for_author,
    get_cited_references_for_author,
)


NUM_RECENT_PAPERS = 10
COCITE_MIN_FREQ = 1
POPULARITY_ALPHA = 0.05
POPULARITY_WINDOW = 6
MAX_AUTHOR_SIM_GAMMA = 0.20
MAX_CITED_SIM_DELTA = 0.10
TOP_CITED_FOR_MAX_SIM = 30


def _get_embedding(all_embeddings, corpus_id):
    return all_embeddings[corpus_id]["key"].reshape(-1)


def _team_query_embedding(author_ids, cutoff_date, sd2publications, all_papers_dict, all_embeddings, distance_metric):
    author_embs = []
    for aid in author_ids:
        author_pubs = get_preexisting_publications_for_author(aid, cutoff_date, sd2publications, all_papers_dict)
        recent = author_pubs[-NUM_RECENT_PAPERS:]
        own_vecs = [_get_embedding(all_embeddings, cid) for cid in recent]
        ref_ids = get_cited_references_for_author(aid, cutoff_date, NUM_RECENT_PAPERS, sd2publications, all_papers_dict)
        ref_vecs = [_get_embedding(all_embeddings, rid) for rid in ref_ids if rid in all_papers_dict]
        per_author = []
        if own_vecs:
            per_author.append(utils.aggregate_embeddings(own_vecs, distance_metric))
        if ref_vecs:
            per_author.append(utils.aggregate_embeddings(ref_vecs, distance_metric))
        if per_author:
            author_embs.append(utils.aggregate_embeddings(per_author, distance_metric))
    if not author_embs:
        return None
    return utils.aggregate_embeddings(author_embs, distance_metric)


def _cited_reference_freq(author_ids, cutoff_date, candidate_set, sd2publications, all_papers_dict):
    freq = Counter()
    for aid in author_ids:
        for ref_id in get_cited_references_for_author(aid, cutoff_date, NUM_RECENT_PAPERS, sd2publications, all_papers_dict):
            if ref_id in candidate_set:
                freq[ref_id] += 1
    return freq


def _co_cited_freq(freq, candidate_set, all_papers_dict):
    co_freq = Counter()
    for ref_id, w in freq.items():
        p = all_papers_dict.get(ref_id)
        if p is None:
            continue
        for r in (p.get("key_references") or []):
            r_cid = r.get("corpus_id") if isinstance(r, dict) else None
            if r_cid is not None and r_cid in candidate_set:
                co_freq[r_cid] += float(w)
    return co_freq


def _popularity(cid, all_papers_dict):
    p = all_papers_dict.get(cid)
    if p is None:
        return 0
    traj = p.get("citation_trajectory") or []
    return sum(traj[-POPULARITY_WINDOW:])


def predict_references(author_ids, cutoff_date, k, candidate_set, index, sd2publications, all_papers_dict, all_embeddings, distance_metric):
    freq = _cited_reference_freq(author_ids, cutoff_date, candidate_set, sd2publications, all_papers_dict)
    co_freq = _co_cited_freq(freq, candidate_set, all_papers_dict)

    author_cents = []
    for aid in author_ids:
        pubs = get_preexisting_publications_for_author(aid, cutoff_date, sd2publications, all_papers_dict)[-NUM_RECENT_PAPERS:]
        vecs = [_get_embedding(all_embeddings, c) for c in pubs]
        vecs = [v / (np.linalg.norm(v) + 1e-12) for v in vecs]
        if vecs:
            c = np.mean(vecs, axis=0)
            c /= (np.linalg.norm(c) + 1e-12)
            author_cents.append(c)
    A = np.stack(author_cents, axis=0).astype(np.float32) if author_cents else None

    cited_vecs = []
    for cid, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:TOP_CITED_FOR_MAX_SIM]:
        if cid not in all_papers_dict:
            continue
        v = _get_embedding(all_embeddings, cid)
        cited_vecs.append(v / (np.linalg.norm(v) + 1e-12))
    R = np.stack(cited_vecs, axis=0).astype(np.float32) if cited_vecs else None

    query = _team_query_embedding(author_ids, cutoff_date, sd2publications, all_papers_dict, all_embeddings, distance_metric)
    emb_ranked = []
    emb_score = {}
    if query is not None:
        retrieved_lists, distances_arr = utils.query_index(index, [query], k)
        retrieved_ids = retrieved_lists[0]
        distances = distances_arr[0]
        rescored = []
        for cid, sim in zip(retrieved_ids, distances):
            if cid is None or cid not in candidate_set:
                continue
            sim = float(sim)
            max_auth = 0.0
            max_ref = 0.0
            if A is not None or R is not None:
                v = _get_embedding(all_embeddings, cid)
                v_norm = v / (np.linalg.norm(v) + 1e-12)
                if A is not None:
                    max_auth = float((A @ v_norm).max())
                if R is not None:
                    max_ref = float((R @ v_norm).max())
            score = (sim
                + POPULARITY_ALPHA * np.log1p(_popularity(cid, all_papers_dict))
                + MAX_AUTHOR_SIM_GAMMA * (max_auth ** 2)
                + MAX_CITED_SIM_DELTA * (max_ref ** 2))
            rescored.append((cid, score))
        rescored.sort(key=lambda x: x[1], reverse=True)
        emb_ranked = rescored
        emb_score = dict(emb_ranked)

        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        for cid in list(freq.keys()) + list(co_freq.keys()):
            if cid in emb_score or cid not in all_papers_dict:
                continue
            v = _get_embedding(all_embeddings, cid)
            v_norm = v / (np.linalg.norm(v) + 1e-12)
            sim = float(np.dot(q_norm, v_norm))
            max_auth = float((A @ v_norm).max()) if A is not None else 0.0
            max_ref = float((R @ v_norm).max()) if R is not None else 0.0
            emb_score[cid] = (sim
                + POPULARITY_ALPHA * np.log1p(_popularity(cid, all_papers_dict))
                + MAX_AUTHOR_SIM_GAMMA * (max_auth ** 2)
                + MAX_CITED_SIM_DELTA * (max_ref ** 2))

    predicted_ids, predicted_scores, seen = [], [], set()

    cited_sorted = sorted(freq.items(), key=lambda kv: (kv[1], emb_score.get(kv[0], -1e9)), reverse=True)
    for cid, f in cited_sorted:
        if cid in seen:
            continue
        seen.add(cid)
        predicted_ids.append(cid)
        predicted_scores.append(2000.0 + float(f))

    cocite_sorted = sorted(((cid, c) for cid, c in co_freq.items() if cid not in seen and c >= COCITE_MIN_FREQ),
                           key=lambda kv: (kv[1], emb_score.get(kv[0], -1e9)), reverse=True)
    for cid, c in cocite_sorted:
        seen.add(cid)
        predicted_ids.append(cid)
        predicted_scores.append(1000.0 + float(c))

    for cid, s in emb_ranked:
        if cid in seen:
            continue
        seen.add(cid)
        predicted_ids.append(cid)
        predicted_scores.append(s)
        if len(predicted_ids) >= k:
            break

    return predicted_ids[:k], predicted_scores[:k]


def main():
    parser = argparse.ArgumentParser(description="Autoresearch agent best baseline for prior work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--k", type=int, default=1000, help="Number of references to predict")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--output_dir", type=str, default="data/task_priorwork_prediction/test/predictions")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, sd2publications, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=True)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.autoresearch.{args.embedding_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors, {len(all_embeddings)} embeddings")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, sd2publications, all_papers_dict)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}
    utils.log(f"Created {len(evaluation_instances)} evaluation instances")

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    all_papers_sorted = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial paper index with cutoff date: {first_date}")
    paper_embeddings = {cid: _get_embedding(all_embeddings, cid) for cid, p in all_papers_dict.items() if p["date"] < first_date}
    utils.log(f"Created embeddings for {len(paper_embeddings)} papers")
    index = utils.create_index(paper_embeddings, distance_metric)
    candidate_set = set(paper_embeddings.keys())

    postdated_papers = [p for p in all_papers_sorted if p["date"] >= first_date]

    utils.log(f"Running autoresearch baseline with embedding: {args.embedding_type}")
    predictions = []
    for paper in tqdm(postdated_papers, desc="Running autoresearch baseline"):
        corpus_id = paper["corpus_id"]
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                predicted_ids, predicted_scores = predict_references(instance["author_ids"], paper["date"], args.k, candidate_set, index, sd2publications, all_papers_dict, all_embeddings, distance_metric)
                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_reference_ids": instance["gt_reference_ids"],
                    "predicted_reference_ids": predicted_ids,
                    "predicted_reference_scores": predicted_scores,
                })
                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        index = utils.add_vector_to_index(index, corpus_id, _get_embedding(all_embeddings, corpus_id))
        candidate_set.add(corpus_id)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
