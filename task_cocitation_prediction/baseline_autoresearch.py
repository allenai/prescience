"""Autoresearch agent best baseline for future-influential co-citation prediction.

RRF (k=6) over three rankings: (1) IDF-weighted reference centroid query, (2) target title-abstract embedding query, (3) cocitation graph score with citer-similarity weighting, recency decay, and topic/category overlap boost.

NOTE: lives on the `cocitation` branch (where `task_cocitation_prediction/dataset.py` and the other cocitation baselines exist). On other branches the import below will fail.
"""

import os
import math
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np

import utils
from task_cocitation_prediction.dataset import create_evaluation_instances

random.seed(42)


K_DEFAULT = 1000
RRF_K = 6
EMB_RETRIEVE = 1500
COCITE_TOP = 1500
RECENCY_TAU_DAYS = 365.0
_EPOCH = datetime(2000, 1, 1)


def _get_embedding(all_embeddings, corpus_id):
    return all_embeddings[corpus_id]["key"].reshape(-1)


def _date_to_days(ds):
    if not ds:
        return None
    return (datetime.strptime(ds, "%Y-%m-%d") - _EPOCH).days


def _add_rrf(rrf_score, ranked_ids, candidate_set):
    for rank, cid in enumerate(ranked_ids):
        if cid is None or cid not in candidate_set:
            continue
        rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (RRF_K + rank)


def _advance_cocite_state(state, new_ids, all_papers_dict, all_embeddings):
    """Add new pre-cutoff candidate papers to the running cocitation index state."""
    indexed = state["cocite_indexed"]
    refsets = state["cocite_refsets"]
    inverted = state["cocite_inverted"]
    normed = state["normed_embs"]
    citer_days = state["citer_days"]
    cand_topics = state["cand_topics"]
    cand_cats = state["cand_cats"]
    for cid in new_ids:
        if cid in indexed:
            continue
        indexed.add(cid)
        v = _get_embedding(all_embeddings, cid)
        n = float(np.linalg.norm(v))
        if n > 0:
            normed[cid] = (v / n).astype(np.float32)
        paper = all_papers_dict.get(cid)
        if paper is None:
            continue
        d = _date_to_days(paper.get("date"))
        if d is not None:
            citer_days[cid] = d
        topics = paper.get("topic_labels") or []
        if topics:
            cand_topics[cid] = frozenset(topics)
        cats = paper.get("categories") or []
        if cats:
            cand_cats[cid] = frozenset(cats)
        refs = paper.get("key_references") or []
        ref_ids = [r["corpus_id"] for r in refs if isinstance(r, dict) and "corpus_id" in r]
        if not ref_ids:
            continue
        K_X = frozenset(ref_ids)
        refsets[cid] = K_X
        for r in ref_ids:
            inverted.setdefault(r, []).append(cid)


def predict_cocited(instance, target_paper, target_embedding, k, candidate_set, index, state, all_papers_dict, all_embeddings, distance_metric, cutoff_date):
    refsets = state["cocite_refsets"]
    inverted = state["cocite_inverted"]
    normed = state["normed_embs"]
    citer_days = state["citer_days"]
    cand_topics_dict = state["cand_topics"]
    cand_cats_dict = state["cand_cats"]

    ref_ids = instance["key_reference_ids"]
    R_P = frozenset(ref_ids)
    N = max(1, len(refsets))
    idf = {r: math.log((N + 1) / (len(inverted.get(r, [])) + 1)) + 1.0 for r in R_P}

    ref_pairs = []
    for r in ref_ids:
        if r not in all_papers_dict:
            continue
        v = _get_embedding(all_embeddings, r)
        ref_pairs.append((idf[r], np.asarray(v).reshape(-1).astype(np.float32)))

    rrf_score = {}

    if ref_pairs:
        if distance_metric == "cosine":
            normed_pairs = [(w, v / (np.linalg.norm(v) + 1e-12)) for w, v in ref_pairs]
            ws = sum(w * v for w, v in normed_pairs)
        else:
            total_w = sum(w for w, _ in ref_pairs)
            ws = sum(w * v for w, v in ref_pairs) / total_w if total_w > 0 else None
        if ws is not None:
            wsn = float(np.linalg.norm(ws))
            if wsn > 0:
                q = ws / wsn if distance_metric == "cosine" else ws
                retrieved_lists, _ = utils.query_index(index, [q], EMB_RETRIEVE)
                _add_rrf(rrf_score, retrieved_lists[0], candidate_set)

    if target_embedding is not None:
        q = np.asarray(target_embedding).reshape(-1)
        retrieved_lists, _ = utils.query_index(index, [q], EMB_RETRIEVE)
        _add_rrf(rrf_score, retrieved_lists[0], candidate_set)

    target_norm = None
    if target_embedding is not None:
        t = np.asarray(target_embedding).reshape(-1).astype(np.float32)
        tn = float(np.linalg.norm(t))
        if tn > 0:
            target_norm = t / tn

    candidate_citers = set()
    for r in R_P:
        lst = inverted.get(r)
        if lst:
            candidate_citers.update(lst)
    cutoff_days = _date_to_days(cutoff_date) or 0

    cand_sim = {}
    if target_norm is not None:
        unique_cands = set()
        for X in candidate_citers:
            K_X = refsets[X]
            if R_P & K_X:
                unique_cands.update(K_X)
        valid = [c for c in unique_cands if c in candidate_set and c in normed]
        if valid:
            mat = np.stack([normed[c] for c in valid], axis=0)
            sims = mat @ target_norm
            for c, s in zip(valid, sims):
                v = float(s)
                cand_sim[c] = v if v > 0 else 0.0

    target_topics = frozenset(target_paper.get("topic_labels") or [])
    target_cats = frozenset(target_paper.get("categories") or [])

    cocite_scores = {}
    for X in candidate_citers:
        K_X = refsets[X]
        overlap_refs = R_P & K_X
        if not overlap_refs:
            continue
        weighted = sum(idf[r] for r in overlap_refs) / math.sqrt(len(K_X))
        if target_norm is not None:
            xn = normed.get(X)
            if xn is not None:
                sim = float(np.dot(xn, target_norm))
                if sim < 0:
                    sim = 0.0
                weighted *= sim
        xd = citer_days.get(X)
        if xd is not None:
            delta = max(0, cutoff_days - xd)
            weighted *= math.exp(-delta / RECENCY_TAU_DAYS)
        for cand in K_X:
            if cand in candidate_set:
                cs = cand_sim.get(cand, 1.0)
                boost = 1.0
                if target_topics:
                    ct = cand_topics_dict.get(cand)
                    if ct and (target_topics & ct):
                        boost += 1.0
                if target_cats:
                    cc = cand_cats_dict.get(cand)
                    if cc and (target_cats & cc):
                        boost += 1.0
                cocite_scores[cand] = cocite_scores.get(cand, 0.0) + weighted * (cs ** 0.7) * boost
    if cocite_scores:
        cocite_ranked = sorted(cocite_scores.items(), key=lambda x: x[1], reverse=True)
        cocite_ids = [cid for cid, _ in cocite_ranked[:COCITE_TOP]]
        _add_rrf(rrf_score, cocite_ids, candidate_set)

    if not rrf_score:
        return [], []

    ranked = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
    predicted_ids = [cid for cid, _ in ranked[:k]]
    predicted_scores = [s for _, s in ranked[:k]]
    return predicted_ids, predicted_scores


def main():
    parser = argparse.ArgumentParser(description="Autoresearch agent best baseline for co-citation prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repository ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing embeddings pkl files")
    parser.add_argument("--embedding_type", type=str, required=True, choices=["gtr", "grit", "specter2"])
    parser.add_argument("--eval_months", type=int, default=4, help="Size of evaluation cohort window in months")
    parser.add_argument("--lookahead_months", type=int, default=8, help="Size of future-citation observation window in months")
    parser.add_argument("--k", type=int, default=K_DEFAULT, help="Number of papers to predict per instance")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum number of random instances to evaluate")
    parser.add_argument("--save_every", type=int, default=5000, help="Save predictions every N instances")
    parser.add_argument("--eval_instances_cache", type=str, default=None, help="Path to cached evaluation instances JSON")
    parser.add_argument("--output_dir", type=str, default="data/task_cocitation_prediction/test/predictions")
    args = parser.parse_args()

    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"

    utils.log(f"Loading corpus from {args.hf_repo_id} (split={args.split})")
    all_papers, _, all_embeddings = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.embeddings_dir, embedding_type=args.embedding_type, load_sd2publications=False)
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}
    output_path = os.path.join(args.output_dir, f"predictions.autoresearch.{args.embedding_type}.json")
    utils.log(f"Loaded {len(all_papers)} papers and {len(all_embeddings)} embeddings")

    utils.log("Creating evaluation instances")
    evaluation_instances = create_evaluation_instances(all_papers, all_papers_dict, eval_months=args.eval_months, lookahead_months=args.lookahead_months, cache_path=args.eval_instances_cache)
    eval_instance_dict = {instance["corpus_id"]: (idx, instance) for idx, (date, instance) in enumerate(evaluation_instances)}

    if args.max_instances is not None and args.max_instances < len(evaluation_instances):
        selected_indices = set(random.sample(range(len(evaluation_instances)), args.max_instances))
        utils.log(f"Randomly selected {len(selected_indices)} instances to evaluate")
    else:
        selected_indices = set(range(len(evaluation_instances)))

    all_papers_sorted = sorted(all_papers, key=lambda p: p["date"])
    first_date = evaluation_instances[0][0]
    utils.log(f"Creating initial paper index with cutoff date: {first_date}")
    initial_embs = {cid: _get_embedding(all_embeddings, cid) for cid, p in all_papers_dict.items() if p["date"] < first_date}
    utils.log(f"Created embeddings for {len(initial_embs)} papers")
    index = utils.create_index(initial_embs, distance_metric)
    candidate_set = set(initial_embs.keys())

    state = {
        "cocite_indexed": set(),
        "cocite_refsets": {},
        "cocite_inverted": {},
        "normed_embs": {},
        "citer_days": {},
        "cand_topics": {},
        "cand_cats": {},
    }
    _advance_cocite_state(state, candidate_set, all_papers_dict, all_embeddings)

    postdated_papers = [p for p in all_papers_sorted if p["date"] >= first_date]

    utils.log(f"Running autoresearch cocitation baseline with embedding: {args.embedding_type}")
    predictions = []
    pending_adds = []
    for paper in tqdm(postdated_papers, desc="Running autoresearch baseline"):
        if pending_adds and pending_adds[0]["date"] < paper["date"]:
            new_ids = []
            for p in pending_adds:
                cid = p["corpus_id"]
                index = utils.add_vector_to_index(index, cid, _get_embedding(all_embeddings, cid))
                candidate_set.add(cid)
                new_ids.append(cid)
            _advance_cocite_state(state, new_ids, all_papers_dict, all_embeddings)
            pending_adds = []

        corpus_id = paper["corpus_id"]
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected_indices:
                target_paper = all_papers_dict[corpus_id]
                target_embedding = _get_embedding(all_embeddings, corpus_id)
                predicted_ids, predicted_scores = predict_cocited(instance, target_paper, target_embedding, args.k, candidate_set, index, state, all_papers_dict, all_embeddings, distance_metric, paper["date"])
                predictions.append({
                    "corpus_id": corpus_id,
                    "gt_cocited_ids": instance["gt_cocited_ids"],
                    "gt_cocited_counts": instance["gt_cocited_counts"],
                    "predicted_cocited_ids": predicted_ids,
                    "predicted_cocited_scores": predicted_scores,
                })
                if len(predictions) % args.save_every == 0:
                    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)

        pending_adds.append(paper)

    if pending_adds:
        new_ids = []
        for p in pending_adds:
            cid = p["corpus_id"]
            index = utils.add_vector_to_index(index, cid, _get_embedding(all_embeddings, cid))
            candidate_set.add(cid)
            new_ids.append(cid)
        _advance_cocite_state(state, new_ids, all_papers_dict, all_embeddings)

    utils.save_json(predictions, output_path, metadata=utils.update_metadata([], args), overwrite=True)
    utils.log(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
