"""Author-history degradation sensitivity for the rebuttal (Reviewer 1 W3, Reviewer 4 Q3).

Randomly drops a fraction of each author's prior publications and re-runs the author-dependent
retrieval baselines on a fixed instance subsample, to show task performance degrades gracefully (or
holds) under incomplete author histories. Covers coauthor (frequency) and prior-work (embedding
fusion). Instances are built once on the clean history so the evaluated set is identical across
degradation levels; only the histories used at prediction time are degraded.

Impact-task author sensitivity is a feature knock-out, run separately via baseline_xgboost_regressor.py."""

import argparse
import json
import os
import random

import numpy as np

import utils
from task_coauthor_prediction.dataset import create_evaluation_instances as coauthor_instances
from task_coauthor_prediction.baseline_frequency import predict_coauthors_one_shot
from task_priorwork_prediction.dataset import create_evaluation_instances as priorwork_instances
from task_priorwork_prediction.baseline_embedding_fusion import create_paper_index, predict_references, get_embedding


def degrade_history(sd2publications, fraction, seed):
    """Return a copy of sd2publications with `fraction` of each author's publications randomly removed (order preserved)."""
    if fraction == 0:
        return sd2publications
    rng = random.Random(seed)
    degraded = {}
    for author_id, pubs in sd2publications.items():
        if not pubs:
            degraded[author_id] = pubs
            continue
        n_drop = int(round(fraction * len(pubs)))
        drop_idx = set(rng.sample(range(len(pubs)), n_drop)) if n_drop else set()
        degraded[author_id] = [p for i, p in enumerate(pubs) if i not in drop_idx]
    return degraded


def score(predicted_ids, gt_ids):
    """nDCG over the full ranking and R-precision (precision@|gt|)."""
    ndcg = utils.calculate_ndcg(predicted_ids, gt_ids)
    r_prec = utils.calculate_precision_recall_f1(predicted_ids[:len(gt_ids)], gt_ids)[0]
    return ndcg, r_prec


def select_indices(n_instances, max_instances, seed):
    """Fixed random subsample of instance indices (matches the baselines' subsampling)."""
    random.seed(seed)
    if max_instances is None or max_instances >= n_instances:
        return set(range(n_instances))
    return set(random.sample(range(n_instances), max_instances))


def run_coauthor(all_papers, sd2publications, all_papers_dict, fractions, args):
    """Coauthor frequency baseline under history degradation."""
    instances = coauthor_instances(all_papers, sd2publications, all_papers_dict, "first")
    selected = select_indices(len(instances), args.max_instances, args.seed)
    utils.log(f"[coauthor] {len(instances)} instances, evaluating {len(selected)}")
    rows = []
    for f in fractions:
        degraded = degrade_history(sd2publications, f, args.seed + int(f * 100))
        ndcgs, rprecs = [], []
        for idx, (date, instance) in enumerate(instances):
            if idx in selected:
                pred_ids, _ = predict_coauthors_one_shot(instance["first_author_id"], args.k, args.num_recent_papers, date, degraded, all_papers_dict)
                n, r = score(pred_ids, instance["gt_coauthor_ids"])
                ndcgs.append(n)
                rprecs.append(r)
        rows.append({"fraction_dropped": f, "n": len(ndcgs), "ndcg": float(np.mean(ndcgs)), "r_precision": float(np.mean(rprecs))})
        utils.log(f"[coauthor] f={f}: nDCG={rows[-1]['ndcg']:.4f} R-prec={rows[-1]['r_precision']:.4f}")
    return rows


def run_priorwork(all_papers, sd2publications, all_papers_dict, all_embeddings, fractions, args):
    """Prior-work embedding-fusion baseline under history degradation (index streamed once, queried per fraction)."""
    distance_metric = "l2" if args.embedding_type == "specter2" else "cosine"
    instances = priorwork_instances(all_papers, sd2publications, all_papers_dict)
    selected = select_indices(len(instances), args.max_instances, args.seed)
    eval_instance_dict = {inst["corpus_id"]: (idx, inst) for idx, (date, inst) in enumerate(instances)}
    utils.log(f"[priorwork] {len(instances)} instances, evaluating {len(selected)}")

    degraded_by_f = {f: degrade_history(sd2publications, f, args.seed + int(f * 100)) for f in fractions}
    accum = {f: {"ndcg": [], "rprec": []} for f in fractions}

    all_papers_sorted = sorted(all_papers, key=lambda p: p["date"])
    first_date = instances[0][0]
    index = create_paper_index(first_date, all_papers_dict, all_embeddings, distance_metric)
    postdated = [p for p in all_papers_sorted if p["date"] >= first_date]
    for paper in utils.tqdm(postdated, desc="[priorwork] streaming"):
        corpus_id = paper["corpus_id"]
        if corpus_id in eval_instance_dict:
            idx, instance = eval_instance_dict[corpus_id]
            if idx in selected:
                for f in fractions:
                    pred_ids, _ = predict_references(instance["author_ids"], paper["date"], args.num_recent_papers, args.k, index, degraded_by_f[f], all_papers_dict, all_embeddings, distance_metric)
                    n, r = score(pred_ids, instance["gt_reference_ids"])
                    accum[f]["ndcg"].append(n)
                    accum[f]["rprec"].append(r)
        index = utils.add_vector_to_index(index, corpus_id, get_embedding(all_embeddings, corpus_id))

    rows = []
    for f in fractions:
        rows.append({"fraction_dropped": f, "n": len(accum[f]["ndcg"]), "ndcg": float(np.mean(accum[f]["ndcg"])), "r_precision": float(np.mean(accum[f]["rprec"]))})
        utils.log(f"[priorwork] f={f}: nDCG={rows[-1]['ndcg']:.4f} R-prec={rows[-1]['r_precision']:.4f}")
    return rows


def render_markdown(results, args):
    """Render the degradation sweep as a drop-in table."""
    lines = [f"# Author-history degradation sensitivity ({args.split}, {args.embedding_type}, n={args.max_instances})", ""]
    lines.append("Randomly drop a fraction of each author's prior publications, re-run on a FIXED instance subsample.")
    lines.append("")
    for task, rows in results.items():
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Fraction dropped | nDCG | R-precision | n |")
        lines.append("|---|---|---|---|")
        base = rows[0]["ndcg"]
        for row in rows:
            delta = f" ({100*(row['ndcg']-base)/base:+.1f}%)" if base else ""
            lines.append(f"| {int(100*row['fraction_dropped'])}% | {row['ndcg']:.4f}{delta} | {row['r_precision']:.4f} | {row['n']:,} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Author-history degradation sensitivity for retrieval tasks")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--embeddings_dir", default="data/corpus/test", help="Directory with embedding pkl files")
    parser.add_argument("--embedding_type", default="grit", choices=["gtr", "grit", "specter2"], help="Embedding type")
    parser.add_argument("--tasks", default="coauthor,priorwork", help="Comma-separated tasks to run")
    parser.add_argument("--fractions", default="0,0.25,0.5,0.75", help="Comma-separated fractions of history to drop")
    parser.add_argument("--max_instances", type=int, default=5000, help="Fixed instance subsample size")
    parser.add_argument("--num_recent_papers", type=int, default=10, help="Recent papers per author")
    parser.add_argument("--k", type=int, default=1000, help="Number of items to predict")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="rebuttal_outputs/C_author_robustness", help="Directory for outputs")
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",")]
    tasks = args.tasks.split(",")
    need_embeddings = "priorwork" in tasks
    all_papers, sd2publications, all_embeddings = utils.load_corpus(split=args.split, embeddings_dir=args.embeddings_dir if need_embeddings else None, embedding_type=args.embedding_type if need_embeddings else None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    utils.log(f"Loaded {len(all_papers)} papers, {len(sd2publications)} authors")

    results = {}
    if "coauthor" in tasks:
        results["coauthor (frequency)"] = run_coauthor(all_papers, sd2publications, all_papers_dict, fractions, args)
    if "priorwork" in tasks:
        results["priorwork (embedding fusion)"] = run_priorwork(all_papers, sd2publications, all_papers_dict, all_embeddings, fractions, args)

    os.makedirs(args.output_dir, exist_ok=True)
    tag = "_".join(t for t in tasks)
    with open(os.path.join(args.output_dir, f"history_degradation.{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    with open(os.path.join(args.output_dir, f"history_degradation.{tag}.md"), "w") as f:
        f.write(render_markdown(results, args))
    utils.log(f"Saved degradation results to {args.output_dir}")


if __name__ == "__main__":
    main()
