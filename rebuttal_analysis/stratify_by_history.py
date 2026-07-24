"""Stratify author-dependent task performance by author-history depth (Reviewer 1 W3, Reviewer 4 Q3).

Complements the degradation experiment: instead of removing history, it groups the existing full-test
predictions by how much prior history the relevant author(s) had, to show the tasks are not carried
only by a few deep-history authors. Reads a predictions JSON already on disk, recomputes per-instance
nDCG / R-precision, and reports them binned by history depth."""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import utils
from task_coauthor_prediction.dataset import get_preexisting_publications_for_author

BINS = [(1, 2), (3, 5), (6, 10), (11, 25), (26, 10**9)]


def bin_label(depth):
    """Map a history depth to its bin label."""
    for low, high in BINS:
        if low <= depth <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    return "0"


def load_predictions(path):
    """Load a predictions JSON saved by the baselines (supports metadata-wrapped or bare list)."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def author_depth(instance_authors, date, sd2publications, all_papers_dict):
    """History depth used for stratification: prior-pub count (coauthor: seed author; priorwork: mean over authors)."""
    depths = [len(get_preexisting_publications_for_author(aid, date, sd2publications, all_papers_dict)) for aid in instance_authors]
    return int(round(np.mean(depths))) if depths else 0


def main():
    parser = argparse.ArgumentParser(description="Stratify task performance by author-history depth")
    parser.add_argument("--task", required=True, choices=["coauthor", "priorwork"], help="Task")
    parser.add_argument("--predictions_path", required=True, help="Existing predictions JSON")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output_dir", default="rebuttal_outputs/C_author_robustness", help="Directory for outputs")
    args = parser.parse_args()

    all_papers, sd2publications, _ = utils.load_corpus(split=args.split, embeddings_dir=None, embedding_type=None, load_sd2publications=True)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    predictions = load_predictions(args.predictions_path)
    utils.log(f"Loaded {len(predictions)} predictions for {args.task}")

    gt_key = "gt_coauthor_ids" if args.task == "coauthor" else "gt_reference_ids"
    pred_key = "predicted_coauthor_ids" if args.task == "coauthor" else "predicted_reference_ids"

    strata = defaultdict(lambda: {"ndcg": [], "rprec": []})
    for pred in utils.tqdm(predictions, desc="Stratifying"):
        paper = all_papers_dict[pred["corpus_id"]]
        instance_authors = [pred["first_author_id"]] if args.task == "coauthor" else [a["author_id"] for a in paper["authors"]]
        depth = author_depth(instance_authors, paper["date"], sd2publications, all_papers_dict)
        gt, predicted = pred[gt_key], pred[pred_key]
        strata[bin_label(depth)]["ndcg"].append(utils.calculate_ndcg(predicted, gt))
        strata[bin_label(depth)]["rprec"].append(utils.calculate_precision_recall_f1(predicted[:len(gt)], gt)[0])

    order = [f"{low}-{high}" if high < 10**9 else f"{low}+" for low, high in BINS]
    rows = [{"bin": b, "n": len(strata[b]["ndcg"]), "ndcg": float(np.mean(strata[b]["ndcg"])), "r_precision": float(np.mean(strata[b]["rprec"]))} for b in order if strata[b]["ndcg"]]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"stratification.{args.task}.json"), "w") as f:
        json.dump({"task": args.task, "predictions_path": args.predictions_path, "rows": rows}, f, indent=2)
    lines = [f"# {args.task}: performance by author-history depth", "", f"Source: `{os.path.basename(args.predictions_path)}` (full test set).", "", "| History depth (prior pubs) | nDCG | R-precision | n |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['bin']} | {row['ndcg']:.4f} | {row['r_precision']:.4f} | {row['n']:,} |")
    with open(os.path.join(args.output_dir, f"stratification.{args.task}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    utils.log(f"Saved stratification to {args.output_dir}")


if __name__ == "__main__":
    main()
