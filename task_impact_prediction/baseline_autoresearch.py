"""Autoresearch agent best baseline for impact (citation count) prediction.

XGBoost regressor on log(1+gt_citations), trained on the train split and evaluated on the test split. Features per paper: target embedding + key-reference embedding mean + 28 scalar bibliometric features (author h-index/num_papers/num_citations stats, ref counts, title-abstract length, topic/category counts, target<->ref cosine, year/month).
"""

import os
import random
import argparse
import numpy as np
import xgboost as xgb
from tqdm import tqdm

import utils
from task_impact_prediction.dataset import load_corpus_impact, create_evaluation_instances, get_embedding_dim


def _author_features(authors):
    if not authors:
        return [0.0] * 18
    h = np.array([float(a.get("h_index") or 0) for a in authors])
    np_ = np.array([float(a.get("num_papers") or 0) for a in authors])
    nc = np.array([float(a.get("num_citations") or 0) for a in authors])
    pub_hist_lens = np.array([float(len(a.get("publication_history") or [])) for a in authors])
    first, last = authors[0], authors[-1]

    def _trio(a):
        return [
            float(a.get("h_index") or 0),
            float(np.log1p(float(a.get("num_papers") or 0))),
            float(np.log1p(float(a.get("num_citations") or 0))),
        ]

    return [
        float(len(authors)),
        float(h.max()), float(h.mean()), float(h.std()),
        float(np_.max()), float(np_.mean()),
        float(np.log1p(nc.max())), float(np.log1p(nc.mean())), float(np.log1p(nc.sum())),
        float(pub_hist_lens.max()), float(pub_hist_lens.sum()),
        *(_trio(first)),
        *(_trio(last)),
    ]


def _ref_features(refs):
    if not refs:
        return [0.0] * 6
    nc = np.array([float(r.get("num_citations") or 0) for r in refs])
    return [
        float(len(refs)),
        float(nc.max()), float(nc.mean()),
        float(np.log1p(nc.sum())),
        float(np.median(nc)),
        float((nc >= 100).sum()),
    ]


def _ref_embed_mean(refs, dim):
    vecs = [r["embedding"] for r in refs if r.get("embedding") is not None]
    if not vecs:
        return np.zeros(dim, dtype=np.float32), 0.0
    arr = np.stack(vecs, axis=0).astype(np.float32)
    return arr.mean(axis=0), float(arr.shape[0])


def _featurize(ex, embedding_dim):
    target_emb = ex.get("target_embedding")
    if target_emb is None:
        target_emb = np.zeros(embedding_dim, dtype=np.float32)
    else:
        target_emb = np.asarray(target_emb, dtype=np.float32).reshape(-1)

    refs = ex.get("target_references") or []
    ref_mean, n_ref_emb = _ref_embed_mean(refs, embedding_dim)

    if target_emb.any() and ref_mean.any():
        denom = float(np.linalg.norm(target_emb) * np.linalg.norm(ref_mean))
        cos = float(np.dot(target_emb, ref_mean) / denom) if denom > 0 else 0.0
    else:
        cos = 0.0

    authors = ex.get("target_authors") or []
    title_abs = ex.get("target_title_abstract") or ""
    topics = ex.get("target_topic_labels") or []
    cats = ex.get("target_arxiv_categories") or []

    cutoff = ex.get("cutoff_date") or ""
    try:
        year = float(cutoff[:4])
        month = float(cutoff[5:7])
    except (ValueError, IndexError):
        year, month = 2000.0, 6.0

    scalar = np.array(
        _author_features(authors)
        + _ref_features(refs)
        + [float(len(title_abs)), float(len(title_abs.split()))]
        + [float(len(topics)), float(len(cats)), n_ref_emb, cos]
        + [year, month],
        dtype=np.float32,
    )
    return np.concatenate([target_emb, ref_mean, scalar], axis=0)


def _build_example(paper, all_embeddings):
    """Construct an autoresearch-style example dict from a target paper record."""
    corpus_id = paper["corpus_id"]
    target_emb = all_embeddings[corpus_id]["key"].reshape(-1) if corpus_id in all_embeddings else None

    refs = []
    for r in paper.get("key_references") or []:
        rid = r.get("corpus_id")
        emb = all_embeddings[rid]["key"].reshape(-1) if rid in all_embeddings else None
        refs.append({"corpus_id": rid, "num_citations": r.get("num_citations") or 0, "embedding": emb})

    target_authors = []
    for a in paper.get("authors") or []:
        target_authors.append({
            "h_index": a.get("h_index"),
            "num_papers": a.get("num_papers"),
            "num_citations": a.get("num_citations"),
            "publication_history": a.get("publication_history") or [],
        })

    title_abstract = (paper.get("title") or "") + " " + (paper.get("abstract") or "")

    return {
        "target_embedding": target_emb,
        "target_references": refs,
        "target_authors": target_authors,
        "target_title_abstract": title_abstract,
        "target_topic_labels": paper.get("topic_labels") or [],
        "target_arxiv_categories": paper.get("categories") or [],
        "cutoff_date": paper["date"],
    }


def main():
    parser = argparse.ArgumentParser(description="Autoresearch agent best baseline for impact prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"], help="Dataset split to train on")
    parser.add_argument("--train_embeddings_dir", type=str, default="data/corpus/train", help="Training-split embedding directory")
    parser.add_argument("--test_embeddings_dir", type=str, default="data/corpus/test", help="Test-split embedding directory")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["gtr", "specter2", "grit"], help="Embedding type")
    parser.add_argument("--impact_months", type=int, default=12, help="Months of citation accumulation to use as ground truth")
    parser.add_argument("--max_instances", type=int, default=None, help="If set, subsample the test instances to this many (random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/predictions", help="Output directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_path = os.path.join(args.output_dir, f"predictions.autoresearch.{args.embedding_type}.json")

    utils.log(f"Loading training corpus from {args.hf_repo_id} (split={args.train_split})")
    train_papers, train_dict, train_embeddings, _ = load_corpus_impact(hf_repo_id=args.hf_repo_id, split=args.train_split, embeddings_dir=args.train_embeddings_dir, embedding_type=args.embedding_type)
    utils.log(f"Loaded {len(train_papers)} training papers")

    utils.log(f"Loading test corpus from {args.hf_repo_id} (split={args.split})")
    test_papers, test_dict, test_embeddings, test_metadata = load_corpus_impact(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.test_embeddings_dir, embedding_type=args.embedding_type)
    utils.log(f"Loaded {len(test_papers)} test papers")

    utils.log("Creating evaluation instances")
    train_instances = create_evaluation_instances(train_papers, args.impact_months)
    test_instances = create_evaluation_instances(test_papers, args.impact_months)
    utils.log(f"Train instances: {len(train_instances)}, Test instances: {len(test_instances)}")

    if args.max_instances is not None and args.max_instances < len(test_instances):
        sampled_idx = sorted(random.sample(range(len(test_instances)), args.max_instances))
        test_instances = [test_instances[i] for i in sampled_idx]
        utils.log(f"Subsampled test instances to {len(test_instances)}")

    embedding_dim = get_embedding_dim(test_embeddings)
    utils.log(f"Embedding dim: {embedding_dim}")

    utils.log("Building training examples")
    training_examples = []
    for date, instance in tqdm(train_instances, desc="Train examples"):
        paper = train_dict.get(instance["corpus_id"])
        if paper is None:
            continue
        ex = _build_example(paper, train_embeddings)
        ex["gt_citations"] = instance["gt_citations"]
        training_examples.append(ex)

    utils.log(f"Featurizing {len(training_examples)} training examples")
    X_train = np.stack([_featurize(ex, embedding_dim) for ex in tqdm(training_examples, desc="Featurize train")], axis=0)
    y_train = np.log1p(np.array([float(ex["gt_citations"]) for ex in training_examples], dtype=np.float32))
    fallback = float(np.expm1(np.median(y_train))) if len(y_train) else 0.0

    params = dict(
        n_estimators=200,
        eta=0.08,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbosity=0,
        tree_method="hist",
        n_jobs=-1,
    )
    utils.log(f"Training XGBoost regressor on {X_train.shape[0]} examples ({X_train.shape[1]} features) with params {params}")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    utils.log("Building test examples and predicting")
    predictions = []
    for date, instance in tqdm(test_instances, desc="Predicting"):
        corpus_id = instance["corpus_id"]
        paper = test_dict.get(corpus_id)
        if paper is None:
            predictions.append({"corpus_id": corpus_id, "predicted_citations": fallback})
            continue
        ex = _build_example(paper, test_embeddings)
        x = _featurize(ex, embedding_dim).reshape(1, -1)
        pred = float(np.expm1(model.predict(x)[0]))
        predictions.append({"corpus_id": corpus_id, "predicted_citations": max(pred, 0.0)})

    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(test_metadata, args))


if __name__ == "__main__":
    main()
