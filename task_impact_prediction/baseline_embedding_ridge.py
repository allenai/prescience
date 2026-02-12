"""Ridge regression baseline for impact prediction with configurable features."""

import os
import argparse
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from tqdm import tqdm

import utils
from task_impact_prediction.dataset import (
    load_corpus_impact, create_evaluation_instances, get_papers_for_instances, build_feature_matrix
)


def build_model_name(args):
    """Build model name from feature flags."""
    parts = ["ridge", args.embedding_type]
    if args.use_author_names:
        parts.append("author_names")
    if args.use_author_numbers:
        parts.append("author_numbers")
    if args.use_author_papers:
        parts.append("author_papers")
    if args.use_prior_work_papers:
        parts.append("prior_work_papers")
    if args.use_prior_work_numbers:
        parts.append("prior_work_numbers")
    if args.use_followup_work_paper:
        parts.append("followup_work_paper")
    return "_".join(parts)


def build_embedding_matrix(instances, embeddings):
    """Build feature matrix from paper embeddings for evaluation instances."""
    X = []
    corpus_ids = []
    for _, instance in tqdm(instances, desc="Building embedding matrix"):
        corpus_id = instance["corpus_id"]
        if corpus_id in embeddings:
            X.append(embeddings[corpus_id]["key"].reshape(-1))
            corpus_ids.append(corpus_id)
    return np.array(X, dtype=np.float32), corpus_ids


def build_target_vector(instances, corpus_ids_in_X):
    """Build target vector aligned with feature matrix."""
    corpus_id_to_gt = {inst["corpus_id"]: inst["gt_citations"] for _, inst in instances}
    y = [corpus_id_to_gt[cid] for cid in corpus_ids_in_X]
    return np.array(y, dtype=np.float32)


HYPEROPT_SPACE = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(100000)),
}


def train_ridge(X_train, y_train, num_evals):
    """Train Ridge regression with hyperopt tuning."""
    val_size = max(1, int(len(X_train) * 0.3))
    train_size = len(X_train) - val_size
    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]

    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    best_so_far = {"loss": float("inf"), "alpha": None, "iteration": 0}

    def objective(params):
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            Ridge(alpha=params["alpha"], random_state=42)
        )
        model.fit(X_tr, y_tr_log)
        preds = model.predict(X_val)
        mse = float(np.mean((preds - y_val_log) ** 2))

        best_so_far["iteration"] += 1
        if mse < best_so_far["loss"]:
            best_so_far["loss"] = mse
            best_so_far["alpha"] = params["alpha"]
        utils.log(f"[{best_so_far['iteration']}/{num_evals}] Best so far: alpha={best_so_far['alpha']:.6f}, loss={best_so_far['loss']:.6f}")

        return {"loss": mse, "status": STATUS_OK}

    best = fmin(fn=objective, space=HYPEROPT_SPACE, algo=tpe.suggest, max_evals=num_evals, trials=Trials())
    utils.log(f"Best hyperparameters: {best}")

    y_train_log = np.log1p(y_train)
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=best["alpha"], random_state=42)
    )
    model.fit(X_train, y_train_log)
    return model


def predict(model, X_test):
    """Generate predictions and transform back from log space."""
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    return np.maximum(y_pred, 0)


def has_feature_flags(args):
    """Check if any feature flags are set."""
    return any([
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper
    ])


def main():
    parser = argparse.ArgumentParser(description="Ridge regression baseline for impact prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"], help="Dataset split to train on")
    parser.add_argument("--train_embeddings_dir", type=str, default="data/corpus/train", help="Directory containing training split embedding files")
    parser.add_argument("--test_embeddings_dir", type=str, default="data/corpus/test", help="Directory containing test split embedding files")
    parser.add_argument("--embedding_type", type=str, default="gtr", choices=["gtr", "specter2", "grit"], help="Embedding type to use")
    parser.add_argument("--use_author_names", action="store_true", help="Include author name embeddings")
    parser.add_argument("--use_author_numbers", action="store_true", help="Include author h-index and citations")
    parser.add_argument("--use_author_papers", action="store_true", help="Include author prior paper embeddings")
    parser.add_argument("--use_prior_work_papers", action="store_true", help="Include key reference embeddings")
    parser.add_argument("--use_prior_work_numbers", action="store_true", help="Include key reference citation counts")
    parser.add_argument("--use_followup_work_paper", action="store_true", help="Include paper embedding")
    parser.add_argument("--impact_months", type=int, default=12, help="Number of months to predict impact for")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of hyperopt evaluations")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/predictions", help="Output directory")
    args = parser.parse_args()

    use_feature_flags = has_feature_flags(args)
    model_name = build_model_name(args) if use_feature_flags else f"ridge_{args.embedding_type}"
    author_embedding_cache = {}

    # Load corpora
    utils.log(f"Loading training corpus from {args.hf_repo_id} (split={args.train_split})")
    train_papers, train_dict, train_embeddings, _ = load_corpus_impact(hf_repo_id=args.hf_repo_id, split=args.train_split, embeddings_dir=args.train_embeddings_dir, embedding_type=args.embedding_type)
    utils.log(f"Loaded {len(train_papers)} training papers")

    utils.log(f"Loading test corpus from {args.hf_repo_id} (split={args.split})")
    test_papers, test_dict, test_embeddings, test_metadata = load_corpus_impact(hf_repo_id=args.hf_repo_id, split=args.split, embeddings_dir=args.test_embeddings_dir, embedding_type=args.embedding_type)
    utils.log(f"Loaded {len(test_papers)} test papers")

    # Create evaluation instances
    utils.log("Creating evaluation instances")
    train_instances = create_evaluation_instances(train_papers, args.impact_months)
    test_instances = create_evaluation_instances(test_papers, args.impact_months)
    utils.log(f"Train instances: {len(train_instances)}, Test instances: {len(test_instances)}")

    # Build feature matrices
    if use_feature_flags:
        # Use configurable features
        train_papers_list = get_papers_for_instances(train_instances, train_dict)
        utils.log(f"Building training features for {len(train_papers_list)} papers")
        X_train, train_corpus_ids = build_feature_matrix(
            train_papers_list, train_dict, train_embeddings, args.embedding_type,
            args.use_author_names, args.use_author_numbers, args.use_author_papers,
            args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
            author_embedding_cache, desc="Training features"
        )
        y_train = build_target_vector(train_instances, train_corpus_ids)
        utils.log(f"Training matrix shape: {X_train.shape}")

        test_papers_list = get_papers_for_instances(test_instances, test_dict)
        utils.log(f"Building test features for {len(test_papers_list)} papers")
        X_test, test_corpus_ids = build_feature_matrix(
            test_papers_list, test_dict, test_embeddings, args.embedding_type,
            args.use_author_names, args.use_author_numbers, args.use_author_papers,
            args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
            author_embedding_cache, desc="Test features"
        )
        utils.log(f"Test matrix shape: {X_test.shape}")
    else:
        # Simple embedding-only mode (original behavior)
        utils.log("Building training features")
        X_train, train_corpus_ids = build_embedding_matrix(train_instances, train_embeddings)
        y_train = build_target_vector(train_instances, train_corpus_ids)
        utils.log(f"Training matrix shape: {X_train.shape}")

        utils.log("Building test features")
        X_test, test_corpus_ids = build_embedding_matrix(test_instances, test_embeddings)
        utils.log(f"Test matrix shape: {X_test.shape}")

    # Train and predict
    utils.log(f"Training Ridge regression with {args.num_evals} hyperopt evaluations")
    model = train_ridge(X_train, y_train, args.num_evals)

    utils.log("Generating predictions")
    y_pred = predict(model, X_test)

    # Format and save predictions
    predictions = [{"corpus_id": cid, "predicted_citations": float(pred)} for cid, pred in zip(test_corpus_ids, y_pred)]

    output_filename = f"predictions.{model_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(test_metadata, args))


if __name__ == "__main__":
    main()
