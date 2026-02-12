"""Poisson regression baseline for impact prediction with configurable features."""

import os
import random
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import utils
from task_impact_prediction.dataset import (
    load_corpus_impact, create_evaluation_instances, get_papers_for_instances, build_feature_matrix
)

HYPEROPT_SPACE = {
    "n_components": hp.quniform("n_components", 20, 100, 10),
    "alpha": hp.loguniform("alpha", np.log(1e-4), np.log(1e2)),
}


def build_model_name(args):
    """Build model name from feature flags."""
    parts = ["poisson", args.embedding_type]
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


def fit_poisson_model(y, X, alpha):
    """Fit Poisson model with fallback from regularized to standard fit."""
    model = sm.GLM(y, X, family=sm.families.Poisson())
    try:
        result = model.fit_regularized(alpha=alpha, maxiter=200)
        return result
    except Exception:
        pass
    result = model.fit(maxiter=200, disp=False)
    return result


def train_poisson(X_train, y_train, num_evals):
    """Train Poisson regression with hyperopt tuning for PCA components and regularization."""
    val_size = max(1, int(len(X_train) * 0.3))
    train_size = len(X_train) - val_size
    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]

    best_so_far = {"loss": float("inf"), "params": None, "iteration": 0}

    def objective(params):
        n_components = int(params["n_components"])
        n_components = min(n_components, X_tr.shape[1], X_tr.shape[0])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        pca = PCA(n_components=n_components)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        X_tr_c = sm.add_constant(X_tr_pca)
        X_val_c = sm.add_constant(X_val_pca)

        try:
            result = fit_poisson_model(y_tr, X_tr_c, params["alpha"])
            y_pred = result.predict(X_val_c)
            mae = float(np.mean(np.abs(y_pred - y_val)))
        except Exception:
            mae = float("inf")

        best_so_far["iteration"] += 1
        if mae < best_so_far["loss"]:
            best_so_far["loss"] = mae
            best_so_far["params"] = params.copy()
        utils.log(f"[{best_so_far['iteration']}/{num_evals}] MAE={mae:.4f}, best={best_so_far['loss']:.4f}")

        return {"loss": mae, "status": STATUS_OK}

    best = fmin(fn=objective, space=HYPEROPT_SPACE, algo=tpe.suggest, max_evals=num_evals, trials=Trials())
    best["n_components"] = int(best["n_components"])
    best["n_components"] = min(best["n_components"], X_train.shape[1], X_train.shape[0])

    utils.log(f"Best hyperparameters: {best}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=best["n_components"])
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_train_c = sm.add_constant(X_train_pca)
    result = fit_poisson_model(y_train, X_train_c, best["alpha"])

    return result, scaler, pca, best


def load_or_train_model(model_path, X_train, y_train, num_evals):
    """Load model from disk or train a new one."""
    if os.path.exists(model_path):
        utils.log(f"Loading model from {model_path}")
        saved = utils.load_pkl(model_path)[0]
        return saved["result"], saved["scaler"], saved["pca"], saved["best"]

    utils.log(f"Training Poisson regression with {num_evals} hyperopt evaluations")
    result, scaler, pca, best = train_poisson(X_train, y_train, num_evals)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    utils.save_pkl({"result": result, "scaler": scaler, "pca": pca, "best": best}, model_path, overwrite=True)
    utils.log(f"Model saved to {model_path}")

    return result, scaler, pca, best


def predict(result, scaler, pca, X_test, test_corpus_ids):
    """Generate predictions from Poisson model."""
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    X_test_c = sm.add_constant(X_test_pca)

    y_pred = result.predict(X_test_c)
    y_pred = np.maximum(y_pred, 0)

    return [{"corpus_id": cid, "predicted_citations": float(pred)} for cid, pred in zip(test_corpus_ids, y_pred)]


def main():
    parser = argparse.ArgumentParser(description="Poisson regression baseline for impact prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"], help="Dataset split to train on")
    parser.add_argument("--train_embeddings_dir", type=str, default="data/corpus/train", help="Directory containing training split embedding files")
    parser.add_argument("--test_embeddings_dir", type=str, default="data/corpus/test", help="Directory containing test split embedding files")
    parser.add_argument("--embedding_type", type=str, default="gtr", choices=["gtr", "specter2", "grit"], help="Embedding type")
    parser.add_argument("--use_author_names", action="store_true", help="Include author name embeddings")
    parser.add_argument("--use_author_numbers", action="store_true", help="Include author h-index and citations")
    parser.add_argument("--use_author_papers", action="store_true", help="Include author prior paper embeddings")
    parser.add_argument("--use_prior_work_papers", action="store_true", help="Include key reference embeddings")
    parser.add_argument("--use_prior_work_numbers", action="store_true", help="Include key reference citation counts")
    parser.add_argument("--use_followup_work_paper", action="store_true", help="Include paper embedding")
    parser.add_argument("--impact_months", type=int, default=12, help="Number of months to predict impact for")
    parser.add_argument("--num_evals", type=int, default=100, help="Number of hyperopt evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/predictions", help="Output directory")
    args = parser.parse_args()

    feature_flags = [
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper
    ]
    if not any(feature_flags):
        parser.error("At least one feature flag must be set (e.g., --use_followup_work_paper)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = build_model_name(args)
    model_path = os.path.join(args.output_dir, "models", f"{model_name}.pkl")
    author_embedding_cache = {}

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

    X_train = y_train = None
    if not os.path.exists(model_path):
        train_papers_filtered = get_papers_for_instances(train_instances, train_dict, require_key_references=True)
        utils.log(f"Building training features for {len(train_papers_filtered)} papers")
        X_train, train_corpus_ids = build_feature_matrix(
            train_papers_filtered, train_dict, train_embeddings, args.embedding_type,
            args.use_author_names, args.use_author_numbers, args.use_author_papers,
            args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
            author_embedding_cache, desc="Training features"
        )
        corpus_id_to_gt = {inst["corpus_id"]: inst["gt_citations"] for _, inst in train_instances}
        y_train = np.array([corpus_id_to_gt[cid] for cid in train_corpus_ids], dtype=np.float32)
        utils.log(f"Training matrix shape: {X_train.shape}")

    result, scaler, pca, best = load_or_train_model(model_path, X_train, y_train, args.num_evals)

    test_papers_list = get_papers_for_instances(test_instances, test_dict)
    utils.log(f"Building test features for {len(test_papers_list)} papers")
    X_test, test_corpus_ids = build_feature_matrix(
        test_papers_list, test_dict, test_embeddings, args.embedding_type,
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
        author_embedding_cache, desc="Test features"
    )
    utils.log(f"Test matrix shape: {X_test.shape}")

    utils.log("Generating predictions")
    predictions = predict(result, scaler, pca, X_test, test_corpus_ids)

    output_filename = f"predictions.{model_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(test_metadata, args))


if __name__ == "__main__":
    main()
