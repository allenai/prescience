"""Two-stage XGBoost baseline for impact prediction (classifier + regressor)."""

import os
import gc
import random
import argparse
import numpy as np
import torch
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import utils
from task_impact_prediction.dataset import (
    load_corpus_impact, create_evaluation_instances, get_papers_for_instances, build_feature_matrix
)

REGRESSOR_SPACE = {
    "eta": hp.loguniform("eta", np.log(0.01), np.log(0.2)),
    "max_depth": hp.quniform("max_depth", 3, 7, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
    "subsample": hp.uniform("subsample", 0.5, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.8),
    "gamma": hp.uniform("gamma", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
}

CLASSIFIER_SPACE = {
    "eta": hp.loguniform("eta", np.log(0.01), np.log(0.2)),
    "max_depth": hp.quniform("max_depth", 3, 7, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
    "subsample": hp.uniform("subsample", 0.5, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.8),
    "gamma": hp.uniform("gamma", 0, 5),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "scale_pos_weight": hp.loguniform("scale_pos_weight", np.log(0.5), np.log(5)),
}


def build_model_name(args):
    """Build model name from feature flags."""
    parts = ["xgboost_two_stage", args.embedding_type]
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


def split_train_val(X, y, val_fraction=0.3):
    """Split data into train and validation sets."""
    val_size = max(1, int(len(X) * val_fraction))
    val_size = min(val_size, len(X) - 1)
    train_size = len(X) - val_size
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def train_classifier(X_train, y_train, num_evals, device):
    """Train XGBoost classifier with hyperopt tuning."""
    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train)

    def objective(params):
        trial_params = params.copy()
        trial_params["max_depth"] = int(trial_params["max_depth"])
        trial_params["min_child_weight"] = int(trial_params["min_child_weight"])

        model = xgb.XGBClassifier(
            **trial_params, objective="binary:logistic", eval_metric="logloss",
            verbosity=0, device=device, use_label_encoder=False
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        loss = -np.mean(y_val * np.log(probs + 1e-12) + (1 - y_val) * np.log(1 - probs + 1e-12))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"loss": loss, "status": STATUS_OK}

    best = fmin(fn=objective, space=CLASSIFIER_SPACE, algo=tpe.suggest, max_evals=num_evals, trials=Trials())
    best["max_depth"] = int(best["max_depth"])
    best["min_child_weight"] = int(best["min_child_weight"])

    utils.log(f"Best classifier hyperparameters: {best}")

    model = xgb.XGBClassifier(
        **best, objective="binary:logistic", eval_metric="logloss",
        verbosity=0, device=device, use_label_encoder=False
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def train_regressor(X_train, y_train, num_evals, device):
    """Train XGBoost regressor with hyperopt tuning."""
    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train)

    def objective(params):
        trial_params = params.copy()
        trial_params["max_depth"] = int(trial_params["max_depth"])
        trial_params["min_child_weight"] = int(trial_params["min_child_weight"])

        model = xgb.XGBRegressor(**trial_params, verbosity=0, device=device)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mse = float(np.mean((preds - y_val) ** 2))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"loss": mse, "status": STATUS_OK}

    best = fmin(fn=objective, space=REGRESSOR_SPACE, algo=tpe.suggest, max_evals=num_evals, trials=Trials())
    best["max_depth"] = int(best["max_depth"])
    best["min_child_weight"] = int(best["min_child_weight"])

    utils.log(f"Best regressor hyperparameters: {best}")

    model = xgb.XGBRegressor(**best, verbosity=0, device=device)
    model.fit(X_train, y_train, verbose=False)
    return model


def load_or_train_classifier(model_path, X_train, y_is_positive, num_evals, device):
    """Load classifier from disk or train a new one."""
    if os.path.exists(model_path):
        utils.log(f"Loading classifier from {model_path}")
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", verbosity=0, use_label_encoder=False)
        model.load_model(model_path)
        return model

    utils.log(f"Training classifier with {num_evals} hyperopt evaluations")
    model = train_classifier(X_train, y_is_positive, num_evals, device)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    utils.log(f"Classifier saved to {model_path}")
    return model


def load_or_train_regressor(model_path, X_positive, y_positive, num_evals, device):
    """Load regressor from disk or train a new one."""
    if os.path.exists(model_path):
        utils.log(f"Loading regressor from {model_path}")
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    if X_positive is None or len(X_positive) == 0:
        raise ValueError("No positive training examples available for regressor training.")

    utils.log(f"Training regressor with {num_evals} hyperopt evaluations")
    model = train_regressor(X_positive, y_positive, num_evals, device)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    utils.log(f"Regressor saved to {model_path}")
    return model


def predict_two_stage(classifier, regressor, X_test, test_corpus_ids):
    """Generate predictions using two-stage model."""
    positive_probs = classifier.predict_proba(X_test)[:, 1]
    will_cite = positive_probs >= 0.5

    predictions = []
    positive_indices = np.where(will_cite)[0]

    if len(positive_indices) > 0:
        X_positive_test = X_test[positive_indices]
        y_pred_log = regressor.predict(X_positive_test)
        y_pred_positive = np.maximum(0, np.expm1(y_pred_log))

        positive_iter = iter(y_pred_positive)
        for i, corpus_id in enumerate(test_corpus_ids):
            if will_cite[i]:
                predictions.append({"corpus_id": corpus_id, "predicted_citations": float(next(positive_iter))})
            else:
                predictions.append({"corpus_id": corpus_id, "predicted_citations": 0.0})
    else:
        for corpus_id in test_corpus_ids:
            predictions.append({"corpus_id": corpus_id, "predicted_citations": 0.0})

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Two-stage XGBoost baseline for impact prediction")
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

    # Validate at least one feature flag is set
    feature_flags = [
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper
    ]
    if not any(feature_flags):
        parser.error("At least one feature flag must be set (e.g., --use_followup_work_paper)")

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = build_model_name(args)
    classifier_path = os.path.join(args.output_dir, "models", f"{model_name}_clf.model")
    regressor_path = os.path.join(args.output_dir, "models", f"{model_name}_reg.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Build training features (only if models don't exist)
    X_full = y_is_positive = X_positive = y_positive = None
    needs_training = not os.path.exists(classifier_path) or not os.path.exists(regressor_path)

    if needs_training:
        train_papers_filtered = get_papers_for_instances(train_instances, train_dict, require_key_references=True)
        random.shuffle(train_papers_filtered)
        utils.log(f"Building training features for {len(train_papers_filtered)} papers")

        X_full, train_corpus_ids = build_feature_matrix(
            train_papers_filtered, train_dict, train_embeddings, args.embedding_type,
            args.use_author_names, args.use_author_numbers, args.use_author_papers,
            args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
            author_embedding_cache, desc="Training features"
        )
        corpus_id_to_gt = {inst["corpus_id"]: inst["gt_citations"] for _, inst in train_instances}
        y_full = np.array([corpus_id_to_gt[cid] for cid in train_corpus_ids], dtype=np.float32)
        y_is_positive = (y_full > 0).astype(int)

        positive_mask = y_is_positive == 1
        X_positive = X_full[positive_mask]
        y_positive = np.log1p(y_full[positive_mask]).astype(np.float32)

        utils.log(f"Training matrix shape: {X_full.shape}, Positive samples: {np.sum(positive_mask)}/{len(y_full)}")

    # Load or train models
    classifier = load_or_train_classifier(classifier_path, X_full, y_is_positive, args.num_evals, device)
    regressor = load_or_train_regressor(regressor_path, X_positive, y_positive, args.num_evals, device)

    # Build test features
    test_papers_list = get_papers_for_instances(test_instances, test_dict)
    utils.log(f"Building test features for {len(test_papers_list)} papers")
    X_test, test_corpus_ids = build_feature_matrix(
        test_papers_list, test_dict, test_embeddings, args.embedding_type,
        args.use_author_names, args.use_author_numbers, args.use_author_papers,
        args.use_prior_work_papers, args.use_prior_work_numbers, args.use_followup_work_paper,
        author_embedding_cache, desc="Test features"
    )
    utils.log(f"Test matrix shape: {X_test.shape}")

    # Predict and save
    utils.log("Generating predictions")
    predictions = predict_two_stage(classifier, regressor, X_test, test_corpus_ids)

    output_filename = f"predictions.{model_name}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    utils.log(f"Saving {len(predictions)} predictions to {output_path}")
    utils.save_json(predictions, output_path, metadata=utils.update_metadata(test_metadata, args))


if __name__ == "__main__":
    main()
