"""SHAP analysis for XGBoost impact prediction model."""

import os
import argparse
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import utils
from task_impact_prediction.dataset import (
    load_corpus_impact, create_evaluation_instances, get_papers_for_instances, build_feature_matrix
)


FEATURE_NAMES_AUTHOR_NUMBERS = [
    "author_first_h_index", "author_first_num_papers", "author_first_avg_citations",
    "author_second_h_index", "author_second_num_papers", "author_second_avg_citations",
    "author_second_last_h_index", "author_second_last_num_papers", "author_second_last_avg_citations",
    "author_last_h_index", "author_last_num_papers", "author_last_avg_citations",
    "author_mean_h_index", "author_mean_num_papers", "author_mean_avg_citations",
    "author_max_h_index", "author_max_num_papers",
    "author_count", "author_first_time_count",
]

FEATURE_NAMES_PRIOR_WORK_NUMBERS = [
    "ref_mean_citations", "ref_max_citations",
]


def get_feature_names(use_author_numbers, use_prior_work_numbers):
    """Get feature names based on enabled flags."""
    names = []
    if use_author_numbers:
        names.extend(FEATURE_NAMES_AUTHOR_NUMBERS)
    if use_prior_work_numbers:
        names.extend(FEATURE_NAMES_PRIOR_WORK_NUMBERS)
    return names


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost impact prediction model")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory with embedding files")
    parser.add_argument("--embedding_type", type=str, default="grit", choices=["gtr", "specter2", "grit"], help="Embedding type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained XGBoost model file")
    parser.add_argument("--impact_months", type=int, default=12, help="Number of months for impact prediction")
    parser.add_argument("--output_dir", type=str, default="data/task_impact_prediction/test/shap_analysis", help="Output directory")
    args = parser.parse_args()

    # Parse feature flags from model filename
    model_name = os.path.basename(args.model_path).replace(".model", "")
    use_author_numbers = "author_numbers" in model_name
    use_prior_work_numbers = "prior_work_numbers" in model_name

    # Validate that we're analyzing a model with only scalar features
    has_embedding_features = any(x in model_name for x in ["author_names", "author_papers", "prior_work_papers", "followup_work_paper"])
    if has_embedding_features:
        utils.log("Warning: Model contains embedding features. SHAP analysis will have many features.")

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    utils.log(f"Loading model from {args.model_path}")
    model = xgb.XGBRegressor()
    model.load_model(args.model_path)

    # Load test corpus from HuggingFace
    utils.log(f"Loading corpus from HuggingFace (repo={args.hf_repo_id}, split={args.split})")
    test_papers, test_dict, test_embeddings, test_metadata = load_corpus_impact(
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        embeddings_dir=args.embeddings_dir,
        embedding_type=args.embedding_type
    )
    utils.log(f"Loaded {len(test_papers)} test papers")

    # Create evaluation instances
    test_instances = create_evaluation_instances(test_papers, args.impact_months)
    utils.log(f"Test instances: {len(test_instances)}")

    # Build test features
    test_papers_list = get_papers_for_instances(test_instances, test_dict)
    utils.log(f"Building test features for {len(test_papers_list)} papers")
    author_embedding_cache = {}
    X_test, test_corpus_ids = build_feature_matrix(
        test_papers_list, test_dict, test_embeddings, args.embedding_type,
        use_author_names=False, use_author_numbers=use_author_numbers, use_author_papers=False,
        use_prior_work_papers=False, use_prior_work_numbers=use_prior_work_numbers, use_followup_work_paper=False,
        author_embedding_cache=author_embedding_cache, desc="Test features"
    )
    utils.log(f"Test matrix shape: {X_test.shape}")

    # Get feature names
    feature_names = get_feature_names(use_author_numbers, use_prior_work_numbers)
    if len(feature_names) != X_test.shape[1]:
        utils.log(f"Warning: Feature count mismatch. Expected {len(feature_names)}, got {X_test.shape[1]}")
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Compute SHAP values using TreeExplainer
    utils.log("Computing SHAP values with TreeExplainer")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    utils.log(f"SHAP values shape: {shap_values.shape}")

    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = sorted(
        [(name, float(val)) for name, val in zip(feature_names, mean_abs_shap)],
        key=lambda x: -x[1]
    )

    # Save JSON summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "model_path": args.model_path,
        "model_name": model_name,
        "num_instances": len(test_corpus_ids),
        "num_features": len(feature_names),
        "expected_value": float(explainer.expected_value),
        "feature_importance": [{"feature": name, "mean_abs_shap": val} for name, val in feature_importance],
    }
    summary_path = os.path.join(args.output_dir, "shap_summary.json")
    utils.save_json(summary, summary_path, metadata=utils.update_metadata(test_metadata, args))
    utils.log(f"Saved summary to {summary_path}")

    # Create beeswarm plot
    utils.log("Creating SHAP summary plot (beeswarm)")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(args.output_dir, "shap_summary_plot.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved beeswarm plot to {beeswarm_path}")

    # Create bar plot
    utils.log("Creating SHAP bar plot")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(args.output_dir, "shap_bar_plot.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    utils.log(f"Saved bar plot to {bar_path}")

    # Print top features
    utils.log("Top 10 features by mean |SHAP|:")
    for name, val in feature_importance[:10]:
        utils.log(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    main()
