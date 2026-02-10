"""Upload PreScience dataset to HuggingFace Hub."""
import os
import argparse
from huggingface_hub import HfApi

import utils

def main():
    parser = argparse.ArgumentParser("Upload PreScience dataset to HuggingFace Hub")
    parser.add_argument("--dataset_dir", type=str, default="data/huggingface", help="Directory with prepared dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g., 'username/prescience')")
    parser.add_argument("--token", type=str, help="HuggingFace token (or use HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required (--token or HF_TOKEN env var)")

    api = HfApi()

    utils.log(f"Creating/updating repository: {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", token=token, private=True, exist_ok=True)

    files_to_upload = [
        ("train.parquet", "Train corpus (373K papers)"),
        ("test.parquet", "Test corpus (465K papers)"),
        ("author_disambiguation.jsonl", "Author disambiguation (S2AND ID → original IDs)"),
        ("author_publications.jsonl", "Author publications (S2AND ID → corpus IDs)"),
        ("README.md", "Dataset card")
    ]

    for filename, description in files_to_upload:
        filepath = os.path.join(args.dataset_dir, filename)
        if os.path.exists(filepath):
            utils.log(f"Uploading {description}: {filename}")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=args.repo_id,
                repo_type="dataset",
                token=token
            )
        else:
            utils.log(f"Warning: {filename} not found, skipping")

    utils.log("Upload complete!")
    utils.log(f"View at: https://huggingface.co/datasets/{args.repo_id}")
    utils.log("Note: Dataset is private. Make it public on HuggingFace website when ready.")

if __name__ == "__main__":
    main()
