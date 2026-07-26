import argparse
import json
import os
import re
from typing import Dict, List

from tqdm import tqdm

import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Replace titles/abstracts in all_papers using a local arXiv snapshot JSON.")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Directory containing all_papers.stage03.json")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Directory where all_papers.stage04.json will be written")
    parser.add_argument("--snapshot_path", type=str, default="data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json", help="Path to the local arXiv snapshot JSON file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def load_snapshot(snapshot_path: str) -> Dict[str, Dict[str, str]]:
    utils.log(f"Loading snapshot from {snapshot_path}")
    records = []
    with open(snapshot_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading snapshot", unit="lines"):
            record = json.loads(line)
            records.append(record)

    utils.log(f"Snapshot contains {len(records)} records. Normalizing metadata")
    snapshot_map: Dict[str, Dict[str, str]] = {}
    for record in tqdm(records, desc="Normalizing snapshot", unit="records"):
        arxiv_id = (record.get("id") or "").strip()
        if not arxiv_id:
            continue
        snapshot_map[arxiv_id] = {
            "title": normalize_text(record.get("title")),
            "abstract": normalize_text(record.get("abstract")),
        }
    utils.log(f"Snapshot provides metadata for {len(snapshot_map)} arXiv IDs")
    return snapshot_map


def refresh_all_papers(all_papers: List[dict], snapshot_map: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    touched = 0
    updated = 0
    missing = 0

    for record in all_papers:
        arxiv_id = (record.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        metadata = snapshot_map.get(arxiv_id)
        if metadata is None:
            missing += 1
            utils.log(f"arXiv ID {arxiv_id} missing from snapshot")
            continue

        if record.get("title") != metadata["title"] or record.get("abstract") != metadata["abstract"]:
            updated += 1
        record["title"] = metadata["title"]
        record["abstract"] = metadata["abstract"]
        touched += 1

    return {"touched": touched, "updated": updated, "missing": missing}


def main() -> None:
    args = parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage05.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage06.json")
    snapshot_path = os.path.abspath(args.snapshot_path)
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot file not found at {snapshot_path}")

    all_papers, metadata = utils.load_json(input_path)
    metadata = metadata if metadata is not None else []

    snapshot_map = load_snapshot(snapshot_path)
    stats = refresh_all_papers(all_papers, snapshot_map)

    utils.save_json(all_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved updated all_papers to {output_path}")
    utils.log(f"Touched {stats['touched']} records; updated {stats['updated']}; missing {stats['missing']}")


if __name__ == "__main__":
    main()
