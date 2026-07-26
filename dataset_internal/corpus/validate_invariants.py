"""Validate the 10 dataset invariants (CLAUDE.md) on a built all_papers stage file."""
import os
import argparse

import utils


def main():
    parser = argparse.ArgumentParser("Validate PreScience dataset invariants on a built corpus.")
    parser.add_argument("--input_path", type=str, default="data/corpus/test/all_papers.stage06.json", help="Path to the all_papers stage JSON to validate")
    args = parser.parse_args()

    all_papers, _ = utils.load_json(args.input_path)
    papers = {p["corpus_id"]: p for p in all_papers}
    targets = [p for p in all_papers if "target" in p["roles"]]
    pub_histories = [p for p in all_papers if "target.author.publication_history" in p["roles"]]
    utils.log(f"Loaded {len(all_papers)} papers ({len(targets)} targets, {len(pub_histories)} publication-history papers)")

    failures = {}

    failures["1 targets have non-empty key_references"] = [p["corpus_id"] for p in targets if not p.get("key_references")]
    failures["2 targets have non-empty authors"] = [p["corpus_id"] for p in targets if not p.get("authors")]

    inv3 = []
    for p in targets:
        for author in p["authors"]:
            for pub_id in author.get("publication_history", []):
                if pub_id not in papers or papers[pub_id]["date"] >= p["date"]:
                    inv3.append((p["corpus_id"], author["author_id"], pub_id))
    failures["3 target authors' publication_history predates target and exists"] = inv3

    failures["4 pub-history papers have key_references field"] = [p["corpus_id"] for p in pub_histories if "key_references" not in p]
    failures["5 pub-history papers have non-empty authors"] = [p["corpus_id"] for p in pub_histories if not p.get("authors")]

    inv6 = []
    for p in all_papers:
        for ref in p.get("key_references", []):
            if ref["corpus_id"] not in papers:
                inv6.append((p["corpus_id"], ref["corpus_id"]))
        for author in p.get("authors", []):
            for pub_id in author.get("publication_history", []):
                if pub_id not in papers:
                    inv6.append((p["corpus_id"], pub_id))
    failures["6 all referenced papers exist in corpus"] = inv6

    reachable = set()
    for p in all_papers:
        if "target" in p["roles"]:
            reachable.add(p["corpus_id"])
        for ref in p.get("key_references", []):
            reachable.add(ref["corpus_id"])
        for author in p.get("authors", []):
            for pub_id in author.get("publication_history", []):
                reachable.add(pub_id)
    failures["7 no orphan papers (all are target or referenced)"] = [cid for cid in papers if cid not in reachable]

    inv8 = [(p["corpus_id"], ref["corpus_id"]) for p in targets for ref in p["key_references"] if "num_citations" not in ref]
    failures["8 target key_references have num_citations"] = inv8

    inv9 = [(p["corpus_id"], a["author_id"]) for p in targets for a in p["authors"] if not all(k in a for k in ("h_index", "num_papers", "num_citations"))]
    failures["9 target authors have h_index/num_papers/num_citations"] = inv9

    failures["10 targets have citation_trajectory"] = [p["corpus_id"] for p in targets if "citation_trajectory" not in p]

    print("\n" + "=" * 70)
    all_pass = True
    for name, bad in failures.items():
        status = "PASS" if not bad else f"FAIL ({len(bad)} violations, e.g. {bad[:3]})"
        if bad:
            all_pass = False
        print(f"  [{'ok ' if not bad else 'XX '}] Invariant {name}: {status}")
    print("=" * 70)
    print("ALL INVARIANTS PASS" if all_pass else "SOME INVARIANTS FAILED")


if __name__ == "__main__":
    main()
