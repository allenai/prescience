#!/usr/bin/env bash
# Build one domain/split corpus end-to-end (stages 1-6 + identity merge + invariant check).
# Usage: build_corpus.sh <category_regex> <start_date> <end_date> <output_dir>
set -euo pipefail

CATEGORY="$1"; START="$2"; END="$3"; OUTDIR="$4"
PY=/home/anirudha/miniconda3/envs/scipred2/bin/python
cd /media/8TBNVME/home/anirudha/prescience

echo "=== [$(date)] BUILD $OUTDIR  category=$CATEGORY  $START..$END ==="

echo "--- stage 1: download targets ---"
$PY -m dataset_internal.corpus.download_target_papers --start_date "$START" --end_date "$END" --categories "$CATEGORY" --output_dir "$OUTDIR"
echo "--- stage 2: key references ---"
$PY -m dataset_internal.corpus.add_key_references --input_dir "$OUTDIR" --output_dir "$OUTDIR"
echo "--- stage 3: authors + histories ---"
$PY -m dataset_internal.corpus.add_authors --input_dir "$OUTDIR" --output_dir "$OUTDIR"
echo "--- stage 4: identity merge (no S2AND) ---"
$PY -m dataset_internal.corpus.merge_authors_identity --input_dir "$OUTDIR" --output_dir "$OUTDIR"
echo "--- stage 5: citation metadata ---"
$PY -m dataset_internal.corpus.add_citation_metadata --input_dir "$OUTDIR" --output_dir "$OUTDIR"
echo "--- stage 6: title/abstract from arxiv snapshot ---"
$PY -m dataset_internal.corpus.replace_title_abstracts_using_snapshot --input_dir "$OUTDIR" --output_dir "$OUTDIR"
cp "$OUTDIR/all_papers.stage06.json" "$OUTDIR/all_papers.json"
echo "--- invariant check ---"
$PY -m dataset_internal.corpus.validate_invariants --input_path "$OUTDIR/all_papers.json"

echo "=== [$(date)] DONE $OUTDIR ==="
