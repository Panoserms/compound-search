#!/usr/bin/env bash
# End-to-end reproduction script for the compound-search study.
# Checks the data, then runs the unit tests, the Stage 0 baseline, the dev evaluations
# of Stages 1-5, the frozen Stage 6 test evaluation, the Stage 7 error analysis, the
# splitter score, and finally checks that the README tables match the results.
#
# Step 6 repeats the ONE test run with the frozen systems. It is a repeat, not a second
# attempt: never change a system and then run it on the test questions again.
#
# Needs: .venv with `pip install -e ".[dense,dev]"`, and the data from
#   .venv/bin/python -I -m compound_search.download --raw data/raw
# The LLM step reads its stored answers from data/cache/rewrites_*.json; Ollama with
# gemma4:e2b-it-qat is only needed if that file is deleted.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON=".venv/bin/python -I"

echo "=== 0. Checking the downloaded data (SHA-256) ==="
$PYTHON -m compound_search.download --raw data/raw --check

echo ""
echo "=== 1. Running unit test suite (pytest) ==="
$PYTHON -m pytest tests -q

echo ""
echo "=== 2. Stage 0: Data audit & plain BM25 baseline on dev ==="
$PYTHON -m compound_search.stage0 --raw data/raw --out results/stage0_baseline.json

echo ""
echo "=== 3. Stages 1-3: Lexical BM25 systems on dev (1,000 train questions, seed 13) ==="
$PYTHON -m compound_search.dev \
  --raw data/raw \
  --out results/dev \
  --versus bm25_plain,bm25_stem \
  bm25_plain bm25_stem bm25_lemma bm25_split bm25_split_stem bm25_split_lemma bm25_parts_stem bm25_hanta_stem

echo ""
echo "=== 4. Stage 4: Dense (multilingual-e5-small) & RRF hybrids on dev ==="
$PYTHON -m compound_search.dev \
  --raw data/raw \
  --out results/dev \
  --versus bm25_plain,bm25_stem,bm25_split_stem,bm25_parts_stem,dense,hybrid_plain_dense,hybrid_stem_dense \
  dense hybrid_plain_dense hybrid_stem_dense hybrid_split_stem_dense hybrid_parts_stem_dense

echo ""
echo "=== 5. Stage 5: Conditional LLM query rewriting (gemma4:e2b-it-qat) on dev ==="
$PYTHON -m compound_search.dev \
  --raw data/raw \
  --out results/dev \
  --versus bm25_plain,bm25_stem,bm25_split_stem,bm25_parts_stem,dense,hybrid_plain_dense,hybrid_stem_dense,hybrid_split_stem_dense,hybrid_parts_stem_dense \
  hybrid_parts_stem_dense_rewrite hybrid_split_stem_dense_rewrite

echo ""
echo "=== 5b. Two systems for the web demo (added after the test run, dev only) ==="
$PYTHON -m compound_search.dev \
  --raw data/raw \
  --out results/dev \
  --versus bm25_plain,bm25_split,bm25_parts,dense,hybrid_plain_dense \
  bm25_parts hybrid_parts_dense

echo ""
echo "=== 6. Stage 6: Single test set run on the 1,025 GermanDPR test questions ==="
$PYTHON -m compound_search.test_eval \
  --raw data/raw \
  --out results/test \
  --summary results/test_summary.json

echo ""
echo "=== 7. Stage 7: Stratified 50-query error analysis on dev ==="
$PYTHON -m compound_search.error_analysis \
  --raw data/raw \
  --dev-results results/dev \
  --cache data/cache \
  --out-json results/error_analysis_50.json \
  --out-md results/error_analysis_50.md

echo ""
echo "=== 7b. Missed questions: how many may be label gaps (dev) ==="
$PYTHON -m compound_search.residual_check --raw data/raw --out results/residual_check.json

echo ""
echo "=== 8. Splitter score against the labelled 100-word sheet (the author's labels, then the AI's) ==="
$PYTHON -m compound_search.labels score \
  --raw data/raw \
  --labels labels/compounds_labels.csv \
  --out results/splitter_eval.json
$PYTHON -m compound_search.labels score \
  --raw data/raw \
  --labels labels/compounds_labels_ai.csv \
  --out results/splitter_eval_ai_labels.json
$PYTHON -m compound_search.labels agree \
  --first labels/compounds_labels.csv \
  --second labels/compounds_labels_ai.csv \
  --out results/label_agreement.json
$PYTHON -m compound_search.labels score \
  --raw data/raw \
  --labels labels/compounds_labels_first_pass.csv \
  --out results/splitter_eval_first_pass.json
$PYTHON -m compound_search.labels agree \
  --first labels/compounds_labels_first_pass.csv \
  --second labels/compounds_labels_ai.csv \
  --out results/label_agreement_first_pass.json

echo ""
echo "=== 9. README tables and chart against results/ ==="
$PYTHON -m compound_search.tables --results results --check
$PYTHON -m compound_search.figure --results results --out docs/recall_at_10.svg --check
$PYTHON -m compound_search.report_html --results results --labels labels --out docs/report.html

echo ""
echo "Done. All reports written under results/."
