#!/usr/bin/env bash
# Reproduce every result, figure and document from the raw data (about 20-25 minutes on an 8-core laptop).
# Usage (from the project root):  bash code/scripts/run_all.sh
set -euo pipefail
export PYTHONPATH=code
PY=${PYTHON:-.venv/bin/python}

$PY -m pytest -q                                          # unit tests
$PY code/scripts/verify_splits.py                         # frozen split invariants
$PY code/scripts/run_phase1_alignment.py                  # web alignment (synthetic skew)
$PY code/scripts/run_phase2_frequency_notch.py            # FFT weave suppression
$PY code/scripts/run_latency_benchmark.py                 # measured per-stage latency
$PY code/scripts/run_phase3_benchmark.py                  # 4-stage ablation + sensitivity tables
$PY code/scripts/run_phase4_categorization.py             # Hu-moment taxonomy
$PY code/scripts/run_descriptor_ablation.py               # patch-level GLCM/LBP/Gabor study
$PY code/scripts/run_texture_descriptor_pipeline.py       # full-roll GLCM/LBP/Gabor detector
$PY code/scripts/run_classical_baselines.py               # Otsu / Gaussian / top-hat baselines
$PY code/scripts/generate_presentation_figures.py         # slide figures
$PY code/scripts/generate_endsem_dashboard.py             # master dashboard
$PY code/scripts/generate_report.py                       # docs/REPORT.md, docs/PRESENTATION_PACK.md, README results block
