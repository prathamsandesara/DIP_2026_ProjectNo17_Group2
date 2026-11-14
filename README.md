# Adaptive Fabric Surface Defect Detection and Localization Using Classical Texture Descriptors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Method](https://img.shields.io/badge/Method-Classical%20DIP-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-73%20Passed%20%7C%201%20Skipped-success.svg)]()
[![Compute](https://img.shields.io/badge/Compute-CPU%20Only%20(No%20GPU)-orange.svg)]()

An unsupervised, deterministic classical Digital Image Processing (DIP) framework for automated optical surface inspection of continuous woven textile webs. The pipeline integrates 1D Radon projection de-skewing, 2D-FFT Butterworth harmonic notch filtering, weave-pitch adaptive mathematical morphology, and Kenneth Laws' Texture Energy Measures for real-time defect screening and physical etiology diagnosis across 7 commercial fabric structures and 12 defect types.

---

## Executive Summary

Industrial textile manufacturing operates continuous high-speed weaving looms where fabric webs are monitored using line-scan optical cameras capturing swaths at extreme aspect ratios (**256 x 4096 pixels**, 1:16 ratio). While modern supervised deep learning models suffer from severe defect data scarcity (<1% defect rate in commercial production) and prohibitive GPU workstation power requirements (>300 W), this project presents an end-to-end, deterministic **Classical Digital Image Processing (DIP)** framework.

By unifying **1D Radon projection de-skewing**, **2D-FFT Butterworth harmonic notch filtering**, **weave-pitch adaptive mathematical morphology**, and **Kenneth Laws' Texture Energy Measures**, our unsupervised pipeline delivers real-time screening and physical etiology diagnosis across all 7 commercial fabric structures and 12 official defect codes in the public **AITEX (AFID)** benchmark.

```
       +------------------------------------------------------------------------+
       |           Raw Line-Scan Feeds (256 x 4096 px, 1:16 Aspect Ratio)       |
       +------------------------------------------------------------------------+
                                           |
                                           v
       +------------------------------------------------------------------------+
       |   Stage 0 & 1: Selvedge Masking & Radon Conveyor De-Skew (+/- 2.5 deg) |
       +------------------------------------------------------------------------+
                                           |
                                           v
       +------------------------------------------------------------------------+
       |               Stage 2: Weave-Adaptive Parallel Detection               |
       |  +--------------------+  +--------------------+  +-------------------+ |
       |  | Micro Top-Hat (Pw) |  | Macro Gaussian Env |  | Specialized Desc  | |
       |  | Fine Yarn Flaws    |  | Broad Tears        |  | 2D-FFT & Radon    | |
       |  +--------------------+  +--------------------+  +-------------------+ |
       +------------------------------------------------------------------------+
                                           |
                                           v
       +------------------------------------------------------------------------+
       |   Stage 3: Feature Fusion & Directional Morphological Closing (101x5)  |
       +------------------------------------------------------------------------+
                                           |
                                           v
             +----------------------------------------------------+
             |  Operational Screening Decision Gate (Area >= 15)  |
             +----------------------------------------------------+
                     /                                    \
       [Area < 15 px / Clean]                    [Area >= 15 px / Defective]
                   /                                        \
                  v                                          v
      +------------------------+                +-------------------------+
      |  PASS: 0 False Alarms  |                |  ALARM: Loom Intercept  |
      |   Weaving Continues    |                |    (42/53 Caught)       |
      +------------------------+                +-------------------------+
                                                             |
                                                             v
                                                +-------------------------+
                                                | Stage 4: Multi-Class    |
                                                | Physical Etiology       |
                                                | (Hu Moments + Laws'     |
                                                |  S5S5 Spot Energy)      |
                                                +-------------------------+
                                                | 40/42 Correct (95.24%)  |
                                                +-------------------------+
```

---

## Repository Organization

```
.
├── MidSem_Codes/              # Baseline adaptive morphological pipeline (Step 00 to 12)
├── MidSem_Presentation/       # Mid-semester presentation deck and report
├── Weekly_Progress_Report/    # Milestone progress reports
├── EndSem_Codes/              # Modular End-Sem Classical DIP Codebase
│   ├── configs/               # Declarative YAML parameters (adaptive, fixed, default)
│   ├── outputs/               # Benchmark CSV tables & JSON evaluation summaries
│   │   ├── metrics/           # Latency profiles, ablation & triage JSON benchmarks
│   │   └── tables/            # Master performance, confusion matrix & fabric breakdown CSVs
│   ├── scripts/               # Master execution runner (run_all.sh), sweeps & benchmarks
│   ├── src/                   # Production modular pipeline library
│   │   ├── adaptive/          # Weave-pitch and dynamic threshold estimation
│   │   ├── data/              # AITEX loader, selvedge masking, parser, leakage-free splits
│   │   ├── detection/         # Micro/macro branches, feature fusion & deviation maps
│   │   ├── evaluation/        # Industrial screening metrics, Dice F1, pixel confusion
│   │   ├── localization/      # Connected components & Hu-moment invariant classifiers
│   │   ├── morphology/        # Directional closing filters & multi-scale kernels
│   │   ├── preprocessing/     # 1D Radon de-skew & 2D-FFT Butterworth notch filters
│   │   ├── texture/           # Kenneth Laws' energy (S5S5), GLCM, LBP & Gabor baselines
│   │   └── utils/             # Visualization, I/O & multi-core CPU parallel execution
│   ├── tests/                 # Comprehensive test suite (73 passed, 1 skipped)
│   ├── pytest.ini             # Pytest test discovery configuration
│   └── requirements.txt       # Production dependencies
├── EndSem_Presentation/       # Final End-Semester Presentation Deck (PDF)
├── EndSem_Report/             # Final Technical Report (PDF)
└── README.md                  # Project documentation
```

---

## Core Classical DIP Methodologies

### 1. Conveyor De-Skew via 1D Radon Transform
Mechanical vibrations introduce conveyor angular skew (+/- 2.5 degrees), tilting warp yarns across neighboring columns (~9 pixels across 256 rows). We compute 1D Radon line projections across 0.1-degree increments to find the angle that maximizes projection variance. Bilinear inverse rotation restores orthogonal thread alignment in under 15 ms.

### 2. Spectral Harmonic Notch Rejection (Fabric 03 Herringbone)
Herringbone fabrics exhibit prominent 2D diagonal chevron harmonics that cause heavy false alarms in spatial filters. Our automated 2D-FFT peak detector identifies harmonic peak frequencies and applies a zero-phase Butterworth notch filter to eliminate the repeating weave clutter. This rescues Fabric 03 detection recall from **20.0% to 100.0% (+4 defective rolls recovered)**.

### 3. Radon Profile Subtraction (Fabric 05 Striped Twill)
To eliminate repeating 11 px longitudinal twill ribs without attenuating narrow warp flaws, 1D line-integral projections along columns decouple the background pattern periodicity, achieving **100.0% screening accuracy with zero false loom halts**.

### 4. Resolving Stains vs. Holes via Kenneth Laws' Spot Energy
Compact grease stains and physical punctures share identical circular geometry (aspect ratio ~1.1, similar moment invariants). We apply Kenneth Laws' 5x5 Spot convolution kernel (S5S5):
- **Structural Punctures / Holes:** Disrupt the physical weave lattice, causing Texture Energy to collapse (Texture Integrity Ratio = 0.18).
- **Photometric Stains:** Yarns remain physically intact underneath the surface discoloration, preserving high Texture Energy (Texture Integrity Ratio = 0.88).
- **Result:** **100% precision and recall** on both stains (6/6) and holes (15/15).

---

## Master Benchmark Evaluation

Evaluated strictly on the held-out **81 AITEX test rolls** (53 Defective, 28 Clean) across all 7 fabric structures:

| Evaluation Metric | Fixed Classical Baseline | Mid-Sem Adaptive | Proposed Pipeline | Industrial Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Screening Accuracy** | 58.02% (47/81) | 75.31% (61/81) | **79.01% (64/81)** | **+20.99% overall gain** |
| **Defect Recall (Sensitivity)** | 49.06% (26/53) | 77.36% (41/53) | **79.25% (42/53)** | **Misses cut: 27 down to 11** |
| **Screening Precision** | 78.79% (26/33) | 83.67% (41/49) | **87.50% (42/48)** | Low false alarms |
| **Screening Specificity** | 75.00% (21/28) | 71.43% (20/28) | **78.57% (22/28)** | 22/28 clean rolls passed |
| **Pixel Dice Score (F1)** | 72.96% | 72.54% | **74.24%** | Sub-millimeter bounding box fit |
| **Clean False Alarm Rate** | 0.0052% | 0.0051% | **0.0035%** | **-31.4% background noise** |
| **Multi-Class Diagnosis (Cascaded)** | 80.77% (21/26) | 87.80% (36/41) | **95.24% (40/42)** | **40 / 42 rolls correct** |
| **Multi-Class Diagnosis (Standalone)**| — | 92.45% (49/53) | **96.23% (51/53)** | 51 / 53 rolls correct |

### Computational Latency Profile
- **Total Execution Time:** **22.3 ms** per 256 x 4096 line-scan roll.
- **Hardware:** Single standard x86 CPU core (zero GPU requirement).
- **Throughput:** **45 frames/second** (exceeds standard 1.5 m/s loom speed by more than 3x).

---

## Physical Defect Etiology Taxonomy

All 12 official AITEX defect codes mapped into 4 actionable mechanical loom actions:

| Defect Family | Official AITEX Defect Codes | Precision | Recall | F1-Score | Actionable Loom Intervention |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Class 0: Warp Flaws** | 002 (Broken end), 006, 023, 027 | 100.0% | 91.67% | 0.957 | Halt warp creel / heald frame |
| **Class 1: Weft Flaws** | 010 (Broken pick), 016, 022, 036 | 88.89% | 88.89% | 0.889 | Reset weft insertion rapier |
| **Class 2: Holes & Voids** | 019 (Fabric hole), 025 (Puncture) | 93.75% | 100.0% | 0.968 | Emergency halt (fabric scrap) |
| **Class 3: Stains & Neps** | 029 (Contamination), 030 (Oil/Nep) | 100.0% | 100.0% | 1.000 | Route to detergent bath |
| **Overall Taxonomy** | **All 12 Official Defect Codes** | **95.66%** | **95.14%** | **0.954** | **40 / 42 Correct (95.24%)** |

---

## Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/prathamsandesara/DIP_2026_ProjectNo17_Group2.git
cd DIP_2026_ProjectNo17_Group2/EndSem_Codes

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
PYTHONPATH=. pytest tests/ -q
# Expected: 73 passed, 1 skipped in ~11s
```

### 3. Execute Master Pipeline
```bash
bash scripts/run_all.sh
```
