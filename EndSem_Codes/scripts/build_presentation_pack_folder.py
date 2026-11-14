#!/usr/bin/env python
"""
Build the complete End-Semester presentation package in:
`END_SEM_PRESENTATION FIGURES AND TABLES/`

Generates:
1. 14 High-Resolution Slide Showcase Images (numbered by slide order)
2. `TABLES_FOR_END_SEM_PRESENTATION.md` containing all slide tables formatted
   in clean GitHub-flavored Markdown mirroring and improving the Mid-Sem style.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.io import load_image_gray, load_mask_binary

DEST = Path("END_SEM_PRESENTATION FIGURES AND TABLES")
DEST.mkdir(parents=True, exist_ok=True)


def generate_7_weaves_gallery() -> Path:
    """Generate 7-panel gallery of all nominal AITEX fabric weaves."""
    print("Generating 7 AITEX Weaves gallery...")
    fabrics = [
        ("00", "Fabric 00: Poplin", "Isotropic 1:1 Orthogonal Grid", "2311694-2040n7u/0001_000_00.png"),
        ("01", "Fabric 01: Twill", "45° Diagonal Rib Wales", "2608691-202020u/0001_000_01.png"),
        ("02", "Fabric 02: Satin", "Smooth Specular Filament", "2306894-210033u/0001_000_02.png"),
        ("03", "Fabric 03: Herringbone", "Periodic Chevron Zigzag", "2311694-1930c7u/0001_000_03.png"),
        ("04", "Fabric 04: Poplin II", "Dense High-Count Plain Weave", "2311517-195063u/0001_000_04.png"),
        ("05", "Fabric 05: Striped Twill", "Longitudinal Rib Channels (90°)", "2306881-210020u/0001_000_05.png"),
        ("06", "Fabric 06: Drill Twill", "Steep 63° Heavy Diagonal Wales", "2311980-185026u/0001_000_06.png"),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(20, 3.8), facecolor="white")
    for idx, (fid, name, desc, rel_path) in enumerate(fabrics):
        p = Path("data/raw/NODefect_images") / rel_path
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        crop = img[:, 1920:2176] if img is not None and img.shape[1] >= 2176 else np.zeros((256, 256), dtype=np.uint8)
        
        axes[idx].imshow(crop, cmap="gray")
        axes[idx].set_title(f"{name}", fontsize=10.5, fontweight="bold", pad=6, color="#801B19")
        axes[idx].set_xlabel(f"{desc}", fontsize=8.2, fontweight="medium", labelpad=4)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.suptitle("AITEX Fabric Database: 7 Industrial Weave Types Evaluated (256x256 Native Micro-Crops)",
                 fontsize=13, fontweight="bold", y=1.04, color="#1E293B")
    plt.tight_layout()
    out_p = DEST / "13_slide4_aitex_7_weaves_gallery.png"
    plt.savefig(out_p, dpi=220, bbox_inches="tight")
    plt.close()
    return out_p


def generate_midsem_vs_endsem_evolution_chart() -> Path:
    """Generate comparison bar chart of Mid-Sem vs End-Sem metrics."""
    print("Generating Mid-Sem vs End-Sem Evolution comparison chart...")
    metrics = [
        "Screening\nAccuracy",
        "Defect\nRecall",
        "Screening\nPrecision",
        "Screening\nSpecificity",
        "Pooled Pixel\nDice",
        "Stain Defect\nRecall",
        "Multi-Class\nTaxonomy Acc",
    ]
    midsem_vals = [75.31, 77.36, 83.67, 71.43, 72.54, 50.00, 88.68]
    endsem_vals = [79.01, 79.25, 87.50, 78.57, 74.24, 100.00, 96.23]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor="white")
    rects1 = ax.bar(x - width/2, midsem_vals, width, label="Mid-Semester Baseline (Weave-Adaptive)", color="#94A3B8")
    rects2 = ax.bar(x + width/2, endsem_vals, width, label="End-Semester Complete Pipeline", color="#801B19")

    ax.set_ylabel("Metric Performance (%)", fontsize=11, fontweight="bold")
    ax.set_title("Quantitative Evolution: Mid-Semester Review Commitments vs. Final End-Semester Delivery",
                 fontsize=13, fontweight="bold", pad=12, color="#801B19")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#475569")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#801B19")

    plt.tight_layout()
    out_p = DEST / "11_slide2_midsem_vs_endsem_evolution.png"
    plt.savefig(out_p, dpi=220, bbox_inches="tight")
    plt.close()
    return out_p


def generate_confusion_matrices_showcase() -> Path:
    """Generate side-by-side screening & taxonomy confusion matrices."""
    print("Generating screening & taxonomy confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")

    # Screening Matrix (81 test rolls: 53 Defective, 28 Clean)
    cm_screen = np.array([[22, 6], [11, 42]]) # TN=22, FP=6, FN=11, TP=42
    im0 = axes[0].imshow(cm_screen, cmap="Greens")
    axes[0].set_title("Operational Screening Confusion Matrix\n(Optimal τ* = 15 px | 81 Rolls)", fontsize=11, fontweight="bold", color="#801B19")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Clean Roll (Pass)", "Defective Roll (Stop)"], fontweight="bold", fontsize=9.5)
    axes[0].set_yticklabels(["Clean (GT)", "Defective (GT)"], fontweight="bold", fontsize=9.5)
    axes[0].set_xlabel("Predicted Screening Decision", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("True Ground Truth Label", fontweight="bold", fontsize=10)

    for i in range(2):
        for j in range(2):
            val = cm_screen[i, j]
            label = f"{val}\n(" + (("TN" if j==0 else "FP") if i==0 else ("FN" if j==0 else "TP")) + ")"
            col = "white" if val > 20 else "black"
            axes[0].text(j, i, label, ha="center", va="center", fontsize=11, fontweight="bold", color=col)

    # Multi-Class Taxonomy Matrix (53 Defective Rolls: 51/53 correct)
    cm_tax = np.array([
        [12, 1, 0, 0],
        [0, 10, 1, 0],
        [0, 0, 23, 0],
        [0, 0, 0, 6]
    ])
    im1 = axes[1].imshow(cm_tax, cmap="Purples")
    axes[1].set_title("Multi-Class Taxonomy Confusion Matrix\n(Hu Moments & Laws' Energy | 96.23% Acc)", fontsize=11, fontweight="bold", color="#801B19")
    class_labels = ["Warp Flaw", "Weft Flaw", "Micro-Hole", "Oil Stain"]
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(class_labels, fontweight="bold", fontsize=8.5, rotation=15)
    axes[1].set_yticklabels(class_labels, fontweight="bold", fontsize=8.5)
    axes[1].set_xlabel("Predicted Defect Family", fontweight="bold", fontsize=10)
    axes[1].set_ylabel("True Defect Family", fontweight="bold", fontsize=10)

    for i in range(4):
        for j in range(4):
            val = cm_tax[i, j]
            col = "white" if val > 8 else ("black" if val > 0 else "#CBD5E1")
            axes[1].text(j, i, str(val), ha="center", va="center", fontsize=11, fontweight="bold", color=col)

    plt.tight_layout()
    out_p = DEST / "12_slide7_screening_confusion_matrices.png"
    plt.savefig(out_p, dpi=220, bbox_inches="tight")
    plt.close()
    return out_p


def copy_and_organize_assets() -> None:
    """Copy all high-res presentation visual assets into destination with numbered prefixes."""
    mapping = {
        "outputs/visualizations/presentation/slide_asset_problem_definition.png": "01_slide3_problem_definition.png",
        "outputs/visualizations/presentation/slide_asset_pipeline_architecture.png": "02_slide5_pipeline_architecture.png",
        "outputs/visualizations/presentation/special_descriptor_fabric03_fft_notch.png": "03_slide6_fabric03_fft_notch_filtering.png",
        "outputs/visualizations/presentation/special_descriptor_fabric05_radon.png": "04_slide6_fabric05_radon_rib_suppression.png",
        "outputs/visualizations/presentation/slide_asset_endsem_results_showcase.png": "05_slide7_endsem_results_showcase.png",
        "outputs/visualizations/presentation/slide_asset_multiclass_showcase.png": "06_slide8_multiclass_defect_taxonomy.png",
        "outputs/visualizations/presentation/special_descriptor_laws_stain_vs_hole.png": "07_slide8_laws_energy_stains_vs_holes.png",
        "outputs/visualizations/presentation/slide_asset_oil_stain_showcase.png": "08_slide8_oil_stain_nep_diagnosis.png",
        "outputs/visualizations/presentation/special_descriptor_fabric01_06_directional_morphology.png": "09_slide9_directional_twill_morphology.png",
        "outputs/visualizations/endsem/fig_threshold_hyperparameter_sweep.png": "10_slide10_threshold_hyperparameter_sweep.png",
        "outputs/visualizations/endsem/fig_alignment_verification.png": "14_slide5_radon_deskew_alignment.png",
    }

    for src_path_str, dst_name in mapping.items():
        src_p = Path(src_path_str)
        dst_p = DEST / dst_name
        if src_p.exists():
            shutil.copy2(src_p, dst_p)
            print(f"Copied: {dst_name}")
        else:
            print(f"Warning: source {src_path_str} not found!")


def write_presentation_tables_markdown() -> None:
    """Write comprehensive TABLES_FOR_END_SEM_PRESENTATION.md matching Mid-Sem style."""
    md_content = """# Master Tables for ECE501 End-Semester Presentation
**Project:** Adaptive Fabric Surface Defect Detection and Localization using Classical Texture Descriptors  
**Course:** ECE501: Digital Image Processing | Ahmedabad University (SEAS)  
**Team (Group 2, Project 17):** Pratham Sandesara (AU2340239), Even Patel (AU2340241), Dharman Patel (AU2340237), Shine Shah (AU2440148)  
**Mentor:** Prof. Mehul Raval  
**Color Code Theme:** Ahmedabad University Maroon `#801B19` (`RGB: 128, 27, 25`), Clean White `#FFFFFF`, Slate Charcoal `#1E293B`

---

## 📋 Table Index
- **Table 1:** Audit of Progress: Mid-Semester Commitments vs. Final End-Semester Delivery *(Slide 2)*
- **Table 2:** Dataset Specification: AITEX Database & Official 12 Defect Codes *(Slide 4)*
- **Table 3:** Master Benchmark Comparison: Fixed Control vs. Mid-Sem Adaptive vs. End-Sem Pipeline *(Slide 6 / 7)*
- **Table 4:** Operational Roll Triage / Confusion Matrix Breakdown *(Slide 7)*
- **Table 5:** Comprehensive Fabric-by-Fabric Performance Breakdown across All 7 Weaves *(Slide 7)*
- **Table 6:** Multi-Class Defect Taxonomy Performance (Laws' TEM & Hu Invariants) *(Slide 8)*
- **Table 7:** Classical Texture Descriptors Specialization Matrix (General vs Fabric-Specific) *(Slide 9)*
- **Table 8:** Hyperparameter Sensitivity Sweep across Screening Thresholds $\\tau \\in \\{10, 15, 20, 25\\text{ px}\\}$ *(Slide 10)*
- **Table 9:** Real-Time Computational Efficiency & Pipeline Stage Latency Profile *(Slide 11)*
- **Table 10:** Individual Contribution Matrix *(Slide 12)*

---

### Table 1: Audit of Progress — Mid-Sem Commitments vs. End-Sem Delivery
*(Formatted for Slide 2: Directly addressing evaluator feedback)*

| Evaluator Feedback / Mid-Sem Commitment | Classical DIP Strategy Implemented | Final End-Sem Verified Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Improve Weak Defect Classes**<br>*(Stains and small holes confused due to identical Aspect Ratio)* | Integrated **Kenneth Laws' Texture Energy Measures ($S_5S_5$ Spot Energy)** and **Normalized Boundary Acutance (NBA)**. | **Stain Recall: 50.0% → 100.0% (6/6)**<br>**Hole Recall: 100.0% (23/23)**<br>**Overall Taxonomy: 88.7% → 96.2%** | **DELIVERED** |
| **2. Hyperparameter Sensitivity Sweep**<br>*(Investigate stability of pixel area threshold $\\tau$)* | Conducted systematic operational sweep across **$\\tau \\in \\{10, 15, 20, 25\\text{ px}\\}$** on all 81 test rolls. | Confirmed **$\\tau^* = 15\\text{ px}$ optimal plateau** ($79.01\\%$ Acc, $87.50\\%$ Precision, $74.24\\%$ Dice). | **DELIVERED** |
| **3. Reframe Problem Statement**<br>*(Articulate industrial line-scan and DIP scope)* | Reframed from generic anomaly detection to **Etiology-Specific Structural Integrity Analysis**. | Fully documented in `docs/REFRAMED_PROBLEM_STATEMENT.md`. | **DELIVERED** |
| **4. Texture Descriptor Augmentation**<br>*(Verify if specialized descriptors improve results)* | Deployed **2D-FFT Butterworth Harmonic Notch Filter** + **Radon Directional Projections**. | **Fabric 03 Recall: 20% → 100% (+4 rolls rescued)**<br>**Fabric 05 Accuracy: 100% (0 False Alarms)** | **DELIVERED** |

---

### Table 2: Dataset Specification — AITEX Fabric Database & 12 Official Codes
*(Formatted for Slide 4: Upgrading Mid-Sem Slide 4 Table with full defect taxonomy)*

| Attribute / Specification | Value / Description | Project Significance & DIP Role |
| :--- | :--- | :--- |
| **Source & Modality** | Industrial Line-Scan Camera | High aspect ratio ($1:16$), grayscale ($256$ gray levels) |
| **Image Resolution** | **$256 \\times 4096\\text{ pixels}$** | Full native resolution preserves single-yarn micro-flaws |
| **Total Database Size** | **$247\\text{ Images}$** | $141$ Defect-Free rolls | $106$ Defective rolls |
| **Evaluated Benchmark Split** | **$81\\text{ Test Rolls}$** | **$53\\text{ Defective Rolls}$ | $28\\text{ Defect-Free Rolls}$** |
| **Ground Truth Annotations** | Pixel-Level Binary Masks | Exact spatial ground truth ($0$ background vs $255$ defect) |
| **Weave Diversity** | **7 Industrial Weaves (00–06)** | Plain Poplin, Twills ($45^\\circ, 63^\\circ$), Satin, Herringbone, Striped |
| **Full Defect Taxonomy** | **12 Official AITEX Codes** | Broken end (`002`), Broken yarn (`006`), Broken pick (`010`), Weft curling (`016`), Fuzzyball/Hole (`019`), Cut selvage (`022`), Crease (`023`), Warp ball (`025`), Knots (`027`), Contamination (`029`), Nep/Oil Spot (`030`), Weft crack (`036`) |
| **Defect Families (Etiologies)** | **4 Distinct Classes** | Class 0: Warp Flaws | Class 1: Weft Flaws | Class 2: Micro-Punctures | Class 3: Photometric Stains |

---

### Table 3: Master Benchmark Comparison — Mid-Sem vs. End-Sem Evolution
*(Formatted for Slide 6 / 7: Upgrading Mid-Sem Slide 6 Results Table)*

| Evaluation Tier | Performance Metric | Fixed Control Baseline (15x15) | Mid-Sem Adaptive Baseline | End-Sem Complete DIP Pipeline | Delta vs. Mid-Sem |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Image-Level (Screening)** | **Screening Accuracy** | $58.02\\%$ | $75.31\\%$ | **79.01%** | **+3.70%** *(+21.0% vs Fixed)* |
| | **Defect Sensitivity (Recall)** | $49.06\\%$ | $77.36\\%$ | **79.25%** | **+1.89%** *(+30.2% vs Fixed)* |
| | **Screening Precision** | $78.79\\%$ | $83.67\\%$ | **87.50%** | **+3.83%** *(+8.7% vs Fixed)* |
| | **Screening Specificity** | $75.00\\%$ | $71.43\\%$ | **78.57%** | **+7.14%** *(+3.6% vs Fixed)* |
| **Pixel-Level (Localization)** | **Pixel Dice (F1-Score)** | $72.96\\%$ | $72.54\\%$ | **74.24%** | **+1.70%** |
| | **Pixel Precision** | $80.29\\%$ | $78.56\\%$ | **79.20%** | **+0.64%** |
| | **Pixel Recall** | $66.85\\%$ | $67.38\\%$ | **69.86%** | **+2.48%** |
| **Noise Safety** | **Clean False Alarm Rate** | $0.0052\\%$ | $0.0051\\%$ | **0.0035%** | **-31.4% less noise** |
| **Taxonomy Diagnosis** | **Multi-Class Accuracy** | N/A *(Binary only)* | $88.68\\%$ | **96.23%** | **+7.55% (51/53 Correct)** |

---

### Table 4: Operational Roll Triage / Confusion Matrix Breakdown
*(Formatted for Slide 7: Side-by-side with Table 3 matching Mid-Sem presentation)*

| Roll Triage Category | Industrial Operational Meaning | Fixed Control Baseline | Mid-Sem Adaptive | End-Sem Complete Pipeline | End-Sem Operational Impact |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **True Positive (TP)** | Genuinely defective rolls successfully caught | $26 / 53$ | $41 / 53$ | **42 / 53** | **Defects Intercepted: Prevents flawed fabric dispatch** |
| **False Negative (FN)** | Defective rolls missed (shipped to client) | $27 / 53$ | $12 / 53$ | **11 / 53** | **Escapes Slashed by 59.3% vs Fixed Baseline** |
| **True Negative (TN)** | Clean rolls correctly verified and passed | $21 / 28$ | $20 / 28$ | **22 / 28** | **High Factory Throughput: 78.57% clean pass rate** |
| **False Positive (FP)** | Clean rolls falsely flagged (unnecessary stop) | $7 / 28$ | $8 / 28$ | **6 / 28** | **False Loom Halts Reduced to only 6 rolls** |
| **Total Test Rolls** | Complete AITEX Benchmark Evaluation | **81 Rolls** | **81 Rolls** | **81 Rolls** | **Verified with zero cherry-picking** |

---

### Table 5: Comprehensive Fabric-by-Fabric Performance Breakdown
*(Formatted for Slide 7: Complete transparency across all 7 weave structures)*

| Fabric ID | Industrial Weave Pattern | Screening Accuracy | Screening Recall | Screening Specificity | Screening Precision | Pixel Dice Score | Confusion Matrix (TP / FP / TN / FN) | Engineering Highlight |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fabric 00** | Plain Weave Poplin | **78.57%** | 80.00% | 75.00% | 88.89% | **88.24%** | $8 / 1 / 3 / 2$ | Crisp orthogonal thread boundaries |
| **Fabric 01** | Coarse Textured Twill ($45^\\circ$) | **71.43%** | 80.00% | 50.00% | 80.00% | **86.37%** | $8 / 2 / 2 / 2$ | Directional top-hat suppresses diagonal wales |
| **Fabric 02** | Fine Filament Satin | **85.71%** | 80.00% | **100.0%** | **100.0%** | **77.96%** | $8 / 0 / 4 / 2$ | **Zero False Alarms on mirror-smooth filament** |
| **Fabric 03** | Herringbone Twill (Chevron) | **77.78%** | **100.0%** | 50.00% | 71.43% | **69.80%** | $5 / 2 / 2 / 0$ | **2D-FFT Notch rescued 4 rolls (100% Recall!)** |
| **Fabric 04** | Plain Poplin II (Dense Grid) | **78.57%** | 70.00% | **100.0%** | **100.0%** | **78.47%** | $7 / 0 / 4 / 3$ | **Zero False Alarms on dense orthogonal weave** |
| **Fabric 05** | Striped Twill (Longitudinal) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **38.24%** | $1 / 0 / 4 / 0$ | **100% Screening Perfection via Radon Projection** |
| **Fabric 06** | Heavy Drill Twill ($63^\\circ$) | **72.73%** | 71.43% | 75.00% | 83.33% | **79.88%** | $5 / 1 / 3 / 2$ | Angle-tuned morphology ($63^\\circ$) suppresses steep ribs |
| **OVERALL** | **Macro Average / Total** | **79.01%** | **79.25%** | **78.57%** | **87.50%** | **74.24%** | **42 / 6 / 22 / 11** | **Industrial Precision: 87.50% (81 Rolls)** |

---

### Table 6: Multi-Class Defect Taxonomy Performance (53 Defective Rolls)
*(Formatted for Slide 8: Validating Kenneth Laws' TEM and Boundary Acutance)*

| Defect Class ID | Defect Family Name | AITEX Defect Codes Represented | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score | Physical DIP Classification Signature |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Class 0** | **Warp Flaws (Longitudinal)** | `002` (Broken end), `006` (Broken yarn), `023` (Crease), `027` (Knots) | $12 / 13$ | $0$ | $1$ | **100.0%** | **92.31%** | **0.960** | Extreme vertical elongation: $AR > 2.8, \\theta \\approx 90^\\circ$ |
| **Class 1** | **Weft Flaws (Transverse)** | `010` (Broken pick), `016` (Weft curling), `022` (Cut selvage), `036` (Weft crack) | $10 / 11$ | $1$ | $1$ | **90.91%** | **90.91%** | **0.909** | Extreme horizontal elongation: $AR < 0.35, \\theta \\approx 0^\\circ$ |
| **Class 2** | **Micro-Punctures (Holes)** | `019` (Fuzzyball / Hole), `025` (Warp ball) | $23 / 23$ | $1$ | $0$ | **95.83%** | **100.0%** | **0.979** | Compact shape ($AR \\approx 1.1$) + **Zero Laws' Spot Energy ($TIR < 0.55$)** + Razor Edge (NBA $> 30$) |
| **Class 3** | **Photometric Stains (Neps)** | `029` (Contamination), `030` (Nep / Oil Spot) | $6 / 6$ | $0$ | $0$ | **100.0%** | **100.0%** | **1.000** | Compact shape ($AR \\approx 1.1$) + **Intact Laws' Spot Energy ($TIR \\ge 0.55$)** + Soft Penumbra (NBA $< 30$) |
| **OVERALL** | **Macro Average / Total** | **All 12 AITEX Defect Codes** | **51 / 53** | **2** | **2** | **96.69%** | **95.81%** | **0.962** | **Overall Taxonomy Accuracy: 96.23% (51/53 Correct)** |

---

### Table 7: Classical Texture Descriptors Specialization Matrix
*(Formatted for Slide 9: Explaining why general descriptors were augmented with fabric-specific modules)*

| Fabric ID | Weave Texture Type | Classical DIP Descriptors Deployed | Specialization Tier | Mathematical Rationale & Industrial Result |
| :---: | :--- | :--- | :---: | :--- |
| **Fabric 00** | Plain Weave Poplin | Multi-Scale White/Black Top-Hat + Hu Moments | General Baseline | Orthogonal grid yields clean morphological response; 88.24% Pixel Dice. |
| **Fabric 01** | Coarse Textured Twill ($45^\\circ$) | Dual-Scale Top-Hat + $45^\\circ$ Linear Structuring Element | **Fabric-Specific** | Linear SE aligned at $45^\\circ$ suppresses diagonal rib false alarms; 86.37% Dice. |
| **Fabric 02** | Fine Filament Satin | Selvedge Margin Blanking + Dynamic Otsu | General Baseline | Mirror-smooth filament lacks periodic texture; achieves **100% Specificity**. |
| **Fabric 03** | Herringbone Twill (Chevron) | **2D-FFT Butterworth Spectral Notch Filter** | **Fabric-Specific** | **Critical Innovation:** Zeros periodic chevron harmonics; **Recall: 20% → 100%**. |
| **Fabric 04** | Plain Poplin II (Dense Grid) | Multi-Scale Top-Hat + Area Morphological Opening | General Baseline | Dense orthogonal yarn packing handled by dual-scale filters; **100% Specificity**. |
| **Fabric 05** | Striped Twill (Longitudinal) | **1D Radon Transform Projection ($\\theta = 90^\\circ$)** | **Fabric-Specific** | Subtraction of $90^\\circ$ column projection profile yields **100% Accuracy & Specificity**. |
| **Fabric 06** | Heavy Drill Twill ($63^\\circ$) | Angle-Tuned Morphological Filtering ($63^\\circ$) | **Fabric-Specific** | Linear SE aligned at $63^\\circ$ suppresses steep twill wales; 79.88% Pixel Dice. |
| **All Fabrics** | **Stains vs. Holes** | **Kenneth Laws' $S_5S_5$ Energy + Boundary Acutance** | **Universal Taxonomy** | Solves stain vs hole ambiguity across all fabrics; **Taxonomy Accuracy: 96.23%**. |

---

### Table 8: Hyperparameter Sensitivity Sweep across Screening Thresholds
*(Formatted for Slide 10: Addressing Evaluator Requirement for 10, 15, 20, 25 px Analysis)*

| Screening Threshold ($\\tau$) | Screening Accuracy | Defect Recall | Screening Specificity | Screening Precision | Pooled Pixel Dice | Operating Characteristic & Industrial Recommendation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\\tau = 10\\text{ px}$** | **79.01%** | **81.13%** ($43/53$) | 75.00% ($21/28$) | 86.00% | 74.24% | *High Sensitivity Mode:* Catches tiny fiber specks; slightly higher false alarms on coarse weaves. |
| **$\\tau = 15\\text{ px}$ (Optimal $\\tau^*$)** | **79.01%** | **79.25%** ($42/53$) | **78.57%** ($22/28$) | **87.50%** | **74.24%** | **Selected Operating Point:** Optimal balance of precision ($87.50\\%$) and specificity ($78.57\\%$). |
| **$\\tau = 20\\text{ px}$** | 75.31% | 73.58% ($39/53$) | 78.57% ($22/28$) | 86.67% | 74.24% | *Conservative Mode:* Prunes 3 minor broken ends; precision remains high. |
| **$\\tau = 25\\text{ px}$** | 75.31% | 71.70% ($38/53$) | **82.14%** ($23/28$) | **88.37%** | 74.24% | *Ultra-Conservative:* Peak specificity ($82.14\\%$), but misses subtle single-yarn pick flaws. |

---

### Table 9: Real-Time Computational Efficiency & Stage Latency Profile
*(Formatted for Slide 11: Deployment readiness and edge hardware profiling)*

| Pipeline Stage | Algorithm / Operation Deployed | Average Latency (ms) | Relative Compute Share | Computational Complexity | Edge Hardware Advantage |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 0: Photometric Conditioning** | Active Selvedge Border Blanking | $1.4\\text{ ms}$ | $6.4\\%$ | $\\mathcal{O}(M \\times N)$ | Linear memory pass |
| **Phase 1: Frequency & Alignment** | Radon De-Skew / 2D-FFT Notch Filter | $7.8\\text{ ms}$ | $35.5\\%$ | $\\mathcal{O}(MN \\log(MN))$ | FFTW / OpenCV optimized |
| **Phase 2: Mathematical Morphology** | Dual-Scale White & Black Top-Hats | $6.2\\text{ ms}$ | $28.2\\%$ | $\\mathcal{O}(M \\times N \\times K)$ | Van Herk / Gil-Werman separable SE |
| **Phase 3: Screening Triage** | Dynamic Otsu & Connected Components | $3.5\\text{ ms}$ | $15.9\\%$ | $\\mathcal{O}(M \\times N)$ | Instant 1D histogram threshold |
| **Phase 4: Multi-Class Diagnosis** | Hu Moments & Laws' $S_5S_5$ Energy | $3.1\\text{ ms}$ | $14.0\\%$ | $\\mathcal{O}(\\text{Defect ROI})$ | Evaluates localized bounding boxes only |
| **TOTAL PIPELINE LATENCY** | **Deterministic End-to-End Execution** | **22.0 ms / roll** | **100.0%** | **Real-Time Throughput** | **45.4 FPS on CPU (Zero GPU Required!)** |

> **Industrial Conveyor Velocity Verification:**  
> Standard industrial weaving looms run at line speeds of $1.0\\text{ to }1.5\\text{ m/s}$. At $22\\text{ ms}$ per inspection strip ($256 \\times 4096\\text{ px}$), the fabric advances by only $0.033\\text{ m}$. Our pipeline is **$>3\\times$ faster than maximum loom speeds**, proving deterministic real-time viability.

---

### Table 10: Individual Contribution Matrix
*(Formatted for Slide 12: Updating Mid-Sem individual contributions)*

| Student Name | Student ID | Core Technical Contributions | Presentation Delivery Area |
| :--- | :---: | :--- | :--- |
| **Pratham Sandesara** | **AU2340239** | Statistical Thresholding, Screening Sensitivity Sweep ($\\tau \\in \\{10, 15, 20, 25\\text{ px}\\}$), Master Benchmark Evaluation & Multi-Class Integration | Problem Definition, Hyperparameter Stability & Master Results |
| **Even Patel** | **AU2340241** | Adaptive Multi-Scale Mathematical Morphology, Directional Top-Hat Kernels ($45^\\circ, 63^\\circ$), Dual-Scale Structuring Elements | Morphological Filtering & Diagonal Twill Specialization |
| **Dharman Patel** | **AU2340237** | 2D-FFT Spectral Harmonic Notch Filtering, Automated Peak Detection, Fabric 03 Chevron Pattern Suppression & AITEX Dataset Preparation | Frequency Domain Filtering & Herringbone Weave Rescue |
| **Shine Shah** | **AU2440148** | Radon Line-Integral Projections, Conveyor De-Skew Alignment, Kenneth Laws' Texture Energy Measures ($S_5S_5$ Spot Energy) & Boundary Acutance Profiling | Alignment De-Skew, Laws' TEM & Stain vs. Hole Separation |
"""
    out_file = DEST / "TABLES_FOR_END_SEM_PRESENTATION.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved: {out_file}")


def main() -> None:
    print("Building complete End-Sem presentation package...")
    generate_7_weaves_gallery()
    generate_midsem_vs_endsem_evolution_chart()
    generate_confusion_matrices_showcase()
    copy_and_organize_assets()
    write_presentation_tables_markdown()
    print("\nPackage generated successfully in 'END_SEM_PRESENTATION FIGURES AND TABLES/'")


if __name__ == "__main__":
    main()

