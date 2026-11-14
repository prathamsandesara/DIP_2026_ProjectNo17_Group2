#!/usr/bin/env python
"""
Generate a complete, professional, 16:9 widescreen PowerPoint presentation:
`ECE501_Grp2_EndSem_Presentation.pptx`

Features:
- Ahmedabad University Maroon (#801B19) theme & styling
- Native editable tables formatted with clean typography and alternating row fills
- High-resolution showcase images embedded directly from `END_SEM_PRESENTATION FIGURES AND TABLES/`
- Full speaker notes embedded on each slide for oral defense
- Ready to open directly in Microsoft PowerPoint, Apple Keynote, or import to Google Slides
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Paths
ASSET_DIR = Path("END_SEM_PRESENTATION FIGURES AND TABLES")
OUTPUT_PPTX = Path("ECE501_Grp2_EndSem_Presentation.pptx")
LOGO_WHITE = Path("au_logo_white.png")
LOGO_MAROON = Path("au_logo_maroon.png")

# Palette
COLOR_MAROON = RGBColor(128, 27, 25)      # AU Maroon #801B19
COLOR_DARK = RGBColor(30, 41, 59)          # Slate Charcoal #1E293B
COLOR_MUTED = RGBColor(100, 116, 139)      # Muted Slate #64748B
COLOR_LIGHT_BG = RGBColor(248, 250, 252)   # Card BG #F8FAFC
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_GREEN = RGBColor(16, 185, 129)       # Emerald #10B981
COLOR_RED = RGBColor(239, 68, 68)          # Crimson #EF4444
COLOR_ROW_ALT = RGBColor(241, 245, 249)    # Table alternate row


def create_header(slide, title_text: str, category_text: str = "ECE501: DIGITAL IMAGE PROCESSING") -> None:
    """Standardized modern slide header."""
    # Category pill
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(9.5)
    p_c.font.bold = True
    p_c.font.color.rgb = COLOR_MAROON

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.5), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(20)
    p_t.font.bold = True
    p_t.font.color.rgb = COLOR_DARK

    # Divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.73), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(226, 232, 240)
    line.line.color.rgb = RGBColor(226, 232, 240)


def create_footer(slide, current_slide: int, total_slides: int = 13) -> None:
    """Standardized modern slide footer."""
    foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(10), Inches(0.3))
    tf = foot_box.text_frame
    p = tf.paragraphs[0]
    p.text = "School of Engineering & Applied Science (SEAS), Ahmedabad University | Group 2"
    p.font.size = Pt(8.5)
    p.font.color.rgb = COLOR_MUTED

    num_box = slide.shapes.add_textbox(Inches(11.0), Inches(7.0), Inches(1.5), Inches(0.3))
    tf_n = num_box.text_frame
    p_n = tf_n.paragraphs[0]
    p_n.alignment = PP_ALIGN.RIGHT
    p_n.text = f"{current_slide} / {total_slides}"
    p_n.font.size = Pt(8.5)
    p_n.font.bold = True
    p_n.font.color.rgb = COLOR_MAROON


def add_table(slide, left, top, width, height, headers, data, col_widths=None) -> None:
    """Render a beautiful PowerPoint table with custom colors and typography."""
    rows = len(data) + 1
    cols = len(headers)
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    tbl = table_shape.table

    if col_widths:
        for idx, w in enumerate(col_widths):
            tbl.columns[idx].width = w

    # Headers
    for c_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_MAROON
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    # Data
    for r_idx, row_vals in enumerate(data):
        row_num = r_idx + 1
        is_alt = (r_idx % 2 == 1)
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(row_num, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_ROW_ALT if is_alt else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_DARK
            # Left align first column, center others
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER


def build_presentation() -> None:
    print("Building full ECE501 End-Semester PowerPoint presentation...")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Solid AU Maroon)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_MAROON
    bg1.line.color.rgb = COLOR_MAROON

    if LOGO_WHITE.exists():
        slide1.shapes.add_picture(str(LOGO_WHITE), Inches(0.9), Inches(0.9), width=Inches(1.2))

    t_box = slide1.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "ECE501: DIGITAL IMAGE PROCESSING | END-SEMESTER EVALUATION"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(254, 205, 211)

    p1 = tf1.add_paragraph()
    p1.text = "Adaptive Fabric Surface Defect Detection and Localization Using Classical Texture Descriptors"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(8)

    p2 = tf1.add_paragraph()
    p2.text = "An Etiology-Driven Classical DIP Pipeline with Dual-Scale Screening and Moment-Laws Taxonomy"
    p2.font.size = Pt(13)
    p2.font.color.rgb = RGBColor(241, 245, 249)
    p2.space_before = Pt(6)

    # Authors card
    auth_box = slide1.shapes.add_textbox(Inches(0.9), Inches(4.7), Inches(11.5), Inches(1.8))
    tf_a = auth_box.text_frame
    tf_a.word_wrap = True
    pa1 = tf_a.paragraphs[0]
    pa1.text = "TEAM (GROUP 2, TOPIC 17):"
    pa1.font.size = Pt(9.5)
    pa1.font.bold = True
    pa1.font.color.rgb = RGBColor(254, 205, 211)

    pa2 = tf_a.add_paragraph()
    pa2.text = "Pratham Sandesara (AU2340239)   •   Even Patel (AU2340241)   •   Dharman Patel (AU2340237)   •   Shine Shah (AU2440148)"
    pa2.font.size = Pt(12)
    pa2.font.bold = True
    pa2.font.color.rgb = COLOR_WHITE
    pa2.space_before = Pt(4)

    pa3 = tf_a.add_paragraph()
    pa3.text = "Project Mentor: Prof. Mehul Raval   |   School of Engineering and Applied Science (SEAS), Ahmedabad University"
    pa3.font.size = Pt(10.5)
    pa3.font.color.rgb = RGBColor(226, 232, 240)
    pa3.space_before = Pt(6)

    slide1.notes_slide.notes_text_frame.text = (
        "Good morning respected evaluators. Today we present our End-Semester Project for ECE501 Digital Image Processing. "
        "In this project, we demonstrate automated fabric surface defect detection and localization on the industrial AITEX "
        "benchmark without relying on opaque deep neural networks. Instead, we formulate a principled, classical digital image "
        "processing pipeline combining Radon directional projections, 2D-FFT spectral notch filtering, Kenneth Laws' Texture "
        "Energy Measures, and Hu moment invariants."
    )

    # =========================================================================
    # SLIDE 2: PROGRESS AUDIT (Mid-Sem Commitments vs End-Sem Delivery)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    create_header(slide2, "Progress Audit: Mid-Semester Commitments vs. Final Delivery", "EVALUATION ALIGNMENT")
    create_footer(slide2, 2)

    headers2 = ["Mid-Sem Evaluator Feedback", "Classical DIP Solution Implemented", "Verified End-Sem Result", "Status"]
    data2 = [
        ["1. Improve Weak Defect Classes\n(Stains vs Holes confused)", "Kenneth Laws' Texture Energy (S5S5 Spot Energy) + Normalized Boundary Acutance (NBA)", "Stain Recall: 50.0% -> 100.0% (6/6)\nHole Recall: 100.0% (23/23)\nOverall Taxonomy: 88.7% -> 96.2%", "DELIVERED"],
        ["2. Hyperparameter Sweep\n(Screening threshold sensitivity)", "Systematic operational sweep across tau in {10, 15, 20, 25 px} on all 81 test rolls", "Confirmed optimal plateau at tau* = 15 px (79.01% Acc, 87.5% Prec, 74.2% Dice)", "DELIVERED"],
        ["3. Reframe Problem Statement\n(Articulate industrial DIP scope)", "Reframed around line-scan 1:16 aspect ratio & physical manufacturing etiologies", "Formalized in REFRAMED_PROBLEM_STATEMENT.md with zero CNN dependency", "DELIVERED"],
        ["4. Augment Texture Descriptors\n(Check if descriptors boost metrics)", "Deployed 2D-FFT Butterworth Harmonic Notch Filter + Radon Angular Projections", "Fabric 03 Recall: 20% -> 100% (+4 rescued rolls)\nFabric 05: 100% Acc / 100% Spec", "DELIVERED"],
    ]
    widths2 = [Inches(2.5), Inches(3.8), Inches(3.8), Inches(1.5)]
    add_table(slide2, Inches(0.8), Inches(1.6), Inches(11.73), Inches(4.8), headers2, data2, widths2)

    slide2.notes_slide.notes_text_frame.text = (
        "During our mid-semester review, the evaluators posed four critical challenges: improving weak classes like oil spots, "
        "rigorously evaluating detection threshold stability across 10, 15, 20, and 25 pixels, refining the industrial line-scan "
        "problem framing, and validating texture descriptor improvements. Every single commitment has been rigorously engineered, "
        "tested across all 81 benchmark rolls, and verified."
    )

    # =========================================================================
    # SLIDE 3: PROBLEM DEFINITION & INDUSTRIAL CONTEXT
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    create_header(slide3, "Industrial Challenge: Inspecting Extreme Aspect-Ratio Fabric Webs", "PROBLEM DEFINITION")
    create_footer(slide3, 3)

    # Left: Text card
    tbox3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(4.8), Inches(4.8))
    tf3 = tbox3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        ("Acquisition Reality:", "Industrial line-scan cameras capture continuous rolls at 256 x 4096 pixels (1:16 aspect ratio) moving at 1.0 to 1.5 m/s."),
        ("Sub-Millimeter Flaw Scarcity:", "A subtle broken yarn or nep occupies fewer than 100 pixels out of >1,000,000 pixels (<0.01% anomaly area)."),
        ("Why Classical DIP Over Deep Learning:", "Standard CNN input resizing (224x224 or 512x512) completely obliterates fine yarn threads. Furthermore, mills lack thousands of labeled defect examples."),
        ("Deterministic Edge Advantage:", "Our classical pipeline executes in 22.0 ms per roll on standard CPU without GPUs, providing mathematically provable interpretability."),
    ]
    for idx, (head, body) in enumerate(bullets3):
        p = tf3.paragraphs[0] if idx == 0 else tf3.add_paragraph()
        p.text = head + " "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_MAROON
        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(9.5)
        run.font.color.rgb = COLOR_DARK
        p.space_after = Pt(8)

    # Right: Image
    img3_p = ASSET_DIR / "01_slide3_problem_definition.png"
    if img3_p.exists():
        slide3.shapes.add_picture(str(img3_p), Inches(5.8), Inches(1.8), width=Inches(6.7))

    slide3.notes_slide.notes_text_frame.text = (
        "In textile optical inspection, web rolls have an extreme 1:16 aspect ratio. Downsampling these images for CNNs destroys "
        "single-yarn defects. Our classical DIP pipeline works at native native resolution, executing deterministically in 22 ms "
        "on low-power edge CPUs without requiring any GPU infrastructure."
    )

    # =========================================================================
    # SLIDE 4: DATASET & 12 OFFICIAL DEFECT TAXONOMY
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    create_header(slide4, "Benchmark Dataset: 7 Weaves & 12 Official AITEX Defect Codes", "BENCHMARK SPECIFICATION")
    create_footer(slide4, 4)

    # Top: Gallery of 7 weaves
    img4_p = ASSET_DIR / "13_slide4_aitex_7_weaves_gallery.png"
    if img4_p.exists():
        slide4.shapes.add_picture(str(img4_p), Inches(0.8), Inches(1.5), width=Inches(11.73))

    # Bottom: 4 Etiology Families Table
    headers4 = ["Structural Defect Family", "Physical Failure Mechanism", "Official AITEX Defect Codes Included", "Classical DIP Signature"]
    data4 = [
        ["Family 0: Warp Flaws", "Longitudinal yarn rupture / tension crease", "002 (Broken end), 006 (Broken yarn), 023 (Crease), 027 (Knots)", "High vertical elongation (theta ~ 90 deg, AR > 2.8)"],
        ["Family 1: Weft Flaws", "Transverse shuttle mispick / reed crack", "010 (Broken pick), 016 (Weft curl), 022 (Cut selvage), 036 (Crack)", "High horizontal elongation (theta ~ 0 deg, AR < 0.35)"],
        ["Family 2: Micro-Punctures", "Mechanical needle tear / fiber entanglement", "019 (Fuzzyball / Hole), 025 (Warp ball)", "Compact round void (AR ~ 1.1) + Texture Destroyed (TIR < 0.55)"],
        ["Family 3: Photometric Stains", "Liquid chemical / oil droplet absorption", "029 (Contamination), 030 (Nep / Oil Spot)", "Compact round shape (AR ~ 1.1) + Weave Intact (TIR >= 0.55)"],
    ]
    widths4 = [Inches(2.2), Inches(3.2), Inches(3.5), Inches(2.83)]
    add_table(slide4, Inches(0.8), Inches(4.0), Inches(11.73), Inches(2.5), headers4, data4, widths4)

    slide4.notes_slide.notes_text_frame.text = (
        "The AITEX database spans 7 industrial weaves from isotropic poplin to directional herringbone and striped twills. "
        "All 12 official defect codes from the database are accounted for and grouped into four distinct physical etiologies: "
        "Warp flaws, Weft flaws, Micro-punctures, and Photometric Stains."
    )

    # =========================================================================
    # SLIDE 5: FIVE-STAGE PIPELINE ARCHITECTURE
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    create_header(slide5, "End-to-End DIP Architecture: From Line-Scan to Multi-Class Triage", "SYSTEM METHODOLOGY")
    create_footer(slide5, 5)

    img5_p = ASSET_DIR / "02_slide5_pipeline_architecture.png"
    if img5_p.exists():
        slide5.shapes.add_picture(str(img5_p), Inches(0.8), Inches(1.5), width=Inches(11.73))

    # Phase descriptions below
    headers5 = ["Phase 0: Conditioning", "Phase 1: Spectral & Alignment", "Phase 2: Morphology", "Phase 3: Screening Gate", "Phase 4: Taxonomy"]
    data5 = [
        ["Active selvedge margin blanking (M_valid) + stationary sensor column blemish suppression.",
         "Radon de-skew ([-2.5°, +2.5°]) + 2D-FFT Butterworth notch weave harmonic suppression.",
         "Adaptive micro Top-Hat (1T, 0.5T) + Macro Gaussian envelope (W=201) + Twill linear kernels.",
         "Statistical calibrated thresholds (P99.99) + Uniform roll screening gate (tau* = 15 px).",
         "Hu moment invariants (phi1, phi2) + Laws' S5S5 Spot Energy (TIR) + Boundary Acutance (NBA)."]
    ]
    widths5 = [Inches(2.3), Inches(2.5), Inches(2.5), Inches(2.2), Inches(2.23)]
    add_table(slide5, Inches(0.8), Inches(4.5), Inches(11.73), Inches(2.0), headers5, data5, widths5)

    slide5.notes_slide.notes_text_frame.text = (
        "Our pipeline executes in five deterministic stages: sensor conditioning, frequency and Radon filtering, multi-scale "
        "mathematical morphology, 15-pixel screening triage, and post-detection Hu-Laws taxonomy classification."
    )

    # =========================================================================
    # SLIDE 6: SPECTRAL & DIRECTIONAL DESCRIPTORS (2D-FFT Notch & Radon)
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    create_header(slide6, "Spectral & Directional Descriptors: 2D-FFT Notch & Radon Transform", "PHYSICAL WEAVE SUPPRESSION")
    create_footer(slide6, 6)

    # Left: Fabric 03 2D-FFT Notch Showcase
    img6a_p = ASSET_DIR / "03_slide6_fabric03_fft_notch_filtering.png"
    if img6a_p.exists():
        slide6.shapes.add_picture(str(img6a_p), Inches(0.8), Inches(1.5), width=Inches(5.7))

    # Right: Fabric 05 Radon Transform Showcase
    img6b_p = ASSET_DIR / "04_slide6_fabric05_radon_rib_suppression.png"
    if img6b_p.exists():
        slide6.shapes.add_picture(str(img6b_p), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # Notes box below
    desc_box6 = slide6.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.73), Inches(2.2))
    tf6 = desc_box6.text_frame
    tf6.word_wrap = True
    p6_1 = tf6.paragraphs[0]
    p6_1.text = "1. 2D-FFT Butterworth Harmonic Notch Filter (Fabric 03 Herringbone Twill):"
    p6_1.font.bold = True
    p6_1.font.size = Pt(10.5)
    p6_1.font.color.rgb = COLOR_MAROON
    p6_2 = tf6.add_paragraph()
    p6_2.text = "• Identifies fundamental chevron frequency peaks in 2D spectrum and applies Gaussian-smooth notch rejection. Zeros out background periodic weave while leaving aperiodic defect energy unperturbed. Rescued Fabric 03 recall from 20.0% to 100.0% (+4 rolls caught)!"
    p6_2.font.size = Pt(9.5)
    p6_2.font.color.rgb = COLOR_DARK
    p6_2.space_before = Pt(2)

    p6_3 = tf6.add_paragraph()
    p6_3.text = "2. 1D Radon Transform Angular Energy Projection (Fabric 05 Striped Twill):"
    p6_3.font.bold = True
    p6_3.font.size = Pt(10.5)
    p6_3.font.color.rgb = COLOR_MAROON
    p6_3.space_before = Pt(6)
    p6_4 = tf6.add_paragraph()
    p6_4.text = "• Evaluates Radon projections across theta in [0, 180°]. Projects longitudinal rib channels along theta=90° and subtracts them from spatial domain. Completely eliminated rib-edge false alarms, achieving 100.0% accuracy, 100.0% recall, and 100.0% specificity!"
    p6_4.font.size = Pt(9.5)
    p6_4.font.color.rgb = COLOR_DARK
    p6_4.space_before = Pt(2)

    slide6.notes_slide.notes_text_frame.text = (
        "The hallmark of classical DIP is mathematical elegance. On Fabric 03—a complex herringbone twill—the 2D-FFT notch filter "
        "nullified chevron frequency peaks, lifting recall from 20% to 100%. On Fabric 05 Striped Twill, Radon projection along "
        "90 degrees extinguished heavy vertical stripes, achieving 100% accuracy and specificity."
    )

    # =========================================================================
    # SLIDE 7: SCREENING & LOCALIZATION RESULTS
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    create_header(slide7, "Operational Screening & Pixel Localization Results (All 7 Weaves)", "BENCHMARK RESULTS")
    create_footer(slide7, 7)

    # Left: Showcase Image
    img7_p = ASSET_DIR / "05_slide7_endsem_results_showcase.png"
    if img7_p.exists():
        slide7.shapes.add_picture(str(img7_p), Inches(0.8), Inches(1.5), width=Inches(5.6))

    # Right: Results Table
    headers7 = ["Weave Type", "Accuracy", "Recall", "Specificity", "Pixel Dice", "Triage (TP / FP / TN / FN)"]
    data7 = [
        ["Fabric 00: Poplin", "78.57%", "80.00%", "75.00%", "88.24%", "8 / 1 / 3 / 2"],
        ["Fabric 01: Twill 45°", "71.43%", "80.00%", "50.00%", "86.37%", "8 / 2 / 2 / 2"],
        ["Fabric 02: Satin", "85.71%", "80.00%", "100.0%", "77.96%", "8 / 0 / 4 / 2 (0 False Alarms)"],
        ["Fabric 03: Herringbone", "77.78%", "100.0%", "50.00%", "69.80%", "5 / 2 / 2 / 0 (100% Recall!)"],
        ["Fabric 04: Poplin II", "78.57%", "70.00%", "100.0%", "78.47%", "7 / 0 / 4 / 3 (0 False Alarms)"],
        ["Fabric 05: Striped Twill", "100.0%", "100.0%", "100.0%", "38.24%", "1 / 0 / 4 / 0 (Flawless 100%)"],
        ["Fabric 06: Drill Twill", "72.73%", "71.43%", "75.00%", "79.88%", "5 / 1 / 3 / 2"],
        ["OVERALL / TOTAL", "79.01%", "79.25%", "78.57%", "74.24%", "42 TP / 6 FP / 22 TN / 11 FN"],
    ]
    widths7 = [Inches(1.8), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(1.63)]
    add_table(slide7, Inches(6.7), Inches(1.5), Inches(5.83), Inches(4.9), headers7, data7, widths7)

    slide7.notes_slide.notes_text_frame.text = (
        "Across all 81 rolls of the AITEX benchmark, our pipeline achieves 79.01% screening accuracy, 79.25% recall, 78.57% specificity, "
        "and an industrial precision of 87.50%. Notice Fabric 05 achieved a flawless 100% score across all metrics, and Fabric 03 "
        "achieved 100% recall with zero false negatives."
    )

    # =========================================================================
    # SLIDE 8: MULTI-CLASS TAXONOMY (Laws' TEM & Stain vs Hole)
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    create_header(slide8, "Multi-Class Taxonomy: Resolving Stains vs. Holes via Laws' TEM", "TAXONOMY BREAKTHROUGH")
    create_footer(slide8, 8)

    # Top: Laws Stain vs Hole Figure
    img8a_p = ASSET_DIR / "07_slide8_laws_energy_stains_vs_holes.png"
    if img8a_p.exists():
        slide8.shapes.add_picture(str(img8a_p), Inches(0.8), Inches(1.5), width=Inches(5.7))

    # Right: Multi-Class Defect Gallery
    img8b_p = ASSET_DIR / "06_slide8_multiclass_defect_taxonomy.png"
    if img8b_p.exists():
        slide8.shapes.add_picture(str(img8b_p), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # Bottom: Taxonomy Table
    headers8 = ["Defect Family", "AITEX Codes Represented", "Support", "Precision", "Recall", "F1-Score", "Classical DIP Signature"]
    data8 = [
        ["Class 0: Warp Flaws", "002, 006, 023, 027", "13", "100.0%", "92.31%", "0.960", "Vertical elongation (AR > 2.8, theta ~ 90°)"],
        ["Class 1: Weft Flaws", "010, 016, 022, 036", "11", "90.91%", "90.91%", "0.909", "Horizontal elongation (AR < 0.35, theta ~ 0°)"],
        ["Class 2: Micro-Punctures", "019 (Hole), 025", "23", "95.83%", "100.0%", "0.979", "Compact round void (AR ~ 1.1) + TIR < 0.55 (Void)"],
        ["Class 3: Photometric Stains", "029, 030 (Oil Spot)", "6", "100.0%", "100.0%", "1.000", "Compact round (AR ~ 1.1) + TIR >= 0.55 (Intact Yarns)"],
        ["OVERALL TAXONOMY", "All 12 AITEX Codes", "53", "96.69%", "95.81%", "0.962", "51 / 53 Correct (96.23% Accuracy!)"],
    ]
    widths8 = [Inches(1.8), Inches(1.8), Inches(0.8), Inches(0.9), Inches(0.9), Inches(0.8), Inches(3.73)]
    add_table(slide8, Inches(0.8), Inches(4.3), Inches(11.73), Inches(2.4), headers8, data8, widths8)

    slide8.notes_slide.notes_text_frame.text = (
        "Slide 8 represents our proudest engineering breakthrough since mid-sem. Kenneth Laws' Texture Energy Measures (S5S5 Spot Energy) "
        "measures the Texture Intactness Ratio (TIR). Liquid oil stains preserve underlying yarn threads (TIR = 0.68), whereas mechanical "
        "punctures destroy them (TIR = 0.39). This surged stain recall from 50% to 100% and delivered 96.23% taxonomy accuracy."
    )

    # =========================================================================
    # SLIDE 9: DEDICATED OIL STAIN DIAGNOSTIC SHOWCASE
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    create_header(slide9, "Photometric Contaminant: Dedicated Oil Stain [Code 030] Analysis", "CASE STUDY")
    create_footer(slide9, 9)

    img9_p = ASSET_DIR / "08_slide8_oil_stain_nep_diagnosis.png"
    if img9_p.exists():
        slide9.shapes.add_picture(str(img9_p), Inches(0.8), Inches(1.5), width=Inches(11.73))

    desc_box9 = slide9.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.73), Inches(2.2))
    tf9 = desc_box9.text_frame
    tf9.word_wrap = True
    p9_1 = tf9.paragraphs[0]
    p9_1.text = "Engineering Diagnostic Breakdown (AITEX Sample 0093_030_01 on Fabric 01 Twill):"
    p9_1.font.bold = True
    p9_1.font.size = Pt(11)
    p9_1.font.color.rgb = COLOR_MAROON

    bullets9 = [
        ("• Subtle Radiometric Contrast:", "The oil stain induces only a Delta_I = 3.43 grayscale drop against background fibers."),
        ("• 4x Zoom Fiber Penumbra:", "Magnification confirms capillary dye absorption without broken yarn threads, creating a soft transition penumbra."),
        ("• Laws' S5S5 Spot Texture Intactness:", "TIR = 0.68 (comfortably exceeding the 0.55 rupture threshold), mathematically proving intact yarn structure."),
        ("• Verification Result:", "100.0% Precision and 100.0% Recall across all oil stain/nep samples (6/6 correctly intercepted and diagnosed)."),
    ]
    for b_title, b_desc in bullets9:
        p = tf9.add_paragraph()
        p.text = b_title + " "
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_DARK
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        p.space_before = Pt(3)

    slide9.notes_slide.notes_text_frame.text = (
        "Here we zoom in on an official AITEX oil spot (Code 030). The oil droplet causes a subtle 3.4 grayscale drop. Laws' spot "
        "energy proves the yarn grid is mechanically intact underneath the liquid stain, allowing our decision tree to achieve 100% "
        "precision and recall on chemical contaminants."
    )

    # =========================================================================
    # SLIDE 10: TEXTURE DESCRIPTORS SPECIALIZATION MATRIX
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    create_header(slide10, "Texture Descriptors Matrix: General Baseline vs. Fabric Specialization", "DESCRIPTOR ENGINEERING")
    create_footer(slide10, 10)

    # Top: Directional twill morphology image
    img10_p = ASSET_DIR / "09_slide9_directional_twill_morphology.png"
    if img10_p.exists():
        slide10.shapes.add_picture(str(img10_p), Inches(0.8), Inches(1.5), width=Inches(11.73))

    # Bottom: Specialization Table
    headers10 = ["Fabric ID", "Weave Pattern", "Classical DIP Descriptors Deployed", "Specialization Tier", "Mathematical Impact"]
    data10 = [
        ["Fabric 00", "Plain Poplin", "Multi-Scale Top-Hat + Hu Moments", "General Baseline", "Orthogonal grid yields clean morphological response (88.2% Dice)."],
        ["Fabric 01", "Twill (45°)", "Dual-Scale Morphology + 45° Linear SE", "Fabric-Specific", "Linear SE aligned at 45° suppresses diagonal rib false alarms (86.4% Dice)."],
        ["Fabric 02", "Fine Satin", "Selvedge Margin Blanking + Dynamic Otsu", "General Baseline", "Mirror-smooth filament lacks periodic texture; achieves 100% Specificity."],
        ["Fabric 03", "Herringbone", "2D-FFT Butterworth Harmonic Notch Filter", "Fabric-Specific", "Zeros periodic chevron harmonics; Recall surged from 20% to 100%!"],
        ["Fabric 04", "Poplin II", "Multi-Scale Morphological Opening", "General Baseline", "Dense orthogonal yarn packing handled cleanly; achieves 100% Specificity."],
        ["Fabric 05", "Striped Twill", "1D Radon Projection (theta = 90°)", "Fabric-Specific", "Projection subtraction yields 100% Accuracy, Recall, and Specificity."],
        ["Fabric 06", "Drill Twill (63°)", "Steeper Linear Structuring Element (63°)", "Fabric-Specific", "Linear SE aligned at 63° suppresses steep twill wales (79.9% Dice)."],
        ["All Fabrics", "All 12 Codes", "Laws' S5S5 Spot Energy + Boundary Acutance", "Universal Taxonomy", "Solves stain vs hole ambiguity across all fabrics; 96.23% Accuracy."],
    ]
    widths10 = [Inches(1.2), Inches(1.8), Inches(3.2), Inches(1.8), Inches(3.73)]
    add_table(slide10, Inches(0.8), Inches(4.3), Inches(11.73), Inches(2.4), headers10, data10, widths10)

    slide10.notes_slide.notes_text_frame.text = (
        "Our system uses a two-tier architecture: general morphological top-hats and Hu moments run universally, while specialized "
        "modules—such as 2D-FFT notch filtering for herringbone and Radon transforms for striped twills—are deployed when coherent "
        "weave structures require targeted suppression."
    )

    # =========================================================================
    # SLIDE 11: HYPERPARAMETER SENSITIVITY SWEEP (10, 15, 20, 25 px)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    create_header(slide11, "Hyperparameter Sensitivity: Area Threshold Sweep (10, 15, 20, 25 px)", "OPERATIONAL STABILITY")
    create_footer(slide11, 11)

    # Left: Sweep Figure
    img11_p = ASSET_DIR / "10_slide10_threshold_hyperparameter_sweep.png"
    if img11_p.exists():
        slide11.shapes.add_picture(str(img11_p), Inches(0.8), Inches(1.5), width=Inches(5.7))

    # Right: Table & Analysis
    headers11 = ["Threshold tau", "Accuracy", "Recall", "Specificity", "Precision", "Operating Characteristic"]
    data11 = [
        ["tau = 10 px", "79.01%", "81.13% (43/53)", "75.00% (21/28)", "86.00%", "High Sensitivity: Catches tiny specks; slightly higher false alarms."],
        ["tau = 15 px (tau*)", "79.01%", "79.25% (42/53)", "78.57% (22/28)", "87.50%", "Optimal Operating Point: Best balance of precision (87.5%) & Dice (74.2%)."],
        ["tau = 20 px", "75.31%", "73.58% (39/53)", "78.57% (22/28)", "86.67%", "Conservative: Prunes 3 minor broken ends; high precision."],
        ["tau = 25 px", "75.31%", "71.70% (38/53)", "82.14% (23/28)", "88.37%", "Ultra-Conservative: Peak specificity (82.1%), but misses subtle pick breaks."],
    ]
    widths11 = [Inches(1.4), Inches(0.9), Inches(1.1), Inches(1.1), Inches(0.9), Inches(1.43)]
    add_table(slide11, Inches(6.7), Inches(1.5), Inches(5.83), Inches(2.8), headers11, data11, widths11)

    # Callout text box
    callout11 = slide11.shapes.add_textbox(Inches(6.7), Inches(4.5), Inches(5.83), Inches(2.2))
    tf11 = callout11.text_frame
    tf11.word_wrap = True
    p11_1 = tf11.paragraphs[0]
    p11_1.text = "Key Takeaways from the Sensitivity Sweep:"
    p11_1.font.bold = True
    p11_1.font.size = Pt(11)
    p11_1.font.color.rgb = COLOR_MAROON

    b_text11 = (
        "• Stability Plateau (10-15 px): Accuracy remains steady at 79.01%, proving our morphological filters are robust and not brittle.\n"
        "• Why tau* = 15 px is Selected: Boosts Specificity from 75.0% to 78.57% (+1 clean roll rescued) while maintaining 87.50% precision and peak pooled Dice (74.24%).\n"
        "• Factory Trade-Off: Prevents stopping looms on transient dust fibers while catching 100% of severe structural fabric tears."
    )
    p11_2 = tf11.add_paragraph()
    p11_2.text = b_text11
    p11_2.font.size = Pt(9.5)
    p11_2.font.color.rgb = COLOR_DARK
    p11_2.space_before = Pt(4)

    slide11.notes_slide.notes_text_frame.text = (
        "In response to evaluator feedback, we swept screening thresholds across 10, 15, 20, and 25 pixels. The 10 to 15 pixel range "
        "forms a stable plateau at 79.01% accuracy. Moving to 15 pixels improves specificity to 78.57% and maximizes precision at 87.50%, "
        "making tau = 15 px the ideal industrial operating point."
    )

    # =========================================================================
    # SLIDE 12: LATENCY & HARDWARE PROFILING
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    create_header(slide12, "Computational Efficiency & Real-Time Edge Deployment", "DEPLOYMENT FEASIBILITY")
    create_footer(slide12, 12)

    # Left: Confusion matrix showcase
    img12_p = ASSET_DIR / "12_slide7_screening_confusion_matrices.png"
    if img12_p.exists():
        slide12.shapes.add_picture(str(img12_p), Inches(0.8), Inches(1.5), width=Inches(5.7))

    # Right: Latency breakdown table
    headers12 = ["Pipeline Stage", "Algorithm Deployed", "Latency", "Share", "Complexity"]
    data12 = [
        ["Phase 0: Conditioning", "Active border margin blanking", "1.4 ms", "6.4%", "O(MN)"],
        ["Phase 1: Frequency & Alignment", "Radon de-skew / 2D-FFT Notch", "7.8 ms", "35.5%", "O(MN log MN)"],
        ["Phase 2: Morphology", "Dual-Scale White/Black Top-Hats", "6.2 ms", "28.2%", "O(MNK)"],
        ["Phase 3: Screening Gate", "Dynamic Otsu & Connected Comps", "3.5 ms", "15.9%", "O(MN)"],
        ["Phase 4: Taxonomy", "Hu Moments & Laws' S5S5 Energy", "3.1 ms", "14.0%", "O(ROI)"],
        ["TOTAL LATENCY", "Deterministic End-to-End", "22.0 ms", "100%", "45.4 FPS (CPU)"],
    ]
    widths12 = [Inches(1.8), Inches(1.8), Inches(0.8), Inches(0.7), Inches(0.73)]
    add_table(slide12, Inches(6.7), Inches(1.5), Inches(5.83), Inches(2.9), headers12, data12, widths12)

    callout12 = slide12.shapes.add_textbox(Inches(6.7), Inches(4.6), Inches(5.83), Inches(2.1))
    tf12 = callout12.text_frame
    tf12.word_wrap = True
    p12_1 = tf12.paragraphs[0]
    p12_1.text = "Industrial Loom Compatibility:"
    p12_1.font.bold = True
    p12_1.font.size = Pt(11)
    p12_1.font.color.rgb = COLOR_MAROON

    p12_2 = tf12.add_paragraph()
    p12_2.text = (
        "• Modern industrial weaving looms operate at line speeds of 1.0 to 1.5 m/s.\n"
        "• At 22.0 ms execution per strip (256 x 4096 px), the fabric advances only 0.033 m during processing.\n"
        "• The pipeline operates >3x faster than maximum loom speeds, guaranteeing zero inspection lag.\n"
        "• Zero GPU dependency: Runs entirely on fanless edge industrial CPUs (<15 W power)."
    )
    p12_2.font.size = Pt(9.5)
    p12_2.font.color.rgb = COLOR_DARK
    p12_2.space_before = Pt(4)

    slide12.notes_slide.notes_text_frame.text = (
        "Deployment feasibility is paramount. Across an entire 256 by 4096 pixel strip, total execution latency is just 22 milliseconds "
        "on a commodity CPU. This achieves 45 frames per second, running over three times faster than maximum weaving loom speeds "
        "without requiring any expensive GPUs."
    )

    # =========================================================================
    # SLIDE 13: CONCLUSION, CONTRIBUTIONS & FUTURE ROADMAP
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    create_header(slide13, "Conclusion, Individual Contributions & Future Directions", "SUMMARY & WRAP-UP")
    create_footer(slide13, 13)

    # Left: Conclusion & Future Work
    cbox13 = slide13.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.0))
    tf13 = cbox13.text_frame
    tf13.word_wrap = True

    p13_1 = tf13.paragraphs[0]
    p13_1.text = "Summary of Technical Contributions:"
    p13_1.font.bold = True
    p13_1.font.size = Pt(11)
    p13_1.font.color.rgb = COLOR_MAROON

    b_c13 = [
        "1. Classical DIP Viability: Proved classical image processing delivers 87.50% precision and 74.24% pooled Dice on extreme line-scans without deep neural networks.",
        "2. Spectral & Directional Invariance: 2D-FFT Butterworth notch filtering rescued Fabric 03 recall to 100%, and Radon projection yielded 100% accuracy on Fabric 05.",
        "3. Solved Stain vs. Hole Dilemma: Integrated Kenneth Laws' Texture Energy (S5S5) and Boundary Acutance, surging stain recall from 50% to 100% and achieving 96.23% taxonomy accuracy.",
    ]
    for b in b_c13:
        p = tf13.add_paragraph()
        p.text = b
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(3)

    p13_2 = tf13.add_paragraph()
    p13_2.text = "Future Research Directions:"
    p13_2.font.bold = True
    p13_2.font.size = Pt(11)
    p13_2.font.color.rgb = COLOR_MAROON
    p13_2.space_before = Pt(8)

    b_f13 = [
        "• FPGA Hardware Synthesis: Implement 2D-FFT and morphological kernels in VHDL for sub-millisecond on-camera processing.",
        "• Photometric Stereo: Multi-directional lighting to reconstruct yarn surface normal vectors for depth-based abrasion detection.",
    ]
    for b in b_f13:
        p = tf13.add_paragraph()
        p.text = b
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(3)

    # Right: Team Contributions Table
    headers13 = ["Team Member", "Enrollment", "Core Technical Contribution"]
    data13 = [
        ["Pratham Sandesara", "AU2340239", "Threshold Sensitivity Sweep (10-25 px), Benchmark Evaluation, Master Pipeline Integration."],
        ["Even Patel", "AU2340241", "Adaptive Multi-Scale Morphology, Directional Top-Hat Kernels (45°, 63°), Dual-Scale Filters."],
        ["Dharman Patel", "AU2340237", "2D-FFT Butterworth Harmonic Notch Filter, Peak Detection, Fabric 03 Chevron Pattern Suppression."],
        ["Shine Shah", "AU2440148", "Radon Line-Integral Projections, Conveyor De-Skew, Kenneth Laws' Texture Energy (S5S5) & Acutance."],
    ]
    widths13 = [Inches(1.8), Inches(1.1), Inches(2.93)]
    add_table(slide13, Inches(6.7), Inches(1.5), Inches(5.83), Inches(3.2), headers13, data13, widths13)

    # Thank you box
    ty_box = slide13.shapes.add_textbox(Inches(6.7), Inches(4.9), Inches(5.83), Inches(1.7))
    tf_ty = ty_box.text_frame
    tf_ty.word_wrap = True
    pty1 = tf_ty.paragraphs[0]
    pty1.text = "THANK YOU!"
    pty1.font.bold = True
    pty1.font.size = Pt(18)
    pty1.font.color.rgb = COLOR_MAROON
    pty1.alignment = PP_ALIGN.CENTER

    pty2 = tf_ty.add_paragraph()
    pty2.text = "Questions & Evaluator Discussion"
    pty2.font.bold = True
    pty2.font.size = Pt(12)
    pty2.font.color.rgb = COLOR_DARK
    pty2.alignment = PP_ALIGN.CENTER
    pty2.space_before = Pt(4)

    slide13.notes_slide.notes_text_frame.text = (
        "In conclusion, our project validates that classical digital image processing remains an indispensable, highly effective tool "
        "for industrial computer vision. By combining Fourier frequency rejection, mathematical morphology, and Kenneth Laws' texture "
        "energy measures, we built a transparent, 22-millisecond inspection pipeline with 96.23% taxonomy accuracy. We thank our "
        "mentor Prof. Mehul Raval and the evaluators. We are now open for questions."
    )

    # Save presentation
    prs.save(str(OUTPUT_PPTX))
    print(f"\nPresentation successfully created: {OUTPUT_PPTX.resolve()}")


if __name__ == "__main__":
    build_presentation()

