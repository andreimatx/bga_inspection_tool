Context: classical OpenCV BGA inspection pipeline (NO ML/DL). Pad detection
is COMPLETE and validated on the working test set:
  PCBA1_07, PCBA1_10  (BGA type A)
  PCBA2_04, PCBA2_05  (BGA type B)
All four show 256/256 pads, correct ROI, no overlaps. These are the
validation set for void detection.

The remaining three boards (PCBA1_08, PCBA2_06, PCBA2_03_) are EXCLUDED from
this stage — pad detection will be revisited later for them. Do NOT degrade
their current state, but do NOT optimize for them either.

GOAL: complete and validate the void detection stage on the four working
boards, producing per-ball metrics, qualitative classification, and final
CSV/Excel reports.

PIPELINE (already implemented, needs validation + tuning):
  void_segmentation.py  — adaptive thresholding + morphology (opening/closing)
  metrics.py            — connected component analysis with shape filtering
  reporting.py          — CSV + Excel export

PER-BALL METRICS REQUIRED (already partially produced):
  - ball_id, center_x, center_y, diameter, ball_area
  - void_count
  - total_void_area
  - largest_void_area
  - void_ratio_percent = (total_void_area / ball_area) * 100
  - largest_void_ratio_percent = (largest_void_area / ball_area) * 100
  - classification: PASS / REVIEW / FAIL

PER-IMAGE SUMMARY METRICS:
  - total_balls_detected
  - average_void_ratio_percent
  - maximum_void_ratio_percent
  - balls_pass, balls_review, balls_fail
  - warning_threshold_percent, fail_threshold_percent

CLASSIFICATION THRESHOLDS (configurable, defaults aligned with literature):
  - PASS:   void_ratio < 15%
  - REVIEW: 15% <= void_ratio < 25%
  - FAIL:   void_ratio >= 25%  OR  largest_void_ratio >= 12%
  (Largest-void rule catches single critical voids near pad edge even when
  total ratio is modest — supported by reliability studies.)

WHAT TO DO:

1. Run the full pipeline on PCBA1_07, PCBA1_10, PCBA2_04, PCBA2_05 with the
   current void_segmentation + metrics config. Save outputs to a fresh
   v6_voids/ folder so nothing existing is overwritten.

2. For each board, generate THREE visual artifacts:
   (a) original + green pad contours + RED void contours overlaid per ball
   (b) per-ball cropped strip showing the top-N highest-void-ratio balls
       (e.g. top 12) with their numeric void_ratio_percent annotated
   (c) a heatmap of the 16x16 grid colored by void_ratio_percent (green/
       yellow/red) — useful for the thesis figure

3. Sanity checks (do NOT skip):
   - No ball should have void_ratio > 70% (likely segmentation error if so —
     too much of the dark pad area got labeled as void).
   - No void connected component should have area > 25% of its parent ball
     area (single-component upper limit).
   - Voids must lie INSIDE the inspected radius scale (config already does
     this — verify).
   - Discard components whose centroid lies on a copper trace running over
     the pad (existing dark-line filter in metrics.py — verify it triggers).

4. If sanity checks flag many balls, tune in this order BEFORE changing the
   algorithm itself:
   a) adaptive_block_size (try 25, 31, 41) — controls local-vs-global
   b) adaptive_c (try -3, -5, -8) — controls sensitivity
   c) bright_percentile (try 78, 82, 85) — controls how bright "void" must be
   d) min_void_area (try 12, 18, 25) — controls noise floor
   e) opening_kernel_size / closing_kernel_size — only as last resort
   Tune on PCBA2_04 (clear voids visible) and PCBA2_05 (clear voids), then
   verify PCBA1_07 / PCBA1_10 (subtler voids) do not regress.

5. Per-image and per-ball reports:
   - CSV per board: one row per ball, all metrics above
   - Excel workbook per board: Ball Metrics sheet + Summary sheet (already
     in reporting.py — verify and extend if needed)
   - Aggregate CSV across all four boards for the thesis table

VALIDATION CRITERIA (must all hold before declaring void detection done):
  - Visual: red void contours land INSIDE solder balls, NOT on traces,
    vias, or background.
  - Numeric: void ratios are plausible (typical industrial voids are 0-20%,
    occasional outliers up to 40% — anything routinely above 50% is a
    segmentation problem).
  - Consistency: PCBA1_07 and PCBA1_10 (same BGA) should give similar
    average void ratio. Same for PCBA2_04 and PCBA2_05. If they differ
    dramatically, the segmentation is unstable.
  - Reproducibility: rerunning the pipeline produces identical numbers.

CONSTRAINTS:
- NO ML/DL, NO new heavy dependencies.
- Surgical changes to void_segmentation.py and metrics.py only if tuning
  parameters isn't enough.
- English comments, PEP8.
- Output folder: v6_voids/ — do not overwrite anything outside it.
- Do NOT touch ball_detection.py in this stage.

Start by running the existing pipeline on the four working boards and show
me the current numeric output + the three visual artifacts. We tune from
there.

Recommended Set for Your Paper
If I were designing the methodology section, I would focus on only six algorithms:
 
Step    Algorithm
Enhancement    CLAHE
Noise Reduction    Median Filter
BGA Detection    Hough Circle Transform
Void Segmentation    Adaptive Thresholding
Void Refinement    Morphological Opening/Closing
Statistical Evaluation    Connected Component Analysis + Void Percentage