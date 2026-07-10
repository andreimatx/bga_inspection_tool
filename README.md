# BGA X-ray Void Inspection Tool

Automated void inspection for BGA (Ball Grid Array) solder joints from 2D
X-ray images. Detects all 256 balls of a 16×16 BGA — including on oblique,
perspective-distorted views — segments the gas voids inside each ball,
measures them, and classifies every ball and board against **IPC-A-610 /
IPC-7095** acceptance criteria. Each run produces presentation-grade Excel
and Word reports plus machine-readable CSVs.

## How it works

1. **Preprocessing** — CLAHE contrast enhancement + median denoising.
2. **Ball detection** — Hough circles + dark-blob evidence, regularized onto
   the 16×16 BGA grid. Oblique views (pitch anisotropy ≥ 1.12) get an
   affine + homography grid fit and a per-ball local snap with
   non-degradation gates; straight views get a rescue-only snap that fixes
   clearly misplaced circles without touching good ones.
3. **Void segmentation** — per ball: median-background subtraction (kernel
   larger than the ball), bond-wire suppression via grayscale closing,
   residual thresholding, then shape filtering. Detected bubbles are
   rendered as fitted circles (physical voids are spherical) when they fill
   ≥ 45% of the fitted disc.
4. **Metrics & classification** — projected void area per ball vs the IPC
   limits (below), plus component-level filters against rim arcs, trace
   stripes and bare-board bleed-through.
5. **Reporting** — Excel (4 sheets, charts, conditional formatting), Word
   (with embedded annotated X-ray), CSVs, and diagnostic overlay images.

## Acceptance criteria (see `src/standards.py`)

| Rule | Source | Result |
|---|---|---|
| Cumulative void area per ball > **25%** | IPC-A-610 (Class 1/2/3) | **FAIL / DEFECT** |
| Total void in the **10–25%** band | IPC-7095 process guidance | REVIEW / process indicator |
| Single void ≥ **12.25%** area (~35% of ball diameter) | IPC-7095 dominant-void guidance | REVIEW / process indicator |
| Voids at the pad interface | IPC-7095 | NOT EVALUATED — impossible from one top-down 2D view; stated, not guessed |

Board verdict: **REJECT** if any ball fails, **ACCEPT WITH PROCESS
INDICATORS** if any ball is flagged, otherwise **ACCEPT**.

## Installation

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
```

## Usage

```bash
# One board (searched under Images_BGA/Raw/<family>/)
python src/main.py PCBA2_05

# Several boards — runs in parallel, one worker per image (default)
python src/main.py PCBA1_07,PCBA1_08,PCBA2_05

# A folder, a single file, sequential mode, custom output root
python src/main.py --input-dir Images_BGA/Raw/PCBA2
python src/main.py --image path/to/board.jpg
python src/main.py PCBA2_05 --jobs 1
python src/main.py PCBA2_05 --output-root results/

# Also dump the 768 per-ball ROI/mask JPGs (~190 MB, visual debugging only)
python src/main.py PCBA2_05 --save-ball-crops

# Custom thresholds (defaults are the IPC values: 10 / 25 / 12.25)
python src/main.py PCBA2_05 --warning-threshold 8 --fail-threshold 20
```

Every run prints per-stage timings, e.g.:

```
[TIMING] Load: 0.2s | Preprocess: 0.3s | Ball detection: 26.7s | Void segmentation: 3.3s | Annotations: 0.4s | Reports: 3.2s
[TIMING] TOTAL: 34.1s
```

## Outputs (per board, under the output root)

| Path | Content |
|---|---|
| `Processed/<board>_01..06_*.jpg` | Diagnostic overlays: CLAHE, denoised, ROI debug, pad contours, annotated metrics, void preview |
| `Reports/<board>_inspection_report.xlsx` | 4 sheets: Executive Summary, IPC Evaluation, Ball Details, Algorithm Analysis (honest assessment) |
| `Reports/<board>_inspection_report.docx` | Printable report with charts and the annotated X-ray |
| `Reports/<board>_ball_metrics.csv` | Per-ball table (positions, areas, void %, IPC assessment) |
| `Reports/<board>_summary.csv` | One-line board summary with the IPC verdict |
| `ROI/`, `Masks/` | Per-ball crops and void masks (only with `--save-ball-crops`) |

Report templates generated from the same code live in `templates/`
(regenerate with `python tools/generate_report_templates.py`).

## Quality gates

```bash
# Unit tests: IPC threshold boundaries, verdict aggregation, merge exactness
python -m pytest tests/ -q            # ~5 s, no images needed

# Full image regression: re-analyzes every reference X-ray and compares
# per-ball metrics against tools/regression_baseline.json
python tools/regression_check.py            # compare
python tools/regression_check.py --update   # rewrite baseline (deliberate!)
```

The regression baseline is the safety net for algorithm changes: all
performance work (memoized evidence scoring, vectorized radius estimation,
spatial-hash candidate merging, parallel batch) was verified bit-identical
against it.

## Performance

Measured on a 3008×2512 px board, 8-core machine:

| Scenario | Time |
|---|---|
| Single board, detection stage | ~27 s |
| Single board, full pipeline | ~35 s |
| Batch of 8 boards (parallel, `--jobs` auto) | ~2.5 min wall |

## Project structure

```
src/
  main.py               CLI, pipeline orchestration, parallel batch
  image_loader.py       grayscale loading, format handling
  preprocessing.py      CLAHE + median denoising
  ball_detection.py     ball/grid detection (Hough, grid fit, homography, snap)
  void_segmentation.py  per-ball void segmentation
  metrics.py            void component filtering + measurements
  standards.py          IPC-A-610 / IPC-7095 criteria and classification
  reporting.py          CSV / Excel / Word report generation
  visualization.py      overlay and preview rendering
tools/
  regression_check.py            image-level regression harness
  generate_report_templates.py   report templates into templates/
tests/                  fast unit tests (no images required)
Images_BGA/Raw/         input X-ray images, one folder per PCBA family
```

## Known limitations

- One top-down 2D X-ray cannot separate package-interface, mid-ball and
  pad-interface voids — the tool measures the projected cumulative area,
  exactly as IPC-A-610 defines the X-ray criterion, and says so in the
  report instead of guessing.
- Components on the opposite board side cast shadows that reduce void
  sensitivity on affected balls; oblique views additionally distort areas.
- Voids smaller than ~10 px² (≈0.15% of a ball) are below the detection
  floor by design.
