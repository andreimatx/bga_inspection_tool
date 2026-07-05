# BGA Void Inspection - Research Notes

These notes summarize the local PDF references for the planned classical image
processing pipeline. The project intentionally avoids machine learning unless
explicitly requested.

## Practical Direction

- Use 2D X-ray images for non-destructive inspection of BGA solder joints.
- Detect solder balls first, then evaluate voids only inside each ball ROI.
- Treat voids as brighter regions inside darker solder balls in the PCBA1/PCBA2
  images.
- Keep `Images_BGA/Raw` read-only and save all generated artifacts in
  `Processed`, `ROI`, `Masks`, and `Results`.
- Main methodology title/direction recommended by the professor:
  "A methodology for estimating probable soldering process irregularities based
  on void morphology extracted from X-ray images of BGA solder joints."

## Recommended Core Methodology

Focus the paper and implementation on six explainable algorithms:

| Step | Algorithm |
| --- | --- |
| Enhancement | CLAHE |
| Noise reduction | Median Filter |
| BGA detection | Hough Circle Transform |
| Void segmentation | Adaptive Thresholding |
| Void refinement | Morphological Opening/Closing |
| Statistical evaluation | Connected Component Analysis + Void Percentage |

Other methods can be mentioned as alternatives in the literature review, but the
implementation should prioritize this six-step chain first.

## Most Relevant Methods From The Papers

- Histogram analysis and contrast enhancement are useful before segmentation.
- CLAHE is appropriate for local contrast improvement on uneven X-ray images.
- Noise reduction should be conservative: median filtering or Gaussian blur.
- Ball detection candidates:
  - Hough circle transform.
  - Contour analysis.
  - Connected components after thresholding.
- Void segmentation candidates:
  - Global thresholding.
  - Adaptive thresholding.
  - Otsu thresholding.
  - Blob detection / connected component analysis.
  - Canny edge detection as a comparison method.
- Morphological opening and closing should be used to remove noise and stabilize
  detected void regions.
- Component filtering should use area, circularity/shape factor, and gray level.

## Useful Paper-Specific Takeaways

- `Image Processing Methods for Detecting Voids in the Solder Joints of
  Surface-Mounted Components` compares Canny, global thresholding, adaptive
  thresholding, and blob detection in OpenCV-Python. It supports trying several
  simple explainable methods before choosing the most stable one.
- `An algorithm based on K-means for calculating void ratio of solder joint from
  X-ray image` describes a four-stage workflow: preprocessing, morphological
  processing, void extraction, and void ratio calculation. It uses gray value,
  eccentricity, and area as useful void features.
- `Adaptive Hybrid Framework for Multiscale Void Inspection...` reinforces
  adaptive processing, Otsu thresholding, guided filtering, shape factor, and
  average-gray filtering to reduce false detections.
- `Quantitative statistically robust characterization of solder joint geometry
  and voids in commercial BGA assemblies` supports consistent threshold-based
  workflows and exporting quantitative metrics for later statistical analysis.
- `Effect of Solder Paste Alloy and Volume on Solder Voiding` and related
  reliability papers frame void ratio as an important quality metric. Reported
  acceptability discussions often mention approximate limits around 25 percent
  void area, but project thresholds should remain configurable.
- The Phoenix V|tome|x S240 document is equipment context for industrial 2D
  X-ray / CT inspection rather than algorithm design.

## Metrics To Report

Per solder ball:

- Ball ID.
- Center X/Y.
- Diameter.
- Ball area.
- Number of voids.
- Total void area.
- Largest void area.
- Void ratio percent.

Whole image / BGA:

- Total number of detected balls.
- Average void ratio.
- Maximum void ratio.
- Statistical summary.
- Annotated inspection image.
- CSV report.
- Excel report.

## Initial Dataset Strategy

- Start calibration with `Images_BGA/Raw/PCBA2/PCBA2_05.jpg` and
  `Images_BGA/Raw/PCBA2/PCBA2_04.jpg` because voids are visually clear.
- Validate on `Images_BGA/Raw/PCBA1/PCBA1_08.jpg` and
  `Images_BGA/Raw/PCBA1/PCBA1_10.jpg` because voids are subtler.
- Use `PCBA2_03.jpg`, `PCBA2_03_.jpg`, `PCBA2_06.jpg`, and `PCBA1_07.jpg` for
  robustness tests on contrast, zoom level, and wider fields of view.

## Implementation Constraints

- Use Python 3.14 with OpenCV, NumPy, Pandas, Matplotlib, Scikit-Image, SciPy,
  Pillow, OpenPyXL, and XlsxWriter.
- Do not introduce TensorFlow, PyTorch, YOLO, CNNs, U-Net, Detectron2, or other
  machine learning frameworks unless explicitly requested.
- Keep functions small, explainable, and reusable.
- Use English comments only.
- Follow PEP8.
