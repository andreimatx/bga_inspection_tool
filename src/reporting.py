"""CSV and Excel reporting for BGA void inspection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_summary(
    image_name: str,
    rows: list[dict[str, object]],
    warning_threshold: float,
) -> dict[str, object]:
    """Build one image-level statistical summary."""
    if not rows:
        return {
            "image_name": image_name,
            "total_balls": 0,
            "average_void_ratio_percent": 0.0,
            "maximum_void_ratio_percent": 0.0,
            "balls_over_warning_threshold": 0,
            "warning_threshold_percent": warning_threshold,
        }

    ratios = [float(row["void_ratio_percent"]) for row in rows]
    return {
        "image_name": image_name,
        "total_balls": len(rows),
        "average_void_ratio_percent": round(sum(ratios) / len(ratios), 4),
        "maximum_void_ratio_percent": round(max(ratios), 4),
        "balls_over_warning_threshold": sum(
            ratio >= warning_threshold for ratio in ratios
        ),
        "warning_threshold_percent": warning_threshold,
    }


def save_reports(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output_dir: Path,
    image_stem: str,
) -> dict[str, Path]:
    """Save detailed and summary reports as CSV and Excel."""
    details_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame([summary])

    details_csv = output_dir / f"{image_stem}_ball_metrics.csv"
    summary_csv = output_dir / f"{image_stem}_summary.csv"
    excel_path = output_dir / f"{image_stem}_inspection_report.xlsx"

    details_df.to_csv(details_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        details_df.to_excel(writer, sheet_name="Ball Metrics", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _autosize_sheet(writer, details_df, "Ball Metrics")
        _autosize_sheet(writer, summary_df, "Summary")

    return {
        "details_csv": details_csv,
        "summary_csv": summary_csv,
        "excel": excel_path,
    }


def _autosize_sheet(
    writer: pd.ExcelWriter,
    dataframe: pd.DataFrame,
    sheet_name: str,
) -> None:
    """Set simple column widths for a readable Excel report."""
    worksheet = writer.sheets[sheet_name]
    for column_index, column_name in enumerate(dataframe.columns):
        values = dataframe[column_name].astype(str).tolist()
        width = max([len(column_name), *[len(value) for value in values]]) + 2
        worksheet.set_column(column_index, column_index, min(width, 32))
