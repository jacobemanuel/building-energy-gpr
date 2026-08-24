from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_INCLUDE = [
    "src",
    "scripts",
    "tests",
    "outputs/figures",
    "outputs/predictions",
    "outputs/metrics",
    "README.md",
    "requirements.txt",
]

REPORT_PDF_NAME = "Final_Project_Report.pdf"
APPENDIX_PDF_NAME = "Final_Project_Report_Appendix.pdf"
SUBMISSION_ZIP_NAME = "Final_Project_JakubMajewski_PEDA_WS25.zip"


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__MACOSX" in parts:
        return True
    if "__pycache__" in parts:
        return True
    if path.name in {
        ".DS_Store",
        "touched_files.zip",
        "gpr_coverage.png",
        "correlation_heatmap.png",
        "target_joint_by_category.png",
        "build_report_assets.sh",
        "build_report_tables.py",
        "sync_report_markdown.py",
        "fetch_data.py",
        "repeated_splits.py",
        "train_gpytorch.py",
        "gpytorch_backend.py",
        "report_text.md",
        "report_text_working.md",
        "TU_Muenchen_Logo.svg.png",
    }:
        return True
    if path.suffix in {".aux", ".bbl", ".blg", ".log", ".out", ".pyc", ".pyo", ".tex"}:
        return True
    if "backends" in parts:
        return True
    if "report" in parts and path.suffix in {".tex", ".bib", ".md"}:
        return True
    return False


def add_path_to_zip(zipf: zipfile.ZipFile, root: Path, rel_path: Path) -> None:
    full_path = root / rel_path
    if full_path.is_dir():
        for item in full_path.rglob("*"):
            if item.is_file() and not should_skip(item):
                zipf.write(item, item.relative_to(root))
    elif full_path.is_file():
        if not should_skip(full_path):
            zipf.write(full_path, rel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean submission zip.")
    parser.add_argument("--output", type=Path, default=Path(SUBMISSION_ZIP_NAME))
    parser.add_argument("--exclude-data", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for entry in DEFAULT_INCLUDE:
            add_path_to_zip(zipf, project_root, Path(entry))

        if not args.exclude_data:
            data_path = project_root / "data" / "raw" / "ENB2012_data.xlsx"
            if data_path.exists():
                add_path_to_zip(zipf, project_root, data_path.relative_to(project_root))

        # add the compiled PDFs separately so the zip is usable on its own
        report_pdf = project_root / "report" / REPORT_PDF_NAME
        if not report_pdf.exists():
            report_pdf = project_root / "report" / "report.pdf"
        if report_pdf.exists():
            zipf.write(report_pdf, Path(REPORT_PDF_NAME))

        appendix_pdf = project_root / "report" / APPENDIX_PDF_NAME
        if not appendix_pdf.exists():
            appendix_pdf = project_root / "report" / "appendix.pdf"
        if appendix_pdf.exists():
            zipf.write(appendix_pdf, Path(APPENDIX_PDF_NAME))

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
