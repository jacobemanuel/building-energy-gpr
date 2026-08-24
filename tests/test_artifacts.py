import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_submission import DEFAULT_INCLUDE, add_path_to_zip

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_evaluate_does_not_emit_design_insight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--output-dir",
                    str(output_dir),
                    "--targets",
                    "Y1",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.evaluate",
                    "--output-dir",
                    str(output_dir),
                    "--targets",
                    "Y1",
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )

            self.assertTrue((output_dir / "metrics" / "results_table.csv").exists())
            self.assertFalse((output_dir / "metrics" / "design_insight.json").exists())

    def test_submission_archive_excludes_caches_and_keeps_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir(parents=True)
            (root / "src" / "__pycache__").mkdir(parents=True)
            (root / "scripts").mkdir(parents=True)
            (root / "tests").mkdir(parents=True)
            (root / "outputs" / "figures").mkdir(parents=True)
            (root / "outputs" / "predictions").mkdir(parents=True)
            (root / "outputs" / "tables").mkdir(parents=True)
            (root / "outputs" / "metrics").mkdir(parents=True)
            (root / "data" / "raw").mkdir(parents=True)
            (root / "report").mkdir(parents=True)

            (root / "src" / "module.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "src" / "__pycache__" / "module.cpython-313.pyc").write_bytes(b"junk")
            (root / "scripts" / "helper.py").write_text("print('helper')\n", encoding="utf-8")
            (root / "tests" / "test_dummy.py").write_text("def test_dummy():\n    assert True\n", encoding="utf-8")
            (root / "outputs" / "figures" / "figure.png").write_bytes(b"png")
            (root / "outputs" / "predictions" / "example.csv").write_text("y_true,y_pred\n1,1\n", encoding="utf-8")
            (root / "outputs" / "tables" / "table.tex").write_text("table\n", encoding="utf-8")
            (root / "outputs" / "metrics" / "metrics.json").write_text("{}\n", encoding="utf-8")
            (root / "data" / "raw" / "ENB2012_data.xlsx").write_bytes(b"xlsx")
            (root / "report" / "report.pdf").write_bytes(b"pdf")
            (root / "report" / "appendix.pdf").write_bytes(b"pdf")
            (root / "README.md").write_text("# demo\n", encoding="utf-8")
            (root / "requirements.txt").write_text("pytest>=8.0.0\n", encoding="utf-8")

            archive_path = root / "submission.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for entry in DEFAULT_INCLUDE:
                    add_path_to_zip(zipf, root, Path(entry))
                add_path_to_zip(zipf, root, Path("data/raw/ENB2012_data.xlsx"))
                zipf.write(root / "report" / "report.pdf", Path("report.pdf"))
                zipf.write(root / "report" / "appendix.pdf", Path("appendix.pdf"))

            with zipfile.ZipFile(archive_path) as zipf:
                names = set(zipf.namelist())

            self.assertIn("outputs/predictions/example.csv", names)
            self.assertIn("data/raw/ENB2012_data.xlsx", names)
            self.assertIn("report.pdf", names)
            self.assertIn("appendix.pdf", names)
            self.assertFalse(any("__pycache__" in name for name in names))
            self.assertFalse(any(name.endswith(".pyc") for name in names))


if __name__ == "__main__":
    unittest.main()
