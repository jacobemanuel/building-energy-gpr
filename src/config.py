from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

RAW_XLSX = DATA_DIR / "ENB2012_data.xlsx"
RAW_CSV = DATA_DIR / "ENB2012_data.csv"

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]

CATEGORICAL_FEATURES = ["X6", "X8"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# known category levels in ENB2012
X6_LEVELS = [2, 3, 4, 5]        # orientation codes in the raw file
X8_LEVELS = [0, 1, 2, 3, 4, 5]  # glazing distribution codes in the raw file

DEFAULT_SEED = 42
TEST_SIZE = 0.2
