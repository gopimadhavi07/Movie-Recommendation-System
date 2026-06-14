import pathlib

# Get the project root directory (absolute path)
# Since config.py is located at src/config.py, the root is its parent's parent.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Define main project directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
APP_DIR = PROJECT_ROOT / "app"

# Log file path
LOG_FILE_PATH = LOGS_DIR / "project.log"
