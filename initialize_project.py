import sys
import importlib
from src.config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    NOTEBOOKS_DIR, SRC_DIR, MODELS_DIR,
    OUTPUTS_DIR, LOGS_DIR, APP_DIR
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("initialize_project")

def create_directories():
    """
    Creates all required project directories if they don't exist.
    """
    dirs_to_create = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR, SRC_DIR, MODELS_DIR,
        OUTPUTS_DIR, LOGS_DIR, APP_DIR
    ]
    
    logger.info("Initializing project folder structure...")
    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory verified/created: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            sys.exit(1)
            
    logger.info("Folder structure initialized successfully!")

def verify_libraries():
    """
    Verifies that the required libraries are installed and logs their versions.
    """
    required_libraries = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("streamlit", "streamlit"),
        ("scipy", "scipy"),
        ("nltk", "nltk"),
        ("joblib", "joblib")
    ]
    
    logger.info("Verifying library installations and versions...")
    missing_libs = []
    
    for lib_import_name, user_friendly_name in required_libraries:
        try:
            module = importlib.import_module(lib_import_name)
            # Try to get the version
            version = getattr(module, "__version__", "unknown")
            logger.info(f" - {user_friendly_name}: Version {version} is installed.")
        except ImportError:
            logger.error(f" - {user_friendly_name} is NOT installed!")
            missing_libs.append(user_friendly_name)
            
    if missing_libs:
        logger.error(f"Verification FAILED. Missing libraries: {', '.join(missing_libs)}")
        return False
        
    logger.info("All library installations verified successfully!")
    return True

if __name__ == "__main__":
    logger.info("=== Starting Project Initialization ===")
    create_directories()
    success = verify_libraries()
    if not success:
        logger.warning("Please install dependencies using: pip install -r requirements.txt")
        sys.exit(1)
    logger.info("=== Project Initialization Completed Successfully ===")
