import sys
import os
import pathlib
import pandas as pd

# Add the project directory to sys.path so src is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, LOG_FILE_PATH, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import read_csv, save_dataframe, display_df_info, check_missing_values

logger = get_logger("test_setup")

def test_directories():
    """
    Verifies that the required folders exist.
    """
    logger.info("Testing directory paths configuration...")
    required_dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]
    for d in required_dirs:
        assert d.exists(), f"Directory {d} does not exist!"
        logger.info(f"Directory verified: {d}")
    logger.info("[SUCCESS] All directories verified successfully.")

def test_logging():
    """
    Verifies that logging works and writes to the log file.
    """
    logger.info("Testing file logger mechanism...")
    test_message = "TEST_LOG_MESSAGE_XYZ_123"
    logger.debug(test_message)
    
    assert LOG_FILE_PATH.exists(), f"Log file at {LOG_FILE_PATH} does not exist!"
    
    # Read the log file and search for the test message
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        logs = f.read()
    
    assert test_message in logs, "Test log message was not written to file!"
    logger.info("[SUCCESS] Logging verification passed. Messages written successfully to file.")

def test_helpers():
    """
    Verifies the helper functions: read_csv, save_dataframe, check_missing_values, display_df_info
    """
    logger.info("Testing pandas/dataframe utilities...")
    
    # Create test dataframe
    data = {
        "id": [1, 2, 3],
        "name": ["A", "B", None],
        "score": [8.5, None, 9.2]
    }
    df = pd.DataFrame(data)
    
    # Test display_df_info
    display_df_info(df, name="Test DF")
    
    # Test missing values check
    missing = check_missing_values(df)
    assert missing.loc["name", "Missing Count"] == 1, "Missing counts incorrect for 'name' column!"
    assert missing.loc["score", "Missing Count"] == 1, "Missing counts incorrect for 'score' column!"
    logger.info("[SUCCESS] check_missing_values helper verification passed.")
    
    # Test saving dataframe
    temp_csv = DATA_DIR / "processed" / "temp_test.csv"
    save_dataframe(df, temp_csv)
    assert temp_csv.exists(), "Saved CSV file does not exist!"
    logger.info("[SUCCESS] save_dataframe helper verification passed.")
    
    # Test reading dataframe
    df_loaded = read_csv(temp_csv)
    assert df_loaded.shape == (3, 3), "Loaded CSV shape is incorrect!"
    # Ensure loaded data holds the null values properly (pandas reads float/NaN for mixed types/numeric nulls)
    assert pd.isnull(df_loaded.loc[2, "name"]), "Null value not loaded correctly for 'name'!"
    logger.info("[SUCCESS] read_csv helper verification passed.")
    
    # Clean up temp test file
    try:
        temp_csv.unlink()
        logger.info("[SUCCESS] Temp test files cleaned up successfully.")
    except Exception as e:
        logger.warning(f"Failed to delete temp test file: {e}")

if __name__ == "__main__":
    logger.info("=== Running Verification Tests ===")
    try:
        test_directories()
        test_logging()
        test_helpers()
        logger.info("=== All Setup Verification Tests Passed! ===")
    except AssertionError as ae:
        logger.error(f"Test failure: {ae}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected verification failure: {e}", exc_info=True)
        sys.exit(1)
