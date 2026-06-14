import pandas as pd
import pathlib
from src.utils.logger import get_logger

logger = get_logger("movie_rec.helpers")

def read_csv(file_path):
    """
    Safely reads a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str or pathlib.Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        Exception: If any error occurs during reading.
    """
    path = pathlib.Path(file_path)
    logger.info(f"Attempting to read CSV file from: {path.name}")
    logger.debug(f"Full CSV file path: {path}")
    
    if not path.exists():
        logger.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"CSV file not found: {path}")
        
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded CSV with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file at {path}: {e}", exc_info=True)
        raise e

def save_dataframe(df, file_path):
    """
    Safely saves a pandas DataFrame to a CSV file, ensuring parent folders exist.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str or pathlib.Path): Destination path.
        
    Returns:
        bool: True if save succeeded.
        
    Raises:
        Exception: If any error occurs during saving.
    """
    path = pathlib.Path(file_path)
    logger.info(f"Attempting to save DataFrame to: {path.name}")
    logger.debug(f"Full destination path: {path}")
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved DataFrame to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to {path}: {e}", exc_info=True)
        raise e

def display_df_info(df, name="Dataset"):
    """
    Logs and returns key metadata info about a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        name (str): Label/Name for the dataset.
    """
    logger.info(f"--- DataFrame Information: {name} ---")
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 * 1024)
    logger.info(f"Memory Usage: {mem_mb:.2f} MB")
    logger.debug(f"Data types:\n{df.dtypes}")

def check_missing_values(df):
    """
    Computes missing count and percentage per column for a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        
    Returns:
        pd.DataFrame: Summary table of columns containing missing values.
    """
    logger.info("Checking for missing values...")
    total_missing = df.isnull().sum()
    percent_missing = (df.isnull().sum() / len(df)) * 100
    
    missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Missing Count', 'Percentage (%)'])
    
    # Filter out columns with 0 missing values
    missing_data_filtered = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    if len(missing_data_filtered) == 0:
        logger.info("No missing values found in the dataset.")
    else:
        logger.info(f"Found missing values in {len(missing_data_filtered)} columns.")
        logger.debug(f"Missing values summary:\n{missing_data_filtered}")
        
    return missing_data
