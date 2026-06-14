import urllib.request
import zipfile
import io
import shutil
import pathlib
from src.utils.logger import get_logger
from src.config import RAW_DATA_DIR

logger = get_logger("movie_rec.downloader")

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens_dataset():
    """
    Downloads the MovieLens small dataset and extracts the required CSV files to data/raw/
    
    Returns:
        bool: True if files exist or download was successful, False otherwise.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    target_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    
    # Check if files already exist
    all_exist = all((RAW_DATA_DIR / f).exists() for f in target_files)
    if all_exist:
        logger.info("Raw MovieLens dataset files already exist in data/raw. Skipping download.")
        return True
        
    logger.info(f"Downloading MovieLens dataset from {MOVIELENS_URL}...")
    try:
        req = urllib.request.Request(
            MOVIELENS_URL, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            zip_data = response.read()
            logger.info("Download completed successfully. Extracting files...")
            
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                extracted_count = 0
                for file_info in zip_ref.infolist():
                    name = file_info.filename
                    base_name = pathlib.Path(name).name
                    if base_name in target_files:
                        target_path = RAW_DATA_DIR / base_name
                        # Extract directly into raw data dir
                        with zip_ref.open(name) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        logger.info(f"Extracted: {base_name} -> {target_path}")
                        extracted_count += 1
                
                if extracted_count == len(target_files):
                    logger.info("Successfully downloaded and extracted all raw datasets!")
                    return True
                else:
                    logger.warning(f"Only extracted {extracted_count}/{len(target_files)} target files.")
                    return False
    except Exception as e:
        logger.error(f"Failed to download or extract MovieLens dataset: {e}", exc_info=True)
        raise e
