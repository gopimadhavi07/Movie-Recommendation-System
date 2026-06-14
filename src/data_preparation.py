import pandas as pd
import numpy as np
import re
import pathlib
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.downloader import download_movielens_dataset
from src.utils.helpers import read_csv, save_dataframe, display_df_info, check_missing_values

logger = get_logger("movie_rec.data_prep")

def ensure_raw_data():
    """
    Downloads and extracts raw data files if not already present.
    """
    logger.info("Verifying raw MovieLens datasets availability...")
    success = download_movielens_dataset()
    if not success:
        raise RuntimeError("Failed to ensure raw data availability.")

def load_raw_datasets():
    """
    Loads raw CSV files into pandas DataFrames.
    
    Returns:
        tuple: (movies_df, ratings_df, tags_df, links_df)
    """
    ensure_raw_data()
    
    movies_df = read_csv(RAW_DATA_DIR / "movies.csv")
    ratings_df = read_csv(RAW_DATA_DIR / "ratings.csv")
    tags_df = read_csv(RAW_DATA_DIR / "tags.csv")
    links_df = read_csv(RAW_DATA_DIR / "links.csv")
    
    return movies_df, ratings_df, tags_df, links_df

def clean_movies(movies_df):
    """
    Cleans movies dataset:
    - Removes duplicates.
    - Extracts release year from title.
    - Standardizes title formatting.
    - Parses genres.
    
    Args:
        movies_df (pd.DataFrame): Raw movies DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned movies DataFrame.
    """
    logger.info("Cleaning movies dataset...")
    df = movies_df.copy()
    
    # Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(subset=["movieId"], inplace=True)
    if df.shape != initial_shape:
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate movieId entries.")
        
    # Extract release year using regex, e.g., "Toy Story (1995)"
    # Pattern looks for 4 digits in parentheses at the end of the title string
    year_pattern = r"\s*\((\d{4})\)\s*$"
    
    def extract_year(title):
        match = re.search(year_pattern, str(title))
        if match:
            return int(match.group(1))
        return np.nan
        
    df["release_year"] = df["title"].apply(extract_year)
    
    # Remove the year from the title string and clean surrounding whitespace
    df["title"] = df["title"].apply(lambda t: re.sub(year_pattern, "", str(t)).strip())
    
    # Handle missing/unknown release years (e.g. set to None or float NaN)
    missing_years = df["release_year"].isnull().sum()
    if missing_years > 0:
        logger.info(f"Found {missing_years} movies with missing/unparseable release year.")
        
    # Clean genres: convert to list of genres
    # e.g., "Action|Adventure|Sci-Fi" -> ["Action", "Adventure", "Sci-Fi"]
    # Handle "(no genres listed)" cases
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if pd.isnull(g) or g == "(no genres listed)" else str(g).split("|")
    )
    
    logger.info("Movies dataset cleaning complete.")
    return df

def clean_ratings(ratings_df):
    """
    Cleans ratings dataset:
    - Removes duplicates.
    - Validates rating ranges [0.5, 5.0].
    - Converts timestamps to readable UTC datetimes.
    
    Args:
        ratings_df (pd.DataFrame): Raw ratings DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned ratings DataFrame.
    """
    logger.info("Cleaning ratings dataset...")
    df = ratings_df.copy()
    
    # Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(subset=["userId", "movieId"], inplace=True)
    if df.shape != initial_shape:
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rating entries.")
        
    # Validate rating boundaries
    invalid_ratings = df[(df["rating"] < 0.5) | (df["rating"] > 5.0)]
    if not invalid_ratings.empty:
        logger.warning(f"Found {len(invalid_ratings)} ratings outside standard bounds [0.5, 5.0]!")
        # Filter out invalid ratings
        df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
        
    # Format timestamps to readable UTC datetimes
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    
    logger.info("Ratings dataset cleaning complete.")
    return df

def clean_tags(tags_df):
    """
    Cleans tags dataset:
    - Normalizes tags text (lowercases, strips whitespace).
    - Removes empty or null tag values.
    - Converts timestamps to readable UTC datetimes.
    
    Args:
        tags_df (pd.DataFrame): Raw tags DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned tags DataFrame.
    """
    logger.info("Cleaning tags dataset...")
    df = tags_df.copy()
    
    # Drop rows with null tags
    df.dropna(subset=["tag"], inplace=True)
    
    # Normalize tag names to lowercase to avoid variations like "Sci-Fi" and "sci-fi"
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    
    # Drop empty string tags
    df = df[df["tag"] != ""]
    
    # Remove duplicates
    df.drop_duplicates(subset=["userId", "movieId", "tag"], inplace=True)
    
    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    
    logger.info("Tags dataset cleaning complete.")
    return df

def merge_datasets(cleaned_movies_df, cleaned_ratings_df, cleaned_tags_df):
    """
    Aggregates rating statistics and tag content, then merges them with the movies DataFrame
    to build a unified movies representation.
    
    Args:
        cleaned_movies_df (pd.DataFrame): Cleaned movies.
        cleaned_ratings_df (pd.DataFrame): Cleaned ratings.
        cleaned_tags_df (pd.DataFrame): Cleaned tags.
        
    Returns:
        tuple: (processed_movies_df, processed_ratings_df)
    """
    logger.info("Starting dataset merge and aggregation...")
    
    # 1. Aggregate ratings: average score and count per movie
    logger.info("Aggregating rating metrics...")
    rating_stats = cleaned_ratings_df.groupby("movieId").agg(
        rating_mean=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()
    
    # Round mean ratings for convenience
    rating_stats["rating_mean"] = rating_stats["rating_mean"].round(2)
    
    # 2. Aggregate tags: join unique tags for each movie as a space-separated string
    logger.info("Aggregating tag strings...")
    movie_tags = cleaned_tags_df.groupby("movieId")["tag"].apply(
        lambda tags: "|".join(sorted(list(set(tags))))
    ).reset_index(name="movie_tags")
    
    # 3. Merge aggregated stats into movies DataFrame
    processed_movies = cleaned_movies_df.merge(rating_stats, on="movieId", how="left")
    processed_movies = processed_movies.merge(movie_tags, on="movieId", how="left")
    
    # Fill NaN values for movies with no ratings or tags
    processed_movies["rating_mean"] = processed_movies["rating_mean"].fillna(0.0)
    processed_movies["rating_count"] = processed_movies["rating_count"].fillna(0).astype(int)
    processed_movies["movie_tags"] = processed_movies["movie_tags"].fillna("")
    
    # Rename columns to ensure readability
    # Keeps movieId, title, genres, release_year, genres_list, rating_mean, rating_count, movie_tags
    logger.info("Dataset merge and aggregation completed successfully.")
    
    return processed_movies, cleaned_ratings_df

def validate_data(processed_movies_df, processed_ratings_df):
    """
    Validates data quality rules on final processed dataframes.
    Raises AssertionError if rules are violated.
    """
    logger.info("Running validation tests on processed DataFrames...")
    
    # Invariant 1: Non-emptiness
    assert not processed_movies_df.empty, "Processed movies DataFrame is empty!"
    assert not processed_ratings_df.empty, "Processed ratings DataFrame is empty!"
    
    # Invariant 2: Column schemas
    expected_movie_cols = ["movieId", "title", "genres", "release_year", "genres_list", "rating_mean", "rating_count", "movie_tags"]
    for col in expected_movie_cols:
        assert col in processed_movies_df.columns, f"Missing expected column '{col}' in processed movies!"
        
    expected_rating_cols = ["userId", "movieId", "rating", "timestamp"]
    for col in expected_rating_cols:
        assert col in processed_ratings_df.columns, f"Missing expected column '{col}' in processed ratings!"
        
    # Invariant 3: Data Integrity Ranges
    assert (processed_ratings_df["rating"] >= 0.5).all() and (processed_ratings_df["rating"] <= 5.0).all(), \
        "Ratings range violation detected!"
    assert (processed_movies_df["rating_count"] >= 0).all(), "Negative rating count detected!"
    assert (processed_movies_df["rating_mean"] >= 0.0).all() and (processed_movies_df["rating_mean"] <= 5.0).all(), \
        "Movie average ratings range violation!"
        
    logger.info("All processed data validation assertions PASSED.")

def run_data_preparation_pipeline():
    """
    Runs the entire dataset preparation workflow.
    Loads raw data, cleans tables, merges, validates, and saves.
    """
    logger.info("=== Starting Data Preparation Pipeline ===")
    
    # Step 1: Load
    movies, ratings, tags, links = load_raw_datasets()
    
    # Step 2: Clean
    cleaned_movies = clean_movies(movies)
    cleaned_ratings = clean_ratings(ratings)
    cleaned_tags = clean_tags(tags)
    
    # Step 3: Merge and Aggregate
    processed_movies, processed_ratings = merge_datasets(cleaned_movies, cleaned_ratings, cleaned_tags)
    
    # Step 4: Validate
    validate_data(processed_movies, processed_ratings)
    
    # Step 5: Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed datasets (helper handles folder creation and logging)
    save_dataframe(processed_movies, PROCESSED_DATA_DIR / "processed_movies.csv")
    save_dataframe(processed_ratings, PROCESSED_DATA_DIR / "processed_ratings.csv")
    
    logger.info("=== Data Preparation Pipeline Completed Successfully ===")
    return processed_movies, processed_ratings

if __name__ == "__main__":
    try:
        run_data_preparation_pipeline()
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)
