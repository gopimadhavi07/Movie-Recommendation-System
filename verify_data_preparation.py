import sys
import json
import pathlib
import pandas as pd
from collections import Counter

# Add the project root to path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_preparation import run_data_preparation_pipeline
from src.utils.logger import get_logger

logger = get_logger("verify_data_prep")

def calculate_summary_statistics(movies_df, ratings_df):
    """
    Computes key metrics and statistics from the processed DataFrames.
    
    Args:
        movies_df (pd.DataFrame): Processed movies.
        ratings_df (pd.DataFrame): Processed ratings.
        
    Returns:
        dict: Summary statistics.
    """
    logger.info("Computing summary statistics...")
    
    num_movies = int(movies_df["movieId"].nunique())
    num_users = int(ratings_df["userId"].nunique())
    num_ratings = int(len(ratings_df))
    
    # Calculate density & sparsity
    max_possible_ratings = num_users * num_movies
    density = num_ratings / max_possible_ratings if max_possible_ratings > 0 else 0
    sparsity = 1.0 - density
    
    avg_ratings_per_movie = float(ratings_df.groupby("movieId")["rating"].count().mean())
    avg_ratings_per_user = float(ratings_df.groupby("userId")["rating"].count().mean())
    
    # Release year stats
    years = movies_df["release_year"].dropna().astype(int)
    min_year = int(years.min()) if not years.empty else None
    max_year = int(years.max()) if not years.empty else None
    
    # Genre frequency
    genre_counts = Counter()
    for genres in movies_df["genres_list"]:
        genre_counts.update(genres)
    
    top_genres = dict(genre_counts.most_common(10))
    
    # Top 5 most rated movies
    most_rated = movies_df.sort_values(by="rating_count", ascending=False).head(5)[
        ["title", "rating_count", "rating_mean"]
    ].to_dict(orient="records")
    
    # Top 5 highest rated movies with at least 50 ratings
    highest_rated = movies_df[movies_df["rating_count"] >= 50].sort_values(
        by="rating_mean", ascending=False
    ).head(5)[
        ["title", "rating_count", "rating_mean"]
    ].to_dict(orient="records")
    
    stats = {
        "num_movies": num_movies,
        "num_users": num_users,
        "num_ratings": num_ratings,
        "density_pct": round(density * 100, 4),
        "sparsity_pct": round(sparsity * 100, 4),
        "avg_ratings_per_movie": round(avg_ratings_per_movie, 2),
        "avg_ratings_per_user": round(avg_ratings_per_user, 2),
        "min_release_year": min_year,
        "max_release_year": max_year,
        "top_genres": top_genres,
        "most_rated_movies": most_rated,
        "highest_rated_movies": highest_rated
    }
    
    # Save statistics to JSON
    summary_path = PROJECT_ROOT / "data" / "processed" / "summary_statistics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Summary statistics successfully saved to {summary_path}")
    return stats

def print_summary(stats):
    """
    Prints statistics cleanly to stdout.
    """
    print("\n" + "="*50)
    print("      MOVIELENS DATASET SUMMARY STATISTICS")
    print("="*50)
    print(f"Total Movies:                {stats['num_movies']:,}")
    print(f"Total Users:                 {stats['num_users']:,}")
    print(f"Total Ratings:               {stats['num_ratings']:,}")
    print(f"Interaction Matrix Density:  {stats['density_pct']}%")
    print(f"Interaction Matrix Sparsity: {stats['sparsity_pct']}%")
    print(f"Avg Ratings per Movie:       {stats['avg_ratings_per_movie']}")
    print(f"Avg Ratings per User:        {stats['avg_ratings_per_user']}")
    print(f"Release Year Range:          {stats['min_release_year']} - {stats['max_release_year']}")
    
    print("\nTop Genres Distribution:")
    for genre, count in stats["top_genres"].items():
        print(f" - {genre:<15} {count:,}")
        
    print("\nTop 5 Most Rated Movies:")
    for i, m in enumerate(stats["most_rated_movies"], 1):
        print(f" {i}. {m['title']} ({m['rating_count']} ratings, Avg: {m['rating_mean']})")
        
    print("\nTop 5 Highest Rated Movies (>=50 ratings):")
    for i, m in enumerate(stats["highest_rated_movies"], 1):
        print(f" {i}. {m['title']} ({m['rating_count']} ratings, Avg: {m['rating_mean']})")
    print("="*50 + "\n")

if __name__ == "__main__":
    logger.info("Starting verify_data_preparation validation...")
    try:
        # Run preparation pipeline
        processed_movies, processed_ratings = run_data_preparation_pipeline()
        
        # Calculate stats
        stats = calculate_summary_statistics(processed_movies, processed_ratings)
        
        # Display summary
        print_summary(stats)
        
        logger.info("[SUCCESS] Dataset Preparation and Integrity Audits Completed Successfully!")
    except Exception as e:
        logger.error(f"Verification script failed: {e}", exc_info=True)
        sys.exit(1)
