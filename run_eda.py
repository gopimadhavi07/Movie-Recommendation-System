import sys
import os
import pathlib
import pandas as pd

# Add the project root to path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.eda import run_full_eda, load_data
from src.utils.logger import get_logger

logger = get_logger("run_eda")

def output_insights(movies_df, ratings_df):
    """
    Computes key narrative insights from the EDA.
    
    Args:
        movies_df (pd.DataFrame): Processed movies.
        ratings_df (pd.DataFrame): Processed ratings.
    """
    logger.info("Extracting narrative insights...")
    
    # 1. Skewness of ratings
    avg_rating = ratings_df["rating"].mean()
    high_ratings_pct = (ratings_df["rating"] >= 4.0).sum() / len(ratings_df) * 100
    
    # 2. User sparsity / activity long tail
    user_counts = ratings_df["userId"].value_counts()
    median_ratings_per_user = user_counts.median()
    max_ratings_user = user_counts.max()
    min_ratings_user = user_counts.min()
    
    # 3. Movie popularity sparsity
    movie_counts = ratings_df["movieId"].value_counts()
    single_rating_movies_pct = (movie_counts == 1).sum() / len(movies_df) * 100
    
    # 4. Correlation count vs rating
    correlation = movies_df[movies_df["rating_count"] > 0]["rating_count"].corr(
        movies_df[movies_df["rating_count"] > 0]["rating_mean"]
    )
    
    print("\n" + "="*60)
    print("      KEY EXPLORATORY DATA ANALYSIS (EDA) INSIGHTS")
    print("="*60)
    print(f"1. Rating Distribution Bias:")
    print(f"   - Average Rating score is {avg_rating:.2f} out of 5.0.")
    print(f"   - {high_ratings_pct:.1f}% of all ratings are 4.0 stars or higher.")
    print(f"   - Insight: Ratings are heavily skewed towards positive marks. Users tend")
    print(f"     to rate movies they enjoy rather than auditing random selections.")
    
    print(f"\n2. User Interaction Sparsity (Power Users vs. Casual Users):")
    print(f"   - The median user has submitted {int(median_ratings_per_user)} ratings.")
    print(f"   - Maximum ratings by a single user is {max_ratings_user} (Power User).")
    print(f"   - Minimum ratings by a single user is {min_ratings_user} (threshold filter).")
    print(f"   - Insight: Collaborative filtering will be highly dependent on a small")
    print(f"     fraction of highly active power users, demanding regularization.")
    
    print(f"\n3. Movie Catalog Long Tail:")
    print(f"   - {single_rating_movies_pct:.1f}% of movies in the catalog have exactly 1 rating.")
    print(f"   - Insight: Presents a severe 'Cold-Start' problem for Collaborative")
    print(f"     filtering. Content-based algorithms are critical here.")
    
    print(f"\n4. Rating count vs. Average Quality Correlation:")
    print(f"   - Correlation coefficient is {correlation:.4f}.")
    print(f"   - Insight: Moderate positive correlation. Highly rated movies tend to")
    print(f"     be rated more frequently. Popular movies are generally well-liked.")
    print("="*60 + "\n")

if __name__ == "__main__":
    logger.info("Initializing run_eda wrapper...")
    try:
        # Run visual EDA pipeline
        run_full_eda()
        
        # Load and compute narrative insights
        movies_df, ratings_df = load_data()
        output_insights(movies_df, ratings_df)
        
        logger.info("[SUCCESS] EDA Visualizations and Narrative Reports Generated Successfully!")
    except Exception as e:
        logger.error(f"EDA validation failed: {e}", exc_info=True)
        sys.exit(1)
