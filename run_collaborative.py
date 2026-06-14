"""
run_collaborative.py
====================
Builds (or loads cached) the collaborative filtering recommendation models,
runs sample queries (Item-based CF and SVD user recommendations),
validates outputs, and prints a formatted report to stdout.

Usage:
    python run_collaborative.py              # use cached model
    python run_collaborative.py --rebuild    # force full rebuild
"""

import sys
import pathlib
import argparse
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.collaborative import (
    build_cf_model,
    get_item_based_recs,
    get_svd_recs
)
from src.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("run_collaborative")

# Sample queries
# Item CF: Movie IDs (e.g., 1 = Toy Story, 318 = Shawshank Redemption)
ITEM_QUERIES = [1, 318, 296, 2571]  # Toy Story, Shawshank, Pulp Fiction, The Matrix
# User SVD: User IDs
USER_QUERIES = [1, 50, 100, 200]

def _print_banner(text: str):
    width = 75
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def _print_item_recs(movie_title: str, recs_df: pd.DataFrame):
    print(f"\n  [ITEM-BASED] Because you liked: \"{movie_title}\"")
    print(f"  {'#':<4} {'Title':<42} {'Genres':<18} {'Distance':>8}")
    print("  " + "-" * 73)
    for rank, row in recs_df.iterrows():
        genre_short = str(row["genres"])[:17]
        title_short = str(row["title"])[:41]
        print(f"  {rank+1:<4} {title_short:<42} {genre_short:<18} {row['cf_distance']:>8.4f}")

def _print_user_recs(user_id: int, recs_df: pd.DataFrame):
    print(f"\n  [USER SVD] Personalized for User ID: {user_id}")
    print(f"  {'#':<4} {'Title':<42} {'Genres':<18} {'SVD Score':>9}")
    print("  " + "-" * 73)
    for rank, row in recs_df.iterrows():
        genre_short = str(row["genres"])[:17]
        title_short = str(row["title"])[:41]
        print(f"  {rank+1:<4} {title_short:<42} {genre_short:<18} {row['svd_score']:>9.4f}")


def main():
    parser = argparse.ArgumentParser(description="Collaborative Filtering Runner")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force full model rebuild even if artifacts exist."
    )
    args = parser.parse_args()

    logger.info("Starting collaborative filtering evaluation...")

    # Load Data
    movies_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_movies.csv")
    ratings_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_ratings.csv")

    # Build or load model
    matrix, item_knn, user_knn, svd_preds, user_map, movie_map, rev_user_map, rev_movie_map = build_cf_model(
        force_rebuild=args.rebuild
    )

    _print_banner("COLLABORATIVE FILTERING EVALUATION REPORT")
    
    print(f"\n{'-'*75}")
    print("  ITEM-BASED CF TEST (Users who watched X also watched Y)")
    for movie_id in ITEM_QUERIES:
        try:
            if movie_id in movie_map:
                title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
                recs = get_item_based_recs(movie_id, top_n=5, item_knn=item_knn, movie_map=movie_map, rev_movie_map=rev_movie_map, movies_df=movies_df)
                _print_item_recs(title, recs)
            else:
                logger.warning(f"Movie ID {movie_id} not in map.")
        except Exception as e:
            logger.error(f"Failed item CF query {movie_id}: {e}")

    print(f"\n{'-'*75}")
    print("  SVD PERSONALIZED TEST (Latent feature factorization)")
    for user_id in USER_QUERIES:
        try:
            if user_id in user_map:
                recs = get_svd_recs(user_id, top_n=5, svd_preds=svd_preds, user_map=user_map, rev_movie_map=rev_movie_map, movies_df=movies_df, ratings_df=ratings_df)
                _print_user_recs(user_id, recs)
            else:
                logger.warning(f"User ID {user_id} not in map.")
        except Exception as e:
            logger.error(f"Failed user SVD query {user_id}: {e}")

    _print_banner("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
