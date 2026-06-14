"""
run_content_based.py
====================
Builds (or loads cached) the content-based recommendation model,
runs a curated set of evaluation queries, validates outputs, and
prints a formatted report to stdout.

Usage:
    python run_content_based.py              # use cached model
    python run_content_based.py --rebuild    # force full rebuild
"""

import sys
import pathlib
import argparse
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.content_based import (
    build_content_based_model,
    get_recommendations,
    validate_recommendations,
)
from src.utils.logger import get_logger

logger = get_logger("run_content_based")

# ── Evaluation queries ──────────────────────────────────────────────────────
# Format: (query_title, expected_genre_hint)
EVAL_QUERIES = [
    ("Toy Story",                       "Animation|Children"),
    ("The Dark Knight",                  "Action|Crime|Thriller"),
    ("Inception",                        "Action|Sci-Fi|Thriller"),
    ("Schindler's List",                 "Drama|War"),
    ("Forrest Gump",                     "Comedy|Drama|Romance"),
    ("Silence of the Lambs, The",        "Crime|Horror|Thriller"),
]


def _print_banner(text: str):
    width = 66
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _print_recs(query: str, recs_df: pd.DataFrame):
    print(f"\n  Query movie : \"{query}\"")
    print(f"  {'#':<4} {'Title':<42} {'Genres':<28} {'Sim':>6}")
    print("  " + "-" * 84)
    for rank, row in recs_df.iterrows():
        genre_short = str(row["genres"])[:27]
        title_short = str(row["title"])[:41]
        print(f"  {rank+1:<4} {title_short:<42} {genre_short:<28} {row['similarity_score']:>6.4f}")


def run_evaluation(movies_df: pd.DataFrame, sim_matrix, title_to_idx: dict, top_n: int = 8):
    """
    Iterates over EVAL_QUERIES, retrieves recommendations, validates them,
    and prints a summary report.
    """
    _print_banner("CONTENT-BASED RECOMMENDATION EVALUATION REPORT")

    all_passed = True
    for query_title, expected_genre in EVAL_QUERIES:
        print(f"\n{'-'*66}")
        try:
            recs = get_recommendations(
                query_title, movies_df, sim_matrix, title_to_idx, top_n=top_n
            )
            _print_recs(query_title, recs)

            # Validation
            val = validate_recommendations(recs, top_n=top_n)
            status = "PASS" if val["passed"] else "FAIL"
            if not val["passed"]:
                all_passed = False
                for chk in val["checks"]:
                    if not chk["passed"]:
                        logger.warning(f"Validation failed — {chk['check']}")

            # Genre overlap check (informational)
            expected_tags = set(expected_genre.split("|"))
            rec_genres = " ".join(recs["genres"].fillna("").tolist())
            overlap = sum(1 for g in expected_tags if g in rec_genres)
            genre_hit = f"{overlap}/{len(expected_tags)} expected genres in top-{top_n}"

            print(f"\n  Validation : [{status}]")
            print(f"  Genre Hit  : {genre_hit}")

        except ValueError as e:
            logger.error(f"Query '{query_title}' failed: {e}")
            all_passed = False

    _print_banner(f"OVERALL RESULT : {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")


def main():
    parser = argparse.ArgumentParser(description="Content-Based Recommendation Runner")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force full model rebuild even if artefacts exist."
    )
    args = parser.parse_args()

    logger.info("Starting content-based model evaluation...")

    # Build or load model
    movies_df, sim_matrix, title_to_idx = build_content_based_model(
        force_rebuild=args.rebuild
    )

    # Run evaluation
    run_evaluation(movies_df, sim_matrix, title_to_idx)

    logger.info("[SUCCESS] Content-based evaluation complete.")


if __name__ == "__main__":
    main()
