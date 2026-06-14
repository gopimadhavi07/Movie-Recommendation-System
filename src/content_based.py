"""
src/content_based.py
====================
Content-Based Movie Recommendation System.

Pipeline:
1. Feature Engineering  — combine genres + tags + title tokens into a 'soup'
2. TF-IDF Vectorization — transform soup into a weighted term-document matrix
3. Cosine Similarity    — compute pairwise similarity across all movies
4. Recommendation       — rank and return top-N most similar movies for a query

Artifacts saved to models/:
  - tfidf_vectorizer.joblib
  - tfidf_matrix.joblib
  - cosine_sim_matrix.joblib
  - content_movies_index.joblib  (title → DataFrame index mapping)
"""

import re
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("movie_rec.content_based")


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def _clean_token(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_feature_soup(movies_df: pd.DataFrame) -> pd.Series:
    """
    Combines genre names, movie tags, and cleaned title words into a
    single whitespace-separated string (the 'soup') for each movie.

    Weights:
      - Genres are repeated 3× to boost genre similarity.
      - Tags appear once.
      - Title words appear once (stopwords handled by TF-IDF).

    Args:
        movies_df: Processed movies DataFrame.

    Returns:
        pd.Series of soup strings, aligned to movies_df index.
    """
    logger.info("Building feature soup for every movie...")

    def _genres_tokens(genre_str):
        if pd.isnull(genre_str) or genre_str == "(no genres listed)":
            return ""
        genres = [_clean_token(g.replace("-", "")) for g in str(genre_str).split("|")]
        # Repeat 3× to upweight genre signal
        return " ".join(genres * 3)

    def _tag_tokens(tag_str):
        if pd.isnull(tag_str) or tag_str == "":
            return ""
        tags = [_clean_token(t) for t in str(tag_str).split("|")]
        return " ".join(tags)

    def _title_tokens(title_str):
        return _clean_token(str(title_str))

    genre_part = movies_df["genres"].apply(_genres_tokens)
    tag_part   = movies_df["movie_tags"].apply(_tag_tokens)
    title_part = movies_df["title"].apply(_title_tokens)

    soup = genre_part + " " + tag_part + " " + title_part
    soup = soup.str.strip()

    logger.info(f"Feature soup built for {len(soup):,} movies.")
    return soup


# ─────────────────────────────────────────────
# 2. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────

def build_tfidf_matrix(soup: pd.Series):
    """
    Fits a TF-IDF vectorizer on the feature soup and returns the
    sparse term-document matrix together with the fitted vectorizer.

    Args:
        soup: Series of feature strings (one per movie).

    Returns:
        tuple: (vectorizer, tfidf_matrix)
    """
    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,                 # ignore very rare tokens
        max_df=0.95,              # ignore nearly-universal tokens
        stop_words="english",
        sublinear_tf=True         # apply log(1+tf) smoothing
    )
    tfidf_matrix = vectorizer.fit_transform(soup)
    n_movies, n_features = tfidf_matrix.shape
    logger.info(
        f"TF-IDF matrix built: {n_movies:,} movies × {n_features:,} features  "
        f"(density {tfidf_matrix.nnz / (n_movies * n_features) * 100:.3f}%)"
    )
    return vectorizer, tfidf_matrix


# ─────────────────────────────────────────────
# 3. COSINE SIMILARITY
# ─────────────────────────────────────────────

def compute_cosine_similarity(tfidf_matrix):
    """
    Computes the full pairwise cosine-similarity matrix.

    For the MovieLens small dataset (~9 k movies) this is a manageable
    dense 9k × 9k float32 matrix (~324 MB).  For larger corpora a
    chunked or approximate-NN approach should be used instead.

    Args:
        tfidf_matrix: Sparse matrix from build_tfidf_matrix().

    Returns:
        np.ndarray of shape (n_movies, n_movies).
    """
    logger.info("Computing pairwise cosine similarity matrix (this may take a moment)...")
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
    logger.info(f"Cosine similarity matrix shape: {sim_matrix.shape}")
    return sim_matrix


# ─────────────────────────────────────────────
# 4. MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model_artifacts(vectorizer, tfidf_matrix, sim_matrix, title_to_idx: dict):
    """
    Persists all model artefacts to models/ using Joblib compression.

    Args:
        vectorizer:    Fitted TfidfVectorizer.
        tfidf_matrix:  Sparse TF-IDF matrix.
        sim_matrix:    Dense cosine similarity matrix.
        title_to_idx:  Dict mapping lowercased title → integer row index.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "tfidf_vectorizer.joblib":     vectorizer,
        "tfidf_matrix.joblib":         tfidf_matrix,
        "cosine_sim_matrix.joblib":    sim_matrix,
        "content_movies_index.joblib": title_to_idx,
    }

    for filename, obj in artifacts.items():
        path = MODELS_DIR / filename
        joblib.dump(obj, path, compress=3)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {filename}  ({size_mb:.2f} MB)")


def load_model_artifacts():
    """
    Loads pre-saved model artefacts from models/.

    Returns:
        tuple: (vectorizer, tfidf_matrix, sim_matrix, title_to_idx)

    Raises:
        FileNotFoundError: If any artefact is missing.
    """
    required = [
        "tfidf_vectorizer.joblib",
        "tfidf_matrix.joblib",
        "cosine_sim_matrix.joblib",
        "content_movies_index.joblib",
    ]
    for fname in required:
        if not (MODELS_DIR / fname).exists():
            raise FileNotFoundError(
                f"Model artefact '{fname}' not found in {MODELS_DIR}. "
                "Run build_content_based_model() first."
            )

    logger.info("Loading content-based model artefacts from disk...")
    vectorizer    = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_matrix  = joblib.load(MODELS_DIR / "tfidf_matrix.joblib")
    sim_matrix    = joblib.load(MODELS_DIR / "cosine_sim_matrix.joblib")
    title_to_idx  = joblib.load(MODELS_DIR / "content_movies_index.joblib")
    logger.info("All model artefacts loaded successfully.")
    return vectorizer, tfidf_matrix, sim_matrix, title_to_idx


# ─────────────────────────────────────────────
# 5. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def _fuzzy_match_title(query: str, title_to_idx: dict) -> str | None:
    """
    Finds the closest title key in title_to_idx to the query string.
    First tries exact match (lowercased), then substring match.

    Args:
        query:        User-supplied movie title.
        title_to_idx: Mapping of lowercased titles → row indices.

    Returns:
        Matched key string, or None if no match found.
    """
    q = query.lower().strip()
    if q in title_to_idx:
        return q
    # Substring search
    matches = [k for k in title_to_idx if q in k]
    if matches:
        # Return the shortest matching title (most specific)
        return min(matches, key=len)
    return None


def get_recommendations(
    movie_title: str,
    movies_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    title_to_idx: dict,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Returns the top-N content-similar movies for a given title.

    Args:
        movie_title:  Title string supplied by the user.
        movies_df:    Processed movies DataFrame (must align with sim_matrix rows).
        sim_matrix:   Pairwise cosine similarity matrix.
        title_to_idx: Lowercased title → row index mapping.
        top_n:        Number of recommendations to return.

    Returns:
        pd.DataFrame with columns [movieId, title, genres, release_year,
                                   rating_mean, rating_count, similarity_score].

    Raises:
        ValueError: If the title cannot be matched to any known movie.
    """
    matched_key = _fuzzy_match_title(movie_title, title_to_idx)
    if matched_key is None:
        raise ValueError(
            f"Movie '{movie_title}' not found in the catalog. "
            "Check the title spelling or use the search bar."
        )

    idx = title_to_idx[matched_key]
    logger.info(f"Generating recommendations for: '{movies_df.iloc[idx]['title']}'  (row {idx})")

    # Get similarity scores for every movie against the query movie
    sim_scores = list(enumerate(sim_matrix[idx]))

    # Sort descending by score, exclude the query movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

    movie_indices = [i for i, _ in sim_scores]
    similarity_vals = [round(float(s), 4) for _, s in sim_scores]

    result = movies_df.iloc[movie_indices][
        ["movieId", "title", "genres", "release_year", "rating_mean", "rating_count"]
    ].copy().reset_index(drop=True)
    result["similarity_score"] = similarity_vals

    logger.info(f"Returning {len(result)} recommendations.")
    return result


# ─────────────────────────────────────────────
# 6. VALIDATION
# ─────────────────────────────────────────────

def validate_recommendations(recs_df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Runs sanity checks on a recommendations DataFrame.

    Args:
        recs_df: Output of get_recommendations().
        top_n:   Expected number of results.

    Returns:
        dict with keys: passed (bool), checks (list of check dicts).
    """
    checks = []

    # Check 1: Non-empty
    ok = not recs_df.empty
    checks.append({"check": "Non-empty results",         "passed": ok})

    # Check 2: Count
    ok = len(recs_df) <= top_n
    checks.append({"check": f"At most {top_n} results",  "passed": ok})

    # Check 3: Similarity scores in [0, 1]
    ok = recs_df["similarity_score"].between(0.0, 1.0).all()
    checks.append({"check": "Scores in [0, 1]",           "passed": ok})

    # Check 4: Scores are sorted descending
    ok = recs_df["similarity_score"].is_monotonic_decreasing
    checks.append({"check": "Scores sorted descending",   "passed": ok})

    # Check 5: Required columns present
    required_cols = ["movieId", "title", "genres", "similarity_score"]
    ok = all(c in recs_df.columns for c in required_cols)
    checks.append({"check": "Required columns present",   "passed": ok})

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def build_content_based_model(force_rebuild: bool = False):
    """
    End-to-end pipeline: load processed data → feature soup → TF-IDF
    → cosine similarity → save artefacts.

    Args:
        force_rebuild: If True, re-builds even when artefacts already exist.

    Returns:
        tuple: (movies_df, sim_matrix, title_to_idx)
    """
    # Short-circuit if artefacts already exist
    if not force_rebuild:
        try:
            _, _, sim_matrix, title_to_idx = load_model_artifacts()
            movies_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_movies.csv")
            logger.info("Existing model artefacts found — skipping rebuild.")
            return movies_df, sim_matrix, title_to_idx
        except FileNotFoundError:
            logger.info("No existing artefacts found — building model from scratch.")

    logger.info("=== Building Content-Based Recommendation Model ===")

    # Load data
    movies_path  = PROCESSED_DATA_DIR / "processed_movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(
            "Processed movies not found. Run data preparation pipeline first."
        )
    movies_df = pd.read_csv(movies_path)
    logger.info(f"Loaded movies_df: {movies_df.shape}")

    # Feature engineering
    soup = build_feature_soup(movies_df)

    # TF-IDF
    vectorizer, tfidf_matrix = build_tfidf_matrix(soup)

    # Cosine similarity
    sim_matrix = compute_cosine_similarity(tfidf_matrix)

    # Build title → index mapping (lowercased for fuzzy matching)
    title_to_idx = {
        title.lower(): idx
        for idx, title in enumerate(movies_df["title"])
    }

    # Persist artefacts
    save_model_artifacts(vectorizer, tfidf_matrix, sim_matrix, title_to_idx)

    logger.info("=== Content-Based Model Build Complete ===")
    return movies_df, sim_matrix, title_to_idx


if __name__ == "__main__":
    import sys
    try:
        build_content_based_model(force_rebuild=True)
    except Exception as exc:
        logger.error(f"Model build failed: {exc}", exc_info=True)
        sys.exit(1)
