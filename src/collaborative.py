"""
src/collaborative.py
====================
Collaborative Filtering Movie Recommendation System.

Pipeline:
1. Data Matrix       — Pivot ratings into a sparse User-Item Matrix.
2. Item-Based CF     — KNN (Cosine similarity) between items (movies).
3. User-Based CF     — KNN (Cosine similarity) between users.
4. Matrix Factor.    — Truncated SVD for latent user/movie features and predicted ratings.
5. Recommendation    — Retrieve top-N collaborative recommendations based on user history.

Artifacts saved to models/:
  - user_item_matrix.joblib
  - item_knn_model.joblib
  - user_knn_model.joblib
  - svd_preds_matrix.joblib
  - cf_user_index.joblib
  - cf_movie_index.joblib
"""

import sys
import pathlib
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("movie_rec.collaborative")

# ─────────────────────────────────────────────
# 1. DATA PREPARATION & MATRIX CREATION
# ─────────────────────────────────────────────

def build_user_item_matrix(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    """
    Creates a sparse User-Item interaction matrix.

    Args:
        ratings_df: Processed ratings DataFrame (userId, movieId, rating).
        movies_df: Processed movies DataFrame (to ensure all valid movies are included).

    Returns:
        tuple: (sparse_matrix, user_to_row_idx, movie_to_col_idx, row_to_user, col_to_movie)
    """
    logger.info("Building User-Item Matrix...")
    
    # Ensure consistent mapping
    unique_users = sorted(ratings_df['userId'].unique())
    unique_movies = sorted(movies_df['movieId'].unique())
    
    user_to_row_idx = {user: idx for idx, user in enumerate(unique_users)}
    movie_to_col_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
    
    row_to_user = {idx: user for user, idx in user_to_row_idx.items()}
    col_to_movie = {idx: movie for movie, idx in movie_to_col_idx.items()}
    
    # Map IDs to indices
    row_indices = ratings_df['userId'].map(user_to_row_idx).dropna().astype(int)
    col_indices = ratings_df['movieId'].map(movie_to_col_idx).dropna().astype(int)
    ratings = ratings_df.loc[row_indices.index, 'rating']
    
    sparse_matrix = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_movies))
    )
    
    logger.info(f"User-Item Matrix created: {sparse_matrix.shape[0]} users x {sparse_matrix.shape[1]} movies.")
    return sparse_matrix, user_to_row_idx, movie_to_col_idx, row_to_user, col_to_movie


# ─────────────────────────────────────────────
# 2. KNN MODELS (ITEM-BASED & USER-BASED)
# ─────────────────────────────────────────────

def train_knn_models(user_item_matrix: csr_matrix):
    """
    Trains NearestNeighbors models for User-User and Item-Item similarity.
    """
    logger.info("Training Item-Based KNN Model...")
    # Item-based uses transposed matrix (movies x users)
    item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    item_knn.fit(user_item_matrix.T)
    
    logger.info("Training User-Based KNN Model...")
    # User-based uses original matrix (users x movies)
    user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    user_knn.fit(user_item_matrix)
    
    return item_knn, user_knn


# ─────────────────────────────────────────────
# 3. MATRIX FACTORIZATION (SVD)
# ─────────────────────────────────────────────

def compute_svd_predictions(user_item_matrix: csr_matrix, k: int = 50):
    """
    Computes SVD and reconstructs the matrix to predict all ratings.
    """
    logger.info(f"Computing SVD with k={k} latent features...")
    # SVD requires float matrix
    matrix_float = user_item_matrix.asfptype()
    
    # Calculate user means to center the ratings
    user_ratings_mean = np.array(matrix_float.mean(axis=1)).flatten()
    # Centering (handle zero rows gracefully)
    matrix_centered = matrix_float.copy()
    matrix_centered.data -= np.take(user_ratings_mean, matrix_centered.nonzero()[0])
    
    # Perform SVD
    # SVD can fail if k >= min(matrix.shape). We bound k.
    k_actual = min(k, min(matrix_float.shape) - 1)
    U, sigma, Vt = svds(matrix_centered, k=k_actual)
    
    # Reconstruct the predicted ratings matrix
    sigma_diag = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
    
    logger.info("SVD prediction matrix computed.")
    return predicted_ratings


# ─────────────────────────────────────────────
# 4. MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_cf_artifacts(matrix, item_knn, user_knn, svd_preds, user_map, movie_map, rev_user_map, rev_movie_map):
    """Persists CF artifacts to models/"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        "user_item_matrix.joblib": matrix,
        "item_knn_model.joblib": item_knn,
        "user_knn_model.joblib": user_knn,
        "svd_preds_matrix.joblib": svd_preds,
        "cf_user_map.joblib": user_map,
        "cf_movie_map.joblib": movie_map,
        "cf_rev_user_map.joblib": rev_user_map,
        "cf_rev_movie_map.joblib": rev_movie_map,
    }
    
    for filename, obj in artifacts.items():
        path = MODELS_DIR / filename
        joblib.dump(obj, path, compress=3)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {filename} ({size_mb:.2f} MB)")

def load_cf_artifacts():
    """Loads CF artifacts from models/"""
    required = [
        "user_item_matrix.joblib", "item_knn_model.joblib", "user_knn_model.joblib",
        "svd_preds_matrix.joblib", "cf_user_map.joblib", "cf_movie_map.joblib",
        "cf_rev_user_map.joblib", "cf_rev_movie_map.joblib"
    ]
    for fname in required:
        if not (MODELS_DIR / fname).exists():
            raise FileNotFoundError(f"CF artifact '{fname}' not found. Run build_cf_model() first.")
            
    logger.info("Loading collaborative filtering artifacts...")
    return tuple(joblib.load(MODELS_DIR / fname) for fname in required)


# ─────────────────────────────────────────────
# 5. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def get_item_based_recs(movie_id: int, top_n: int, item_knn, movie_map, rev_movie_map, movies_df: pd.DataFrame):
    """Recommends movies similar to a given movie based on user rating patterns."""
    if movie_id not in movie_map:
        raise ValueError(f"Movie ID {movie_id} not found in CF model.")
        
    idx = movie_map[movie_id]
    # Find neighbors (skip the first one as it's the movie itself)
    distances, indices = item_knn.kneighbors(item_knn._fit_X[idx].reshape(1, -1), n_neighbors=top_n+1)
    
    rec_indices = indices.flatten()[1:]
    rec_distances = distances.flatten()[1:]
    
    rec_movie_ids = [rev_movie_map[i] for i in rec_indices]
    
    # Map to DataFrame
    result = movies_df[movies_df['movieId'].isin(rec_movie_ids)].copy()
    # Sort by distance
    result['cf_distance'] = result['movieId'].map(dict(zip(rec_movie_ids, rec_distances)))
    result = result.sort_values('cf_distance').reset_index(drop=True)
    return result

def get_svd_recs(user_id: int, top_n: int, svd_preds: np.ndarray, user_map, rev_movie_map, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
    """Recommends movies for a user based on SVD predicted ratings."""
    if user_id not in user_map:
        raise ValueError(f"User ID {user_id} not found in CF model.")
        
    user_idx = user_map[user_id]
    user_preds = svd_preds[user_idx]
    
    # Movies already rated by user
    user_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    # Sort predictions descending
    movie_indices = np.argsort(user_preds)[::-1]
    
    rec_movie_ids = []
    rec_scores = []
    
    for idx in movie_indices:
        m_id = rev_movie_map[idx]
        if m_id not in user_rated_movies:
            rec_movie_ids.append(m_id)
            rec_scores.append(user_preds[idx])
        if len(rec_movie_ids) == top_n:
            break
            
    result = movies_df[movies_df['movieId'].isin(rec_movie_ids)].copy()
    result['svd_score'] = result['movieId'].map(dict(zip(rec_movie_ids, rec_scores)))
    result = result.sort_values('svd_score', ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def build_cf_model(force_rebuild: bool = False):
    """End-to-end CF pipeline."""
    if not force_rebuild:
        try:
            artifacts = load_cf_artifacts()
            logger.info("Existing CF model artifacts found — skipping rebuild.")
            return artifacts
        except FileNotFoundError:
            logger.info("No existing CF artifacts found — building model.")
            
    logger.info("=== Building Collaborative Filtering Model ===")
    
    movies_path = PROCESSED_DATA_DIR / "processed_movies.csv"
    ratings_path = PROCESSED_DATA_DIR / "processed_ratings.csv"
    
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # 1. Build Matrix
    matrix, user_map, movie_map, rev_user_map, rev_movie_map = build_user_item_matrix(ratings_df, movies_df)
    
    # 2. KNN Models
    item_knn, user_knn = train_knn_models(matrix)
    
    # 3. SVD
    svd_preds = compute_svd_predictions(matrix, k=50)
    
    # 4. Save
    save_cf_artifacts(matrix, item_knn, user_knn, svd_preds, user_map, movie_map, rev_user_map, rev_movie_map)
    
    logger.info("=== CF Model Build Complete ===")
    return matrix, item_knn, user_knn, svd_preds, user_map, movie_map, rev_user_map, rev_movie_map

if __name__ == "__main__":
    try:
        build_cf_model(force_rebuild=True)
    except Exception as e:
        logger.error(f"CF Model build failed: {e}", exc_info=True)
        sys.exit(1)
