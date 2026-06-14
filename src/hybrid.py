"""
src/hybrid.py
=============
Hybrid Recommendation Engine.
Combines Content-Based (Cosine Similarity) and Collaborative (SVD) scores.
Handles cold-start via Trending fallbacks.
"""

import pandas as pd
import numpy as np
from src.discovery import calculate_trending_score
from src.content_based import get_recommendations as get_cb_recs
from src.collaborative import get_svd_recs
from src.utils.logger import get_logger

logger = get_logger("movie_rec.hybrid")

def get_hybrid_recommendations(
    user_id: int, 
    user_ratings: pd.DataFrame, 
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    cb_sim_matrix: np.ndarray, 
    cb_title_idx: dict,
    svd_preds: np.ndarray,
    cf_user_map: dict,
    cf_rev_movie_map: dict,
    top_n: int = 10,
    mood_genres: list = None
) -> pd.DataFrame:
    """
    Generates hybrid recommendations.
    If user has 0 ratings: returns Trending.
    If user has ratings but not in SVD model (new user): returns Content-Based on top rated movie.
    If user is in SVD model: combines SVD score and Content-Based score of top 3 recent highly-rated movies.
    """
    
    # 1. COLD START - TRENDING
    if user_ratings.empty:
        logger.info(f"User {user_id} has no ratings. Returning Trending movies.")
        recs = calculate_trending_score(movies_df).head(top_n)
        recs['hybrid_score'] = recs['trending_score']
        return _apply_mood_boost(recs, mood_genres)
        
    # User's top rated movies
    top_rated = user_ratings[user_ratings['rating'] >= 4.0].sort_values('rating', ascending=False)
    
    if top_rated.empty:
        top_rated = user_ratings.sort_values('rating', ascending=False)
        
    top_movie_ids = top_rated['movie_id'].head(3).tolist()
    watched_movie_ids = user_ratings['movie_id'].tolist()
    
    # 2. CONTENT-BASED SCORES
    # We gather the similarity vectors for the user's top movies
    cb_scores = np.zeros(len(movies_df))
    valid_cb = 0
    for m_id in top_movie_ids:
        title = movies_df[movies_df['movieId'] == m_id]['title'].values[0]
        q = title.lower().strip()
        matched_key = None
        if q in cb_title_idx: matched_key = q
        else:
            matches = [k for k in cb_title_idx if q in k]
            if matches: matched_key = min(matches, key=len)
            
        if matched_key:
            idx = cb_title_idx[matched_key]
            cb_scores += cb_sim_matrix[idx]
            valid_cb += 1
            
    if valid_cb > 0:
        cb_scores = cb_scores / valid_cb  # Average similarity
        
    # 3. COLLABORATIVE SVD SCORES
    svd_scores = np.zeros(len(movies_df))
    has_svd = False
    
    if user_id in cf_user_map:
        logger.info(f"User {user_id} found in SVD model. Using Collaborative + Content blend.")
        user_idx = cf_user_map[user_id]
        preds = svd_preds[user_idx]
        
        # Map SVD array indices back to DataFrame rows
        for i, pred_val in enumerate(preds):
            m_id = cf_rev_movie_map[i]
            # Find the row index in movies_df for this m_id
            row_idx = movies_df.index[movies_df['movieId'] == m_id].tolist()
            if row_idx:
                svd_scores[row_idx[0]] = pred_val
        has_svd = True
    else:
        logger.info(f"User {user_id} NOT in SVD model. Using pure Content-Based.")

    # 4. HYBRIDIZATION
    # Normalize SVD scores to [0, 1] range to mix with Cosine similarity
    if has_svd and svd_scores.max() > svd_scores.min():
        svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min())
    else:
        svd_norm = np.zeros(len(movies_df))
        
    # Weights
    w_cb = 0.6 if not has_svd else 0.4
    w_cf = 0.0 if not has_svd else 0.6
    
    final_scores = (w_cb * cb_scores) + (w_cf * svd_norm)
    
    # 5. FILTERING AND SORTING
    result_df = movies_df.copy()
    result_df['hybrid_score'] = final_scores
    
    # Remove already watched movies
    result_df = result_df[~result_df['movieId'].isin(watched_movie_ids)]
    
    # Sort
    result_df = result_df.sort_values('hybrid_score', ascending=False)
    
    # Apply Mood Boost if selected
    result_df = _apply_mood_boost(result_df, mood_genres)
    
    return result_df.head(top_n).reset_index(drop=True)


def _apply_mood_boost(df: pd.DataFrame, mood_genres: list) -> pd.DataFrame:
    """Boosts the score of movies containing the specified mood genres."""
    if not mood_genres:
        return df
        
    df = df.copy()
    pattern = "|".join(mood_genres)
    
    # Give a 25% score bump to movies matching the mood
    mask = df['genres'].str.contains(pattern, case=False, na=False)
    if 'hybrid_score' in df.columns:
        df.loc[mask, 'hybrid_score'] *= 1.25
        df = df.sort_values('hybrid_score', ascending=False)
    elif 'trending_score' in df.columns:
        df.loc[mask, 'trending_score'] *= 1.25
        df = df.sort_values('trending_score', ascending=False)
        
    # Also prioritize putting mood matches at the very top if scores are close
    # by adding a temporary sort tier
    df['is_mood'] = mask.astype(int)
    sort_col = 'hybrid_score' if 'hybrid_score' in df.columns else 'trending_score'
    df = df.sort_values(['is_mood', sort_col], ascending=[False, False]).drop(columns=['is_mood'])
    
    return df
