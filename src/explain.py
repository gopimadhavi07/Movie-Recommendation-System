"""
src/explain.py
==============
Explainable AI (XAI) Module for Movie Recommendations.
Generates human-readable, natural language justifications for why a movie is recommended.
"""

import pandas as pd
import random

def generate_explanation(rec_movie: pd.Series, user_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> str:
    """
    Generates a reason string for why `rec_movie` was recommended based on `user_ratings`.
    """
    if user_ratings.empty:
        return "Recommended because it is currently trending and highly popular."
        
    # Get user's top rated movies
    top_user_ratings = user_ratings[user_ratings['rating'] >= 4.0].sort_values('rating', ascending=False)
    
    if top_user_ratings.empty:
        return "Recommended based on overall collaborative patterns."
        
    # Find genre overlap
    rec_genres = set(str(rec_movie['genres']).split('|'))
    
    # Check if there's a specific highly rated movie with high genre overlap
    best_overlap = 0
    best_movie_title = ""
    shared_genres = []
    
    for _, row in top_user_ratings.iterrows():
        m_id = row['movie_id']
        m_title = movies_df[movies_df['movieId'] == m_id]['title'].values[0]
        m_genres = set(str(movies_df[movies_df['movieId'] == m_id]['genres'].values[0]).split('|'))
        
        overlap = rec_genres.intersection(m_genres)
        if len(overlap) > best_overlap:
            best_overlap = len(overlap)
            best_movie_title = m_title
            shared_genres = list(overlap)
            
    if best_overlap > 0:
        genre_str = shared_genres[0] if len(shared_genres) == 1 else f"{shared_genres[0]} and {shared_genres[1]}"
        templates = [
            f"Because you highly rated **{best_movie_title}**, which shares similar {genre_str} themes.",
            f"Similar to **{best_movie_title}** based on your interest in {genre_str} movies.",
            f"Recommended for its {genre_str} elements, much like **{best_movie_title}** which you enjoyed."
        ]
        return random.choice(templates)
        
    return "Based on your watch history and similar users' preferences."
