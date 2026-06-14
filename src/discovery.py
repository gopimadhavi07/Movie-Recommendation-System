"""
src/discovery.py
================
Discovery module for Search and Trending movies.
Includes robust Bayesian average calculation for trending movies to prevent 
low-volume movies with perfect scores from dominating the rankings.
"""

import pandas as pd
import numpy as np

def calculate_trending_score(movies_df: pd.DataFrame, m_percentile: float = 0.85) -> pd.DataFrame:
    """
    Calculates a Bayesian Average for each movie to determine trending ranking.
    Formula: (v / (v + m)) * R + (m / (v + m)) * C
      v = number of ratings for the movie
      m = minimum ratings required to be listed in chart
      R = average rating of the movie
      C = mean vote across the whole report
    """
    # Create a copy so we don't modify the original
    df = movies_df.copy()
    
    # Calculate C (mean vote across all movies)
    C = df['rating_mean'].mean()
    
    # Calculate m (minimum ratings required to be listed, e.g. 85th percentile)
    m = df['rating_count'].quantile(m_percentile)
    
    # Filter out movies that don't meet the minimum rating count
    trending_df = df[df['rating_count'] >= m].copy()
    
    def bayesian_average(row):
        v = row['rating_count']
        R = row['rating_mean']
        return (v / (v + m) * R) + (m / (v + m) * C)
        
    trending_df['trending_score'] = trending_df.apply(bayesian_average, axis=1)
    
    # Sort by trending score descending
    trending_df = trending_df.sort_values('trending_score', ascending=False).reset_index(drop=True)
    return trending_df

def search_movies(movies_df: pd.DataFrame, title_query: str = "", genres: list = None, min_year: int = None, max_year: int = None) -> pd.DataFrame:
    """
    Multi-field search for movies.
    """
    result = movies_df.copy()
    
    if title_query:
        result = result[result['title'].str.contains(title_query, case=False, na=False)]
        
    if genres and len(genres) > 0:
        # User might select multiple genres. We find movies that contain AT LEAST ONE of the selected genres.
        # Constructing a regex pattern like 'Action|Comedy'
        pattern = "|".join(genres)
        result = result[result['genres'].str.contains(pattern, case=False, na=False)]
        
    if min_year is not None:
        result = result[result['release_year'] >= min_year]
        
    if max_year is not None:
        result = result[result['release_year'] <= max_year]
        
    return result.sort_values('rating_count', ascending=False).reset_index(drop=True)
