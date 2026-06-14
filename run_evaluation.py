"""
run_evaluation.py
=================
Offline Evaluation Module for CineSuggest (Phase 14).
Performs an 80/20 train/test split on the ratings data,
trains an SVD matrix factorization model, and computes
Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("movie_rec.evaluation")

def run_evaluation(test_size: float = 0.2, k: int = 50):
    logger.info("=== Starting Offline Evaluation ===")
    
    ratings_path = PROCESSED_DATA_DIR / "processed_ratings.csv"
    if not ratings_path.exists():
        logger.error("Ratings file not found. Run data preparation first.")
        return
        
    ratings_df = pd.read_csv(ratings_path)
    logger.info(f"Loaded {len(ratings_df)} ratings.")
    
    # 80/20 Train Test Split
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    logger.info(f"Split data into {len(train_df)} training ratings and {len(test_df)} testing ratings.")
    
    # Create User-Item Matrix from Train
    unique_users = sorted(ratings_df['userId'].unique())
    unique_movies = sorted(ratings_df['movieId'].unique())
    
    user_to_row_idx = {user: idx for idx, user in enumerate(unique_users)}
    movie_to_col_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
    
    row_indices = train_df['userId'].map(user_to_row_idx).astype(int)
    col_indices = train_df['movieId'].map(movie_to_col_idx).astype(int)
    
    train_matrix = csr_matrix(
        (train_df['rating'], (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_movies))
    ).asfptype()
    
    # Calculate means
    user_means = np.array(train_matrix.mean(axis=1)).flatten()
    
    # Center
    train_centered = train_matrix.copy()
    train_centered.data -= np.take(user_means, train_centered.nonzero()[0])
    
    # SVD
    logger.info(f"Training SVD with k={k}...")
    k_actual = min(k, min(train_matrix.shape) - 1)
    U, sigma, Vt = svds(train_centered, k=k_actual)
    
    sigma_diag = np.diag(sigma)
    preds = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)
    
    # Calculate Error
    logger.info("Calculating metrics on test set...")
    
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        u = row['userId']
        m = row['movieId']
        r = row['rating']
        
        # If user/movie in our mapped index
        if u in user_to_row_idx and m in movie_to_col_idx:
            u_idx = user_to_row_idx[u]
            m_idx = movie_to_col_idx[m]
            pred_r = preds[u_idx, m_idx]
            
            # Clip predictions to [0.5, 5.0]
            pred_r = max(0.5, min(5.0, pred_r))
            
            y_true.append(r)
            y_pred.append(pred_r)
            
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print("\n=======================================================")
    print("                EVALUATION METRICS                     ")
    print("=======================================================")
    print(f"  Test Set Size : {len(test_df):,} ratings")
    print(f"  SVD k-factors : {k}")
    print("-------------------------------------------------------")
    print(f"  RMSE          : {rmse:.4f} (Root Mean Squared Error)")
    print(f"  MAE           : {mae:.4f} (Mean Absolute Error)")
    print("=======================================================\n")
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
