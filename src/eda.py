import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import sys
from collections import Counter
from src.config import PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.utils.logger import get_logger

logger = get_logger("movie_rec.eda")

# Set visualization style settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.figsize": (10, 6)
})

# Custom color palette (matching CineSuggest aesthetics: pink/purple/violet)
PRIMARY_COLOR = "#ff007f"    # CineSuggest pink
SECONDARY_COLOR = "#7f00ff"  # CineSuggest purple
ACCENT_COLOR = "#89b4fa"     # Blue accent

def load_data():
    """
    Loads preprocessed datasets from data/processed/
    """
    movies_path = PROCESSED_DATA_DIR / "processed_movies.csv"
    ratings_path = PROCESSED_DATA_DIR / "processed_ratings.csv"
    
    if not movies_path.exists() or not ratings_path.exists():
        logger.error("Processed datasets not found! Run data preparation pipeline first.")
        raise FileNotFoundError("Processed movies/ratings files missing.")
        
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df

def plot_rating_distribution(ratings_df, save_path):
    """
    Plots the frequency distribution of movie ratings.
    """
    logger.info("Plotting rating distribution...")
    plt.figure(figsize=(9, 5.5))
    
    # Calculate counts and percentages
    ax = sns.countplot(
        x="rating", 
        data=ratings_df, 
        color=PRIMARY_COLOR, 
        edgecolor="black", 
        linewidth=0.7
    )
    
    # Add counts on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2., height),
            ha="center", va="bottom", fontsize=9, xytext=(0, 3),
            textcoords="offset points"
        )
        
    plt.title("Frequency Distribution of Movie Ratings (MovieLens 100k)", pad=15)
    plt.xlabel("Rating Score")
    plt.ylabel("Rating Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved rating distribution to {save_path.name}")

def plot_user_activity(ratings_df, save_path):
    """
    Plots user rating counts (log-scale) to show active user long-tails.
    """
    logger.info("Plotting user activity distribution...")
    plt.figure(figsize=(9, 5.5))
    
    user_counts = ratings_df["userId"].value_counts()
    
    sns.histplot(
        user_counts, 
        kde=True, 
        color=SECONDARY_COLOR, 
        bins=30,
        edgecolor="black",
        linewidth=0.7
    )
    
    plt.title("Distribution of User Activity Levels (Ratings per User)", pad=15)
    plt.xlabel("Number of Ratings Submitted by User")
    plt.ylabel("Frequency (User Count)")
    
    # Highlight statistical milestones
    median_val = user_counts.median()
    plt.axvline(median_val, color="red", linestyle="--", alpha=0.8, label=f"Median Ratings: {int(median_val)}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved user activity distribution to {save_path.name}")

def plot_genre_distribution(movies_df, save_path):
    """
    Plots overall genre frequency.
    """
    logger.info("Plotting genre frequency distribution...")
    plt.figure(figsize=(10, 6.5))
    
    # Explode genre lists
    # genres format is: "Action|Adventure|Sci-Fi"
    genres_series = movies_df["genres"].dropna().str.split("|").explode()
    genres_counts = genres_series[genres_series != "(no genres listed)"].value_counts()
    
    sns.barplot(
        x=genres_counts.values, 
        y=genres_counts.index, 
        hue=genres_counts.index,
        palette="viridis",
        legend=False
    )
    
    plt.title("Frequency Count of Movies by Genre", pad=15)
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre Name")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved genre distribution to {save_path.name}")

def plot_movies_by_year(movies_df, save_path):
    """
    Plots the count of movies released each year.
    """
    logger.info("Plotting movies released over time...")
    plt.figure(figsize=(10, 5.5))
    
    year_counts = movies_df["release_year"].dropna().astype(int).value_counts().sort_index()
    
    # Line chart of counts
    plt.plot(year_counts.index, year_counts.values, color=PRIMARY_COLOR, linewidth=2)
    plt.fill_between(year_counts.index, year_counts.values, color=PRIMARY_COLOR, alpha=0.2)
    
    plt.title("Trends in Movie Releases Over Time (1900 - 2018)", pad=15)
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies Released")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved movies by year plot to {save_path.name}")

def plot_rating_count_vs_mean(movies_df, save_path):
    """
    Plots ratings average against count to demonstrate popularity correlation.
    """
    logger.info("Plotting ratings count vs rating mean...")
    
    # We only care about movies that actually have ratings
    rated_movies = movies_df[movies_df["rating_count"] > 0]
    
    # Jointplot with scatter and distributions
    g = sns.jointplot(
        x="rating_count", 
        y="rating_mean", 
        data=rated_movies,
        kind="hex", 
        color=SECONDARY_COLOR,
        height=6.5,
        gridsize=25,
        marginal_kws=dict(bins=25, fill=True)
    )
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Correlation: Movie Rating Volume vs. Average Mean Score", fontsize=13)
    g.ax_joint.set_xlabel("Number of Ratings (Popularity)")
    g.ax_joint.set_ylabel("Average Rating (Quality)")
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved rating count vs mean correlation to {save_path.name}")

def run_full_eda():
    """
    Runs all plotters and exports all figures to the outputs folder.
    """
    logger.info("=== Starting Exploratory Data Analysis ===")
    
    # Load
    movies, ratings = load_data()
    
    # Ensure output folder exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Execute
    plot_rating_distribution(ratings, OUTPUTS_DIR / "rating_distribution.png")
    plot_user_activity(ratings, OUTPUTS_DIR / "user_activity.png")
    plot_genre_distribution(movies, OUTPUTS_DIR / "genre_distribution.png")
    plot_movies_by_year(movies, OUTPUTS_DIR / "movies_by_year.png")
    plot_rating_count_vs_mean(movies, OUTPUTS_DIR / "rating_count_vs_mean.png")
    
    logger.info("=== EDA Pipeline Completed Successfully ===")

if __name__ == "__main__":
    try:
        run_full_eda()
    except Exception as e:
        logger.error(f"EDA Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)
