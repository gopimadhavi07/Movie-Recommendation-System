"""
src/moods.py
============
Pre-defined mappings from human moods to MovieLens genres.
"""

MOOD_MAPPINGS = {
    "Cozy / Relaxed": ["Comedy", "Romance", "Animation", "Children"],
    "Adrenaline / Exciting": ["Action", "Thriller", "Sci-Fi", "Adventure"],
    "Thought-Provoking / Deep": ["Drama", "Mystery", "Documentary"],
    "Spooky / Scary": ["Horror", "Thriller", "Mystery"],
    "Escapism / Magical": ["Fantasy", "Sci-Fi", "Adventure", "Animation"],
    "Heartwarming": ["Romance", "Drama", "Comedy", "Children"],
    "Dark / Gritty": ["Crime", "Film-Noir", "Thriller", "Drama"]
}

def get_genres_for_mood(mood_name: str) -> list:
    """Returns the list of genres associated with a given mood, or empty list if not found."""
    return MOOD_MAPPINGS.get(mood_name, [])

def get_all_moods() -> list:
    """Returns a list of all available moods."""
    return list(MOOD_MAPPINGS.keys())
