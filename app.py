# Using Flask to create a web application for movie recommendations

from flask import Flask, render_template, request, jsonify
from pickle import load
import requests
import pandas as pd
import re

app = Flask(__name__)

# Load the movie recommendation model
with open("movieModel.pkl", "rb") as f:
    model_data = load(f)

new_features = model_data['new_features']
cs = model_data['cs']
data = model_data['data']

# Precompute a normalized title column to make title matching
# more robust to punctuation, spacing, and case differences
def _normalize_title(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    # Replace any non-alphanumeric character with a space,
    # then collapse multiple spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

new_features["normalized_title"] = new_features["title"].apply(_normalize_title)

# OMDB API Configuration (get your key from https://www.omdbapi.com/apikey.aspx)
OMDB_API_KEY =  "6ac1a66a"
OMDB_API_URL = "http://www.omdbapi.com/"


def get_movie_poster(movie_title):
    """Fetch movie poster from OMDB API using movie title.

    Falls back to a placeholder image if poster is not available
    or the API key is missing/invalid.
    """
    # If the key is not set, immediately return placeholder
    if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_OMDB_API_KEY":
        return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"

    try:
        params = {
            'apikey': OMDB_API_KEY,
            't': movie_title,
        }
        response = requests.get(OMDB_API_URL, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True':
                poster_url = data.get('Poster')
                if poster_url and poster_url != 'N/A':
                    return poster_url
    except Exception as e:
        print(f"Error fetching poster for {movie_title}: {e}")

    # Return placeholder image if poster not found or any error
    return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"

def recommend_movies(movie_title):
    """Get movie recommendations based on title using cosine similarity."""
    try:
        # Normalize the input title to match against precomputed normalized titles
        normalized_query = _normalize_title(movie_title)

        # Find the movie index using normalized titles so that queries like
        # "Mission Impossible rogue nation" still match
        movie_index = new_features[new_features["normalized_title"] == normalized_query].index
        
        if len(movie_index) == 0:
            return None
        
        movie_index = movie_index[0]
        distances = cs[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        
        recommendations = []
        for i in movie_list:
            movie_data = new_features.iloc[i[0]]
            movie_info = {
                'title': movie_data.title,
                'poster': get_movie_poster(movie_data.title)
            }
            recommendations.append(movie_info)
        
        return recommendations
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return None


def recommend_by_actor(actor_name):
    """Return movies where the given actor appears (top 10)."""
    try:
        actor_key = actor_name.replace(" ", "").lower()
        movie_indices = new_features[
            new_features["tags"].str.contains(actor_key, case=False, regex=False)
        ].index.tolist()

        movies = []
        for idx in movie_indices[:10]:
            title = new_features.iloc[idx].title
            movies.append({
                "title": title,
                "poster": get_movie_poster(title)
            })
        return movies
    except Exception as e:
        print(f"Error in recommend_by_actor: {e}")
        return None


def recommend_by_director(director_name):
    """Return movies directed by the given director (top 10)."""
    try:
        director_key = director_name.replace(" ", "").lower()
        movie_indices = new_features[
            new_features["tags"].str.contains(director_key, case=False, regex=False)
        ].index.tolist()

        movies = []
        for idx in movie_indices[:10]:
            title = new_features.iloc[idx].title
            movies.append({
                "title": title,
                "poster": get_movie_poster(title)
            })
        return movies
    except Exception as e:
        print(f"Error in recommend_by_director: {e}")
        return None


def recommend_by_genre(genre_name):
    """Return movies for a given genre keyword (top 10)."""
    try:
        genre_key = genre_name.replace(" ", "").lower()
        movie_indices = new_features[
            new_features["tags"].str.contains(genre_key, case=False, regex=False)
        ].index.tolist()

        movies = []
        for idx in movie_indices[:10]:
            title = new_features.iloc[idx].title
            movies.append({
                "title": title,
                "poster": get_movie_poster(title)
            })
        return movies
    except Exception as e:
        print(f"Error in recommend_by_genre: {e}")
        return None


def recommend_by_timeline(start_year: int, end_year: int):
    """Return movies released between start_year and end_year (top 10)."""
    try:
        start = f"{start_year}-01-01"
        end = f"{end_year}-12-31"
        timeline_movies = data[(data['release_date'] >= start) & (data['release_date'] <= end)]

        movies = []
        for title in timeline_movies['title'].head(10):
            movies.append({
                "title": title,
                "poster": get_movie_poster(title)
            })
        return movies
    except Exception as e:
        print(f"Error in recommend_by_timeline: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
def search():
    """Handle movie search and return recommendations"""
    try:
        data = request.get_json()
        movie_name = data.get('movie_name', '').strip()
        search_type = data.get('search_type', 'title').strip().lower()
        
        if not movie_name:
            return jsonify({'error': 'Please enter a value to search'}), 400

        # Route to appropriate recommender based on search_type
        if search_type == 'title':
            recommendations = recommend_movies(movie_name)
            error_message = f'Movie "{movie_name}" not found in our database'

        elif search_type == 'actor':
            recommendations = recommend_by_actor(movie_name)
            error_message = f'No movies found for actor "{movie_name}"'

        elif search_type == 'director':
            recommendations = recommend_by_director(movie_name)
            error_message = f'No movies found for director "{movie_name}"'

        elif search_type == 'genre':
            recommendations = recommend_by_genre(movie_name)
            error_message = f'No movies found for genre "{movie_name}"'

        elif search_type == 'timeline':
            # Expect the user to enter something like "2000-2010" or "2000 to 2010"
            years = re.findall(r"(19\d{2}|20\d{2})", movie_name)
            if len(years) < 2:
                return jsonify({'error': 'Please enter a valid year range, e.g. 2000-2010'}), 400
            start_year, end_year = sorted(map(int, years[:2]))
            recommendations = recommend_by_timeline(start_year, end_year)
            error_message = f'No movies found between {start_year} and {end_year}'
            movie_name = f"{start_year}-{end_year}"

        else:
            return jsonify({'error': 'Invalid search type'}), 400

        if not recommendations:
            return jsonify({'error': error_message}), 404

        return jsonify({'recommendations': recommendations, 'search_query': movie_name, 'search_type': search_type})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)