import pickle
import pandas as pd

# Load the saved model
with open('movie_recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

new_features = model_data['new_features']
cv = model_data['vectorizer']
cs = model_data['similarity_matrix']
data = model_data['data']

# Recommendation by movie title
def recommend_by_movie(movie):
    try:
        movie_index = new_features[new_features["title"]==movie].index[0]
        distances = cs[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]
        
        print(f"\nMovies similar to '{movie}':")
        for i, item in enumerate(movie_list, 1):
            print(f"{i}. {new_features.iloc[item[0]].title}")
    except IndexError:
        print(f"Movie '{movie}' not found in dataset!")

# Recommendation by actor
def recommend_by_actor(actor_name):
    actor_name = actor_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(actor_name, case=False, regex=False)].index.tolist()
    
    if not movie_indices:
        print(f"No movies found for actor '{actor_name}'")
        return
    
    print(f"\nTop 10 movies featuring this actor:")
    for i, idx in enumerate(movie_indices[:10], 1):
        print(f"{i}. {new_features.iloc[idx].title}")

# Recommendation by director
def recommend_by_director(director_name):
    director_name = director_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(director_name, case=False, regex=False)].index.tolist()
    
    if not movie_indices:
        print(f"No movies found for director '{director_name}'")
        return
    
    print(f"\nTop 10 movies by this director:")
    for i, idx in enumerate(movie_indices[:10], 1):
        print(f"{i}. {new_features.iloc[idx].title}")

# Recommendation by genre
def recommend_by_genre(genre_name):
    genre_name = genre_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(genre_name, case=False, regex=False)].index.tolist()
    
    if not movie_indices:
        print(f"No movies found for genre '{genre_name}'")
        return
    
    print(f"\nTop 10 {genre_name.title()} movies:")
    for i, idx in enumerate(movie_indices[:10], 1):
        print(f"{i}. {new_features.iloc[idx].title}")

# Recommendation by timeline
def recommend_by_timeline(start_year, end_year):
    timeline_movies = data[(data['release_date'] >= f'{start_year}-01-01') & (data['release_date'] <= f'{end_year}-12-31')]
    
    if timeline_movies.empty:
        print(f"No movies found between {start_year}-{end_year}")
        return
    
    print(f"\nTop 10 movies from {start_year}-{end_year}:")
    for i, movie in enumerate(timeline_movies['title'].head(10), 1):
        print(f"{i}. {movie}")


# Example usage:
if __name__ == "__main__":
    # Test all recommendation types
    recommend_by_movie("Avatar")
    recommend_by_actor("Robert Downey Jr")
    recommend_by_director("Christopher Nolan")
    recommend_by_genre("Action")
    recommend_by_timeline(2010, 2015)
