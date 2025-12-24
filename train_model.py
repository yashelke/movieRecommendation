import pandas as pd
import numpy as np
import ast 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from  string import punctuation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pickle import dump
from sklearn.naive_bayes import MultinomialNB



movie_data = pd.read_csv("tmdb_5000_movies.csv")
credits_data = pd.read_csv("tmdb_5000_credits.csv")


# Data Preprocessing :-

# print(movie_data.head(10).to_string())
# print(credits_data.head(10).to_string())



# 20 features in movie_data
# 4 features in credits_data

# 23 features in combined data
data = movie_data.merge(credits_data,on="title")

# print(data.head(5).to_string())
# print(data.shape)
# print(data.head(1).to_string())

# Selecting relevant features
# genres, id, keywords, title, overview, cast, crew


# reduced to 7 features
features =data[["movie_id","title","overview","genres","keywords","cast","crew"]]
# print(features)

#  overview has 3 null values
# print(features.isnull().sum())

# drop the null values

features = features.dropna(inplace=False)

# print(features.isnull().sum())
# print(features.duplicated().sum())

# formatting the columns of the features

# print(features.iloc[0].genres)


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L



converted_genres = features["genres"].apply(convert)
converted_keywords = features["keywords"].apply(convert)


features["genres"] = converted_genres
features["keywords"] = converted_keywords

# print(features.iloc[0].genres)
# print(features.iloc[0].keywords)


# For cast column we will take only top 3 actors of each movie in the features dataset


def convert_cast(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
          L.append(i['name'])
          counter+=1
        else:
            break
    return L


converted_cast = features["cast"].apply(convert_cast)   

features["cast"] = converted_cast

# print(features.iloc[0].cast)

# For crew column we will take only director of each movie in the features dataset

def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L    


converted_crew = features["crew"].apply(fetch_director)

features["crew"] = converted_crew

# print(features["crew"])


# Now we have formatted all the columns of the features dataset
# print(features.head().to_string())



# print(features.iloc[0].overview)


converted_overview = features["overview"].apply(lambda x:x.split())

features["overview"] = converted_overview

# print(features.iloc[0].overview)

features["genres"] = features["genres"].apply(lambda x:[i.replace(" ","") for i in x])
features["keywords"] = features["keywords"].apply(lambda x:[i.replace(" ","") for i in x])
features["cast"] = features["cast"].apply(lambda x:[i.replace(" ","") for i in x])
features["crew"] = features["crew"].apply(lambda x:[i.replace(" ","") for i in x])

# print(features.iloc[0].genres)
# print(features.iloc[0].keywords)
# print(features.iloc[0].cast)
# print(features.iloc[0].crew)


features["tags"] = features["overview"] + features["genres"] + features["keywords"] + features["cast"] + features["crew"]

# print(features)

# We have reduced the 7 features to 3 features in this new_features dataset
new_features = features[["movie_id","title","tags"]]
# print(new_features.head())
# print(new_features.shape)
# print(new_features.iloc[0].tags)



# print(new_features.iloc[0].tags)

new_features = features[["movie_id", "title", "tags"]].copy()
new_features["tags"] = new_features["tags"].apply(lambda x: " ".join(x).lower())
# print(new_features.iloc[0].tags)

# Vectorization

cv = CountVectorizer(max_features=5000,stop_words="english")

vectors = cv.fit_transform(new_features["tags"])

new_features_features = pd.DataFrame(vectors.toarray(), columns = cv.get_feature_names_out())

# vectors = tv.fit_transform(clean_review)
# features = pd.DataFrame(vectors.toarray(), columns = tv.get_feature_names_out())

# print(new_features.iloc[0])

# ["love","loving","loved","lovable","loves"]  to stem words to a single word "love"


ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


new_features["tags"] = new_features["tags"].apply(stem)
# print(new_features.iloc[0].tags)

# print(new_features.iloc[0])

# Implemnenting cosine similarity

cs = cosine_similarity(vectors)
# print(cs.shape)


# print(cs)


# Recommendation System with title of the movie as input
def recommendation(movie):
    movie_index = new_features[new_features["title"]==movie].index[0]
    distances = cs[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]

    for i in movie_list:
        print(new_features.iloc[i[0]].title)
        
# recommendation("Batman Begins")

# Recommendation System with Actor as input

def recommendation_actor(actor_name):
    actor_name = actor_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(actor_name, case=False, regex=False)].index.tolist()
    
    # Get movies where the actor actually appears
    actor_movies = []
    for idx in movie_indices:
        actor_movies.append(new_features.iloc[idx].title)
    
    # Print top 10 movies the actor is in
    for i, movie in enumerate(actor_movies[:10], 1):
        print(f"{i}. {movie}")

# recommendation_actor("Daniel Craig")


# Recommendation System with Director as input
def recommendation_director(director_name):
    director_name = director_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(director_name, case=False, regex=False)].index.tolist()
    
    # Get movies where the director actually appears
    director_movies = []
    for idx in movie_indices:
        director_movies.append(new_features.iloc[idx].title)
    
    # Print top 10 movies the director is in
    for i, movie in enumerate(director_movies[:10], 1):
        print(f"{i}. {movie}")
        
# recommendation_director("Steven Spielberg")


# Genre as input

def recommendation_genre(genre_name):
    genre_name = genre_name.replace(" ","").lower()
    movie_indices = new_features[new_features["tags"].str.contains(genre_name, case=False, regex=False)].index.tolist()
    
    # Get movies where the genre actually appears
    genre_movies = []
    for idx in movie_indices:
        genre_movies.append(new_features.iloc[idx].title)
    
    # Print top 10 movies of the genre
    for i, movie in enumerate(genre_movies[:10], 1):
        print(f"{i}. {movie}")
        
# recommendation_genre("Adventure")


# timeline in the rnage like movies released in 2000-2010

def recommendation_timeline(start_year, end_year):
    timeline_movies = data[(data['release_date'] >= f'{start_year}-01-01') & (data['release_date'] <= f'{end_year}-12-31')]
    
    # Print top 10 movies released in the timeline
    for i, movie in enumerate(timeline_movies['title'].head(10), 1):
        print(f"{i}. {movie}")
    

# recommendation_timeline(1980, 1990)

# Save the recommendation system components
with open("movieModel.pkl", "wb") as f:
    dump({
        'new_features': new_features,
        'cs': cs,
        'data': data
    }, f)





    
        