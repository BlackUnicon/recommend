import pymongo
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# def standardize_data(row):
#     row = re.sub(r"[\.,\?]+$-", "", row)
#     row = row.replace(",", " ").replace(".", " ") \
#         .replace(";", " ").replace("“", " ") \
#         .replace(":", " ").replace("”", " ") \
#         .replace('"', " ").replace("'", " ") \
#         .replace("!", " ").replace("?", " ") \
#         .replace("-", " ").replace("?", " ")
#     row = row.strip().lower()
#     return row

# uri = ""
# client = MongoClient(uri)
# db = client[""]
# collection = db[""]

# results = collection.find({})

def get_recommendations(id):
    # Get the index of the movie that matches the title
    idx = indices[id]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]

data = pd.read_csv("data_test.csv")

data['body'] = data['body'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['body'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['_id']).drop_duplicates()
print(get_recommendations('62b6c8b2023830051e30e4c6'))
print(data[data['_id'] == '62b6c8b2023830051e30e4c6']['title'])

