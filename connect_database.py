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

    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

data = pd.read_csv("data_test.csv")

data['bodyFull'] = data['bodyFull'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['bodyFull'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['_id']).drop_duplicates()
print(get_recommendations('62b6c83b023830051e30e449'))
print(data[data['_id'] == '62b6c83b023830051e30e449']['title'])

