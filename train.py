import pymongo
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

def get_recommendations(id):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

data = pd.read_csv("data_test.csv")
id = '62b6d6e1a341c48fa5a3ffd4'

data['sum'] = data['title'] + data['bodyFull'] + data['keyWords'] + data['body']
data['sum'] = data['sum'].astype(str).str.replace(r"W","")
data['sum'] = data['sum'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['sum'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['_id']).drop_duplicates()

print(get_recommendations(id))
print(data[data['_id'] == id]['title'])

