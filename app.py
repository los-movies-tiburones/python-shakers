import os
from functools import lru_cache

import flask
from flask import jsonify
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import request

app = flask.Flask(__name__)
app.config["MONGO_URI"] = os.environ['MONGODB_URI']
mongo = PyMongo(app)

cursor = mongo.db.movie.find(
    {},
    {"genres": 1, "title": 1, "averageRating": 1}
)
movies = [i for i in cursor]
movies = pd.DataFrame(movies)
movies['genres'] = movies['genres'].fillna("").astype('str')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])


@lru_cache
def genre_recommendations(title, quantity_movies):
    titles = movies['title']
    indices = pd.Series(movies.index, index=movies['title'])
    idx = indices[title]
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix[idx])
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = list(filter(lambda x: x[0] != idx, sim_scores))
    sim_scores = sim_scores[:quantity_movies]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


@app.route('/movies/recommendations', methods=['GET'])
def home():
    movie = str(request.args.get('name'))
    recommendations = list(genre_recommendations(movie, 10).to_numpy())
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run()
