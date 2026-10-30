
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved files
df = pickle.load(open('models/df.pkl', 'rb'))
tfidf_matrix = pickle.load(open('models/tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('models/indices.pkl', 'rb'))

# Recommendation function
def recommend(title, n=10):

    if title not in indices:
        return ['Movie not found']

    idx = indices[title]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = np.argsort(similarity)[::-1]

    similar_indices = [
        i for i in similar_indices if i != idx
    ][:n]

    return df['title'].iloc[similar_indices].tolist()


# Test
movie_name = "Toy Story"

print("Movies similar to:", movie_name)

for movie in recommend(movie_name):
    print(movie)

