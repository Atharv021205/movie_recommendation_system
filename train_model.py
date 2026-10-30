
import pandas as pd
import numpy as np
import ast
import re
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
df = pd.read_csv("data/movies_metadata.csv", low_memory=False)

print("Dataset loaded:", df.shape)

# 2. Select required columns
df = df[
    ['id', 'title', 'overview', 'genres',
     'poster_path', 'release_date', 'vote_average']
].copy()

# 3. Remove missing titles and duplicate movies
df = df.dropna(subset=['title'])
df = df.drop_duplicates(subset=['id'])

# 4. Handle missing values
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')

# 5. Extract genre names
def extract_genres(value):
    try:
        genres = ast.literal_eval(value)

        return ' '.join(
            genre['name']
            for genre in genres
            if isinstance(genre, dict) and 'name' in genre
        )

    except (ValueError, SyntaxError, TypeError):
        return ''

df['genres'] = df['genres'].apply(extract_genres)

# 6. Combine overview and genres
df['tags'] = df['overview'] + ' ' + df['genres']

# 7. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['tags'] = df['tags'].apply(clean_text)

# 8. Reset index
df = df.reset_index(drop=True)

# 9. Apply TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(df['tags'])

print("TF-IDF Shape:", tfidf_matrix.shape)

# 10. Create movie title index
indices = pd.Series(
    df.index,
    index=df['title']
).drop_duplicates()

# 11. Save model files
os.makedirs('models', exist_ok=True)

df.to_pickle('models/df.pkl')

with open('models/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open('models/indices.pkl', 'wb') as f:
    pickle.dump(indices, f)

with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model training completed successfully!")
