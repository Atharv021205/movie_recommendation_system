
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load model
@st.cache_resource
def load_data():
    df = pickle.load(open("models/df.pkl", "rb"))
    tfidf_matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))
    indices = pickle.load(open("models/indices.pkl", "rb"))
    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_data()

# Recommendation function
def recommend(title, n=10):
    idx = indices[title]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = np.argsort(similarity)[::-1]
    similar_indices = [
        i for i in similar_indices if i != idx
    ][:n]

    return df.iloc[similar_indices]

# Website heading
st.title("🎬 CineMatch")
st.subheader("Discover movies you'll love!")

st.write("Find movies similar to your favourite movie.")

# Movie selection
movie_list = sorted(df["title"].dropna().unique())

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)

if st.button("Find Similar Movies"):
    st.subheader(f"Movies similar to {selected_movie}")

    results = recommend(selected_movie)

    columns = st.columns(5)

    for i, (_, movie) in enumerate(results.iterrows()):
        with columns[i % 5]:

            poster = movie.get("poster_path")

            if pd.notna(poster) and poster:
                st.image(
                    "https://image.tmdb.org/t/p/w500" + poster
                )
            else:
                st.write("Poster unavailable")

            st.markdown(f"**{movie['title']}**")

            st.caption(
                f"⭐ {movie['vote_average']} | "
                f"{movie['release_date']}"
            )
