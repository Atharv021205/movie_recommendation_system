

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests

from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}
.movie-title {
    font-size: 18px;
    font-weight: bold;
}
.movie-info {
    color: #bbbbbb;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TMDB API ----------------

TMDB_TOKEN = st.secrets["TMDB_TOKEN"]

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_TOKEN}"
}

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600)
def get_movie_details(title):

    try:
        response = requests.get(
            f"{BASE_URL}/search/movie",
            headers=HEADERS,
            params={"query": title},
            timeout=15
        )

        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            return results[0]

    except requests.RequestException:
        return None

    return None


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_data():

    with open("models/df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("models/indices.pkl", "rb") as f:
        indices = pickle.load(f)

    return df, tfidf_matrix, indices


df, tfidf_matrix, indices = load_data()


# ---------------- RECOMMENDATION ----------------

def recommend(title, n=10):

    if title not in indices:
        return df.iloc[0:0]

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


# ---------------- HEADER ----------------

st.title("🎬 CineMatch")

st.markdown(
    "### Discover your next favourite movie"
)

st.write(
    "Find movies similar to your favourite movies using Machine Learning."
)

st.divider()


# ---------------- MOVIE SEARCH ----------------

movie_list = sorted(
    df["title"].dropna().astype(str).unique()
)

selected_movie = st.selectbox(
    "🔍 Search or select a movie",
    movie_list
)

if st.button("Find Similar Movies", type="primary"):

    st.session_state["selected_movie"] = selected_movie


# ---------------- DISPLAY RESULTS ----------------

if "selected_movie" in st.session_state:

    selected_movie = st.session_state["selected_movie"]

    st.divider()

    st.header(f"🎥 {selected_movie}")

    details = get_movie_details(selected_movie)

    if details:

        col1, col2 = st.columns([1, 3])

        with col1:

            poster = details.get("poster_path")

            if poster:
                st.image(IMAGE_URL + poster)
            else:
                st.info("Poster unavailable")

        with col2:

            rating = details.get("vote_average", 0)
            release = details.get("release_date", "Unknown")

            st.markdown(f"⭐ **Rating:** {rating}/10")
            st.markdown(f"📅 **Release Date:** {release}")

            st.markdown("### Overview")

            st.write(
                details.get(
                    "overview",
                    "Overview unavailable."
                )
            )

    else:
        st.warning(
            "TMDB details unavailable for this movie. Showing recommendations anyway."
        )

    st.divider()

    st.header("🍿 Recommended Movies")

    recommendations = recommend(selected_movie, 10)

    columns = st.columns(5)

    for i, (_, movie) in enumerate(
        recommendations.iterrows()
    ):

        movie_details = get_movie_details(
            str(movie["title"])
        )

        with columns[i % 5]:

            if movie_details:

                poster = movie_details.get("poster_path")

                if poster:
                    st.image(
                        IMAGE_URL + poster,
                        use_container_width=True
                    )
                else:
                    st.info("Poster unavailable")

                st.markdown(
                    f"**{movie_details.get('title', movie['title'])}**"
                )

                rating = movie_details.get("vote_average", 0)

                release = movie_details.get(
                    "release_date", "Unknown"
                )

                st.caption(f"⭐ {rating:.1f}/10")

                if release:
                    st.caption(f"📅 {release[:4]}")

            else:

                st.markdown(f"**{movie['title']}**")
                st.caption("TMDB details unavailable")

st.divider()

st.caption(
    "Movie data and images provided by TMDB. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)
