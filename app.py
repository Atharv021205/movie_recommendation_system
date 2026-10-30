
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.subtitle {
    font-size: 20px;
    color: #bbbbbb;
}

.movie-card {
    background-color: #1c1c24;
    padding: 12px;
    border-radius: 12px;
    min-height: 100px;
}

.movie-name {
    font-size: 17px;
    font-weight: bold;
    color: white;
}

.stButton > button {
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TMDB API CONFIGURATION
# ==========================================

try:
    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]
except Exception:
    st.error(
        "TMDB token not found. Check your .streamlit/secrets.toml file."
    )
    st.stop()

if not TMDB_TOKEN.strip():
    st.error("TMDB token is empty. Please add your API Read Access Token.")
    st.stop()

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_TOKEN}"
}


# ==========================================
# 4. TMDB MOVIE SEARCH
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_details(title):

    try:
        response = requests.get(
            f"{BASE_URL}/search/movie",
            headers=HEADERS,
            params={
                "query": title,
                "include_adult": False
            },
            timeout=20
        )

        if response.status_code != 200:
            return {
                "error": f"TMDB API returned status {response.status_code}"
            }

        results = response.json().get("results", [])

        if not results:
            return None

        # Prefer exact title match
        for movie in results:
            if movie.get("title", "").casefold() == title.casefold():
                return movie

        # Otherwise return the first search result
        return results[0]

    except requests.RequestException as error:
        return {"error": str(error)}


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_data():

    with open("models/df.pkl", "rb") as file:
        df = pickle.load(file)

    with open("models/tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)

    with open("models/indices.pkl", "rb") as file:
        indices = pickle.load(file)

    return df, tfidf_matrix, indices


try:
    df, tfidf_matrix, indices = load_data()

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()


# ==========================================
# 6. RECOMMENDATION FUNCTION
# ==========================================

def recommend(title, n=10):

    if title not in indices:
        return df.iloc[0:0]

    idx = int(indices[title])

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommended_indices = [
        i for i in sorted_indices if i != idx
    ][:n]

    return df.iloc[recommended_indices]


# ==========================================
# 7. POSTER DISPLAY FUNCTION
# ==========================================

def display_poster(movie):

    poster_path = movie.get("poster_path")

    if poster_path:
        poster_url = IMAGE_URL + poster_path
        st.image(poster_url, use_container_width=True)
    else:
        st.info("Poster unavailable")


# ==========================================
# 8. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover your next favourite movie</div>',
    unsafe_allow_html=True
)

st.write(
    "Find movies similar to your favourite movies using Machine Learning."
)

st.divider()


# ==========================================
# 9. MOVIE SELECTION
# ==========================================

movie_list = sorted(
    df["title"].dropna().astype(str).unique()
)

selected_movie = st.selectbox(
    "🔍 Search or select a movie",
    movie_list
)

if st.button("🎬 Find Similar Movies", type="primary"):

    st.session_state["selected_movie"] = selected_movie


# ==========================================
# 10. DISPLAY SELECTED MOVIE
# ==========================================

if "selected_movie" in st.session_state:

    selected_movie = st.session_state["selected_movie"]

    st.divider()

    st.header(f"🎥 {selected_movie}")

    with st.spinner("Fetching movie details from TMDB..."):

        movie_details = get_movie_details(selected_movie)

    if movie_details and "error" in movie_details:

        st.error(movie_details["error"])
        st.info(
            "Check your API Read Access Token and internet connection."
        )

    elif movie_details:

        col1, col2 = st.columns([1, 3])

        with col1:
            display_poster(movie_details)

        with col2:

            rating = movie_details.get("vote_average", 0)
            release_date = movie_details.get("release_date") or "Unknown"

            st.subheader(movie_details.get("title", selected_movie))

            st.markdown(f"⭐ **Rating:** {rating:.1f}/10")
            st.markdown(f"📅 **Release Date:** {release_date}")

            st.markdown("### Overview")

            st.write(
                movie_details.get("overview")
                or "Overview unavailable."
            )

    else:

        st.warning(
            "Movie not found on TMDB. Recommendations will still be shown."
        )


    # ======================================
    # 11. RECOMMENDED MOVIES
    # ======================================

    st.divider()

    st.header("🍿 Recommended Movies")

    recommendations = recommend(selected_movie, 10)

    if recommendations.empty:

        st.warning("No recommendations found.")

    else:

        for start in range(0, len(recommendations), 5):

            movie_columns = st.columns(5)

            batch = recommendations.iloc[start:start + 5]

            for column, (_, movie) in zip(movie_columns, batch.iterrows()):

                title = str(movie["title"])

                with column:

                    with st.spinner(f"Loading {title}..."):

                        details = get_movie_details(title)

                    if details and "error" in details:

                        st.caption(title)
                        st.caption("TMDB unavailable")

                    elif details:

                        display_poster(details)

                        st.markdown(
                            f"**{details.get('title', title)}**"
                        )

                        rating = details.get("vote_average", 0)
                        release = details.get("release_date") or "Unknown"

                        st.caption(f"⭐ {rating:.1f}/10")
                        st.caption(f"📅 {release[:4]}")

                    else:

                        st.markdown(f"**{title}**")
                        st.caption("Poster unavailable")


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)

st.caption(
    "CineMatch | Movie Recommendation System using NLP and Machine Learning"
)



