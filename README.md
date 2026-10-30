# 🎬 CineMatch – Movie Recommendation System

CineMatch is a content-based Movie Recommendation System built using Natural Language Processing (NLP) and Machine Learning. It recommends movies similar to a user's selected movie based on textual metadata.

## 🚀 Live Demo

[Open CineMatch]([🚀 Open CineMatch](https://movierecommendationsystem-mz8ucrgkf2vp7djsdvzovi.streamlit.app/))

## ✨ Features

- Search and select movies from the dataset.
- Get similar movie recommendations.
- Content-based recommendation using TF-IDF.
- Cosine Similarity to measure movie similarity.
- Movie posters, ratings, release dates and overviews using TMDB API.
- Interactive web application built with Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- TMDB API

## ⚙️ How It Works

1. Movie metadata is loaded and preprocessed.
2. TF-IDF converts textual movie information into numerical vectors.
3. Cosine Similarity calculates similarity between movies.
4. The system ranks movies based on similarity scores.
5. Similar movies are displayed through the Streamlit application.
6. TMDB API provides movie information and posters.

## 📂 Project Structure

```text
movie_recommendation_system/
│
├── data/
│   └── movies_metadata.csv
│
├── models/
│   ├── df.pkl
│   ├── indices.pkl
│   ├── tfidf.pkl
│   └── tfidf_matrix.pkl
│
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd movie_recommendation_system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your TMDB API Read Access Token in:

`.streamlit/secrets.toml`

```toml
TMDB_TOKEN = "YOUR_TMDB_READ_ACCESS_TOKEN"
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 Project Objective

The objective of CineMatch is to explore how NLP and Machine Learning can be used to build a content-based recommendation system that helps users discover movies based on their interests.

## 👨‍💻 Author

**Atharv Gupta**

AIML Student | LNCT Bhopal

[GitHub](https://github.com/Atharv021205)  
[LinkedIn](https://www.linkedin.com/in/atharv-gupta-a9898630a/)

## 🙌 Acknowledgements

- TMDB API for movie information and images.
- Scikit-learn for machine learning tools.
- Streamlit for application deployment.

This product uses the TMDB API but is not endorsed or certified by TMDB.