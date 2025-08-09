%%writefile app.py
import os
import re
import typing

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from justwatch import JustWatch  # Unofficial JustWatch wrapper

# ============ Setup Poster Fetching ============

jw = JustWatch(country='IN')  # You can change 'IN' to 'US' or any ISO country code
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"

def _normalize_justwatch_url(s: str) -> typing.Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith('//'):
        return 'https:' + s
    if s.startswith('/'):
        return 'https://images.justwatch.com' + s
    if s.startswith('http'):
        return s
    if re.fullmatch(r'\d+', s):
        return f'https://images.justwatch.com/poster/{s}/s592'
    return None

def _find_poster_in_obj(obj) -> typing.Optional[str]:
    if isinstance(obj, str):
        if '/poster/' in obj or 'images.justwatch.com' in obj:
            return obj
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_poster_in_obj(v)
            if found:
                return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_poster_in_obj(v)
            if found:
                return found
    return None

@st.cache_data(ttl=60 * 60 * 24)  # Cache for 24 hours
def fetch_poster_justwatch(title: str) -> str:
    try:
        results = jw.search_for_item(query=title, count=4)
        if not results:
            return PLACEHOLDER_POSTER
        item = results[0]
        # Try direct keys
        for key in ("poster", "poster_path", "poster_url", "thumbnail", "image"):
            if key in item and item[key]:
                url = _normalize_justwatch_url(item[key])
                if url:
                    return url
        # Try recursive search in the object
        found = _find_poster_in_obj(item)
        if found:
            url = _normalize_justwatch_url(found)
            if url:
                return url
        # Try fetching details by ID if available
        title_id = item.get("id") or item.get("title_id") or item.get("object_id")
        if title_id:
            details = jw.get_title(title_id)
            found = _find_poster_in_obj(details)
            if found:
                url = _normalize_justwatch_url(found)
                if url:
                    return url
    except Exception as e:
        print("JustWatch poster fetch error:", e)
    return PLACEHOLDER_POSTER

# ============ Load Assets ============

try:
    assets = joblib.load('recommender_assets.joblib')
    svd_model = assets['svd_model']
    movies_df = assets['movies_df']
    ratings_df = assets['ratings_df']
    genre_similarity_matrix = assets['genre_similarity_matrix']
    movie_id_to_index_map = assets['movie_id_to_index_map']
    index_to_movie_id_map = assets['index_to_movie_id_map']
    movie_ids_array = assets['movie_ids_array']
except FileNotFoundError:
    st.error("Saved assets file not found! Ensure 'recommender_assets.joblib' is in your repository.")
    st.stop()

# ============ Recommendation Logic ============

def fast_hybrid_recommendations(user_id, svd_model, ratings_df, alpha=0.5, n=10):
    try:
        inner_uid = svd_model.trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    all_movie_ids = [svd_model.trainset.to_raw_iid(iid) for iid in range(svd_model.trainset.n_items)]
    user_svd_scores = [svd_model.predict(user_id, mid).est for mid in all_movie_ids]

    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
    if len(liked_movies) == 0:
        liked_movies = user_ratings[user_ratings['rating'] >= 3]['movieId'].values

    if len(liked_movies) > 0:
        liked_indices = [movie_id_to_index_map.get(mid) for mid in liked_movies if mid in movie_id_to_index_map]
        if liked_indices:
            content_scores = genre_similarity_matrix[liked_indices].mean(axis=0)
            svd_df = pd.DataFrame({'movieId': all_movie_ids, 'svd_score': user_svd_scores})
            content_df = pd.DataFrame({'movieId': movie_ids_array, 'content_score': content_scores})
            scores_df = pd.merge(svd_df, content_df, on='movieId', how='inner')

            s_min, s_max = scores_df['svd_score'].min(), scores_df['svd_score'].max()
            c_min, c_max = scores_df['content_score'].min(), scores_df['content_score'].max()

            scores_df['svd_norm'] = ((scores_df['svd_score'] - s_min) / (s_max - s_min)) if (s_max > s_min) else 0
            scores_df['content_norm'] = ((scores_df['content_score'] - c_min) / (c_max - c_min)) if (c_max > c_min) else 0
            scores_df['hybrid_score'] = alpha * scores_df['svd_norm'] + (1 - alpha) * scores_df['content_norm']

            rated = user_ratings['movieId'].unique()
            scores_df = scores_df[~scores_df['movieId'].isin(rated)]
            top_n_df = scores_df.nlargest(n, 'hybrid_score')
            return top_n_df['movieId'].tolist()

    predictions = {mid: score for mid, score in zip(all_movie_ids, user_svd_scores)}
    rated_movies = user_ratings['movieId'].unique()
    unrated = {mid: score for mid, score in predictions.items() if mid not in rated_movies}
    top_n = sorted(unrated, key=unrated.get, reverse=True)[:n]
    return top_n

# ============ Streamlit App UI ============

st.set_page_config(layout="wide", page_title="Movie Recommender")

st.markdown("""
<style>
    .main { background-color: #141414; }
    .stApp { color: white; }
    h1, h2, h3, h4, h5, h6 { color: #E50914; }
    .st-eb { background-color: #222222; }
</style>
""", unsafe_allow_html=True)

st.title('🎬 Movie Recommender')

st.sidebar.header('Enter Your User ID')
user_id_input = st.sidebar.number_input('User ID (1 to 6040)', min_value=1, max_value=6040, value=12, step=1)

if st.sidebar.button('Get Recommendations'):
    with st.spinner('Finding movies you might like...'):
        rec_ids = fast_hybrid_recommendations(user_id=user_id_input, svd_model=svd_model,
                                              ratings_df=ratings_df, alpha=0.5, n=10)

        if rec_ids:
            st.subheader(f'Top 10 Recommendations for User {user_id_input}')
            rec_df = pd.DataFrame(rec_ids, columns=['movieId']).merge(movies_df, on='movieId')

            row1, row2 = st.columns(5), st.columns(5)
            for i, movie in rec_df.iterrows():
                col = row1[i] if i < 5 else row2[i - 5]
                with col:
                    poster_url = fetch_poster_justwatch(movie['title'])
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"<p style='text-align:center;color:white;'>{movie['title']}</p>",
                                unsafe_allow_html=True)
        else:
            st.error("Could not generate recommendations for this user.")
