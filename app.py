import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests # Needed for API calls
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================================
# LOAD SAVED ASSETS
# =====================================================================================
@st.cache_resource
def load_assets():
    try:
        # This function loads the saved model and data from the .joblib file
        return joblib.load('recommender_assets.joblib')
    except FileNotFoundError:
        st.error("Saved assets file not found! Ensure 'recommender_assets.joblib' is in your repository.")
        st.stop()

assets = load_assets()
svd_model = assets['svd_model']
movies_df = assets['movies_df']
ratings_df = assets['ratings_df']
genre_similarity_matrix = assets['genre_similarity_matrix']
movie_id_to_index_map = assets['movie_id_to_index_map']
index_to_movie_id_map = assets['index_to_movie_id_map']
movie_ids_array = assets['movie_ids_array']

# =====================================================================================
# MOVIE POSTER FETCHING FUNCTION (Updated for OMDb)
# =====================================================================================
@st.cache_data
def fetch_poster(movie_title):
    """Fetches a movie poster URL from The OMDb API."""
    # --- IMPORTANT: PASTE YOUR OMDb API KEY HERE ---
    api_key = "7711f131"
    # ----------------------------------------------

    if api_key == "7711f131":
        return "https://via.placeholder.com/500x750.png?text=Please+Add+API+Key"

    title_only = movie_title.split('(')[0].strip()
    
    url = f"http://www.omdbapi.com/?t={title_only}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get('Poster') and data['Poster'] != 'N/A':
            return data['Poster']
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Found" # Fallback image


# =====================================================================================
# RECOMMENDATION FUNCTION
# =====================================================================================
def fast_hybrid_recommendations(user_id, svd_model, ratings_df, alpha=0.5, n=10):
    try: inner_uid = svd_model.trainset.to_inner_uid(user_id)
    except ValueError: return []
    
    all_movie_ids = [svd_model.trainset.to_raw_iid(iid) for iid in range(svd_model.trainset.n_items)]
    user_svd_scores = [svd_model.predict(user_id, mid).est for mid in all_movie_ids]
    
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
    if len(liked_movies) == 0: liked_movies = user_ratings[user_ratings['rating'] >= 3]['movieId'].values
    
    if len(liked_movies) > 0:
        liked_indices = [movie_id_to_index_map.get(mid) for mid in liked_movies if mid in movie_id_to_index_map]
        if liked_indices:
            content_scores = genre_similarity_matrix[liked_indices].mean(axis=0)
            svd_df = pd.DataFrame({'movieId': all_movie_ids, 'svd_score': user_svd_scores})
            content_df = pd.DataFrame({'movieId': movie_ids_array, 'content_score': content_scores})
            scores_df = pd.merge(svd_df, content_df, on='movieId', how='inner')

            s_min, s_max = scores_df['svd_score'].min(), scores_df['svd_score'].max()
            c_min, c_max = scores_df['content_score'].min(), scores_df['content_score'].max()
            
            scores_df['svd_norm'] = (scores_df['svd_score'] - s_min) / (s_max - s_min) if (s_max - s_min) > 0 else 0
            scores_df['content_norm'] = (scores_df['content_score'] - c_min) / (c_max - c_min) if (c_max - c_min) > 0 else 0
            
            scores_df['hybrid_score'] = (alpha * scores_df['svd_norm']) + ((1 - alpha) * scores_df['content_norm'])
            
            rated_movies = user_ratings['movieId'].unique()
            scores_df = scores_df[~scores_df['movieId'].isin(rated_movies)]
            
            top_n_df = scores_df.nlargest(n, 'hybrid_score')
            return top_n_df['movieId'].tolist()

    # Fallback to pure SVD if content profile can't be built
    predictions = {mid: score for mid, score in zip(all_movie_ids, user_svd_scores)}
    rated_movies = user_ratings['movieId'].unique()
    unrated = {mid: score for mid, score in predictions.items() if mid not in rated_movies}
    top_n = sorted(unrated, key=unrated.get, reverse=True)[:n]
    return top_n

# =====================================================================================
# FINAL STREAMLIT APP UI (with Final Fixes)
# =====================================================================================

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- Custom CSS for Netflix-like theme ---
st.markdown("""
<style>
    .main { background-color: #141414; }
    .stApp { color: white; }
    h1, h2, h3, h4, h5, h6 { color: #E50914; }
    .st-eb { background-color: #222222; }
</style>
""", unsafe_allow_html=True)

st.title('🎬 Movie Recommender')

# --- User Input ---
st.sidebar.header('Enter Your User ID')
user_id_input = st.sidebar.number_input('User ID (1 to 6040)', min_value=1, max_value=6040, value=12, step=1)

# --- Generate Recommendations ---
if st.sidebar.button('Get Recommendations'):
    with st.spinner('Finding movies you might like...'):
        recommended_movie_ids = fast_hybrid_recommendations(
            user_id=user_id_input, svd_model=svd_model, ratings_df=ratings_df, alpha=0.5, n=10
        )
        
        if recommended_movie_ids:
            st.subheader(f'Top 10 Recommendations for User {user_id_input}')
            
            # Ensure the DataFrame is in the same order as the recommendations
            recommended_movies_df = pd.DataFrame(recommended_movie_ids, columns=['movieId']).merge(movies_df, on='movieId')

            # Create a layout with 2 rows of 5 columns for perfect alignment
            cols_row1 = st.columns(5)
            cols_row2 = st.columns(5)
            
            # Distribute the first 5 movies
            for i in range(min(5, len(recommended_movies_df))):
                with cols_row1[i]:
                    movie = recommended_movies_df.iloc[i]
                    poster_url = fetch_poster(movie['title'])
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"<p style='text-align: center; color: white;'>{movie['title']}</p>", unsafe_allow_html=True)

            # Distribute the next 5 movies
            if len(recommended_movies_df) > 5:
                for i in range(5, len(recommended_movies_df)):
                    with cols_row2[i-5]:
                        movie = recommended_movies_df.iloc[i]
                        poster_url = fetch_poster(movie['title'])
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"<p style='text-align: center; color: white;'>{movie['title']}</p>", unsafe_allow_html=True)
        else:
            st.error("Could not generate recommendations for this user.")
