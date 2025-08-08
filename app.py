
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================================
# LOAD SAVED ASSETS (This is the file you created in Step 6.1)
# =====================================================================================
try:
    assets = joblib.load('recommender_assets.joblib')
    svd_model = assets['svd_model']
    movies_df = assets['movies_df']
    ratings_df = assets['ratings_df']
    genre_similarity_matrix = assets['genre_similarity_matrix']
    movie_id_to_index_map = assets['movie_id_to_index_map']
    index_to_movie_id_map = assets['index_to_movie_id_map']
    movie_ids_array = assets['movie_ids_array']
    print("Assets loaded successfully.")
except FileNotFoundError:
    st.error("Saved assets file not found! Please run the notebook step to generate 'recommender_assets.joblib'.")
    st.stop()


# =====================================================================================
# RECOMMENDATION FUNCTIONS (Copied from your notebook)
# =====================================================================================
def fast_content_recommendations(user_id, ratings_df, n=10):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
    if len(liked_movies) == 0: liked_movies = user_ratings[user_ratings['rating'] >= 3]['movieId'].values
    if len(liked_movies) == 0: return []

    liked_indices = [movie_id_to_index_map.get(mid) for mid in liked_movies if mid in movie_id_to_index_map]
    if not liked_indices: return []

    sim_scores = genre_similarity_matrix[liked_indices].mean(axis=0)

    rated_indices = [movie_id_to_index_map.get(mid) for mid in user_ratings['movieId'].values if mid in movie_id_to_index_map]
    sim_scores[rated_indices] = -1

    top_indices = np.argsort(sim_scores)[::-1][:n]
    return [index_to_movie_id_map[i] for i in top_indices]

def fast_hybrid_recommendations(user_id, svd_model, ratings_df, alpha=0.5, n=10):
    try:
        inner_uid = svd_model.trainset.to_inner_uid(user_id)
    except ValueError:
        return fast_content_recommendations(user_id, ratings_df, n=n)

    user_vector = svd_model.pu[inner_uid]
    item_vectors = svd_model.qi
    svd_scores = user_vector @ item_vectors.T + svd_model.bi + svd_model.bu[inner_uid] + svd_model.trainset.global_mean

    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].values
    if len(liked_movies) == 0: liked_movies = user_ratings[user_ratings['rating'] >= 3]['movieId'].values

    if len(liked_movies) > 0:
        liked_indices = [movie_id_to_index_map.get(mid) for mid in liked_movies if mid in movie_id_to_index_map]
        if liked_indices:
            content_scores = genre_similarity_matrix[liked_indices].mean(axis=0)

            s_min, s_max = svd_scores.min(), svd_scores.max()
            c_min, c_max = content_scores.min(), content_scores.max()

            norm_svd = (svd_scores - s_min) / (s_max - s_min) if (s_max - s_min) > 0 else 0
            norm_content = (content_scores - c_min) / (c_max - c_min) if (c_max - c_min) > 0 else 0

            hybrid_scores = alpha * norm_svd + (1 - alpha) * norm_content

            rated_indices = [movie_id_to_index_map.get(mid) for mid in user_ratings['movieId'].values if mid in movie_id_to_index_map]
            hybrid_scores[rated_indices] = -1

            top_indices = np.argsort(hybrid_scores)[::-1][:n]
            return [index_to_movie_id_map[i] for i in top_indices]

    # Fallback to pure SVD if content profile is empty
    rated_movies = user_ratings['movieId'].unique()
    unrated_movies = movies_df[~movies_df['movieId'].isin(rated_movies)]['movieId'].tolist()
    predictions = [svd_model.predict(user_id, mid) for mid in unrated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [pred.iid for pred in predictions[:n]]

# =====================================================================================
# STREAMLIT APP UI
# =====================================================================================

st.set_page_config(layout="wide", page_title="Movie Recommender")

st.title('🎬 Hybrid Movie Recommender')
st.write("This app recommends movies using a model trained on the MovieLens 1M dataset.")

# --- User Input ---
st.sidebar.header('Enter Your User ID')
user_id_input = st.sidebar.number_input('User ID (1 to 6040)', min_value=1, max_value=6040, value=12, step=1)

# --- Generate Recommendations ---
if st.sidebar.button('Get Recommendations'):
    with st.spinner('Finding movies you might like...'):
        recommended_movie_ids = fast_hybrid_recommendations(
            user_id=user_id_input,
            svd_model=svd_model,
            ratings_df=ratings_df,
            alpha=0.5,
            n=10
        )

        if recommended_movie_ids:
            st.subheader(f'Top 10 Recommendations for User {user_id_input}')

            recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
            st.table(recommended_movies[['title', 'genres']])
        else:
            st.error("Could not generate recommendations for this user. They may have too few ratings.")
