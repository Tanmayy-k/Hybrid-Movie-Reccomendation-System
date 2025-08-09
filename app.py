import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================================
# LOAD SAVED ASSETS (Using Streamlit's caching for efficiency)
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
# HELPER FUNCTION FOR STAR RATINGS
# =====================================================================================
@st.cache_data
def get_average_rating(movie_id):
    """Calculates the average rating for a movie and creates a star representation."""
    avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
    if pd.isna(avg_rating):
        return "No ratings", ""
    
    star_rating = '⭐' * int(round(avg_rating))
    return f"{avg_rating:.1f}/5.0", star_rating

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

    # Fallback to pure SVD
    predictions = {mid: score for mid, score in zip(all_movie_ids, user_svd_scores)}
    rated_movies = user_ratings['movieId'].unique()
    unrated = {mid: score for mid, score in predictions.items() if mid not in rated_movies}
    top_n = sorted(unrated, key=unrated.get, reverse=True)[:n]
    return top_n

# =====================================================================================
# FINAL STREAMLIT APP UI (with Minimalist Cards & Star Ratings)
# =====================================================================================

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- Custom CSS for Netflix-like theme & Metro Cards ---
st.markdown("""
<style>
    .main { background-color: #141414; }
    .stApp { color: white; }
    h1, h2, h3, h4, h5, h6 { color: #E50914; }
    .st-eb { background-color: #222222; }
    
    .movie-card {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        border: 1px solid #444;
        height: 250px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        color: white;
        margin-bottom: 5px;
    }
    .movie-genre {
        font-size: 12px;
        color: #999;
    }
    .movie-rating {
        font-size: 14px;
        color: #FFC300; /* Gold color for stars */
    }
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
            
            recommended_movies_df = pd.DataFrame(recommended_movie_ids, columns=['movieId']).merge(movies_df, on='movieId')

            # --- Display recommendations in perfectly aligned cards ---
            cols = st.columns(5)
            for i, row in recommended_movies_df.iterrows():
                with cols[i % 5]:
                    avg_rating_text, stars = get_average_rating(row['movieId'])
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-genre">{row['genres']}</div>
                        <div class="movie-rating">{stars}<br>{avg_rating_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("Could not generate recommendations for this user.")
