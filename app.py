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
# HELPER FUNCTIONS
# =====================================================================================
@st.cache_data
def get_average_rating(movie_id):
    """Calculates the average rating for a movie and creates a star representation."""
    avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
    if pd.isna(avg_rating):
        return 0, "No ratings", "" # Return 0 for sorting
    
    star_rating = '⭐' * int(round(avg_rating))
    return avg_rating, f"{avg_rating:.1f}/5.0", star_rating

@st.cache_data
def get_user_top_genres(user_id):
    """Finds the top genres for a given user based on their high ratings."""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    high_rated = user_ratings[user_ratings['rating'] >= 4]
    if high_rated.empty:
        return "No specific taste profile found."
    
    user_movies = movies_df[movies_df['movieId'].isin(high_rated['movieId'])]
    genre_counts = user_movies['genres'].str.split('|', expand=True).stack().value_counts()
    top_genres = genre_counts.head(3).index.tolist()
    
    if len(top_genres) == 0:
        return "A diverse taste in movies."
    return f"Based on your love for {', '.join(top_genres)} movies..."

# =====================================================================================
# RECOMMENDATION FUNCTION
# =====================================================================================
def fast_hybrid_recommendations(user_id, svd_model, ratings_df, alpha=0.5, n=20):
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
# FINAL STREAMLIT APP UI
# =====================================================================================

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- Custom CSS for theme & cards ---
st.markdown("""
<style>
    .main { background-color: #141414; }
    .stApp { color: white; }
    h1, h2, h3, h4, h5, h6 { color: #E50914; }
    .st-eb { background-color: #222222; }
    .movie-card {
        background-color: #2D2D2D; /* Uniform dark grey for all cards */
        border-radius: 10px; padding: 15px; margin: 10px; text-align: center;
        border: 1px solid #444; height: 250px; display: flex; flex-direction: column;
        justify-content: space-between;
    }
    .movie-title { font-size: 16px; font-weight: bold; color: white; margin-bottom: 5px; }
    /* --- FIX: Increased genre font size --- */
    .movie-genre { font-size: 14px; color: #999; }
    .movie-rating { font-size: 14px; color: #FFC300; }
</style>
""", unsafe_allow_html=True)

st.title('🎬 Movie Recommender')

# --- User Input Section ---
st.sidebar.header('Enter Your User ID')

# Initialize session state for user ID if it doesn't exist
if 'user_id' not in st.session_state:
    st.session_state.user_id = ''

# --- FIX: Function to handle recommendation logic ---
def show_recommendations():
    user_id_input = st.session_state.user_id
    if user_id_input:
        with st.spinner('Finding movies you might like...'):
            recommended_movie_ids = fast_hybrid_recommendations(
                user_id=user_id_input, svd_model=svd_model, ratings_df=ratings_df, alpha=0.5, n=20
            )
            
            # Store recommendations in session state
            if recommended_movie_ids:
                st.session_state.recommendations = pd.DataFrame(recommended_movie_ids, columns=['movieId']).merge(movies_df, on='movieId')
                st.session_state.recommendations['avg_rating'] = st.session_state.recommendations['movieId'].apply(lambda x: get_average_rating(x)[0])
            else:
                st.session_state.recommendations = None
    else:
        st.session_state.recommendations = None # Clear recommendations if no user ID

# --- FIX: Use a form for the input and button ---
with st.sidebar.form(key='recommendation_form'):
    user_id_input = st.number_input('User ID (1 to 6040)', min_value=1, max_value=6040, value=None, key='user_id', placeholder="Type user ID & press Enter")
    submit_button = st.form_submit_button(label='Get Recommendations')

# Trigger recommendations if the form is submitted (button click or enter)
if submit_button:
    show_recommendations()

# --- Display Area ---
if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
    recommended_movies_df = st.session_state.recommendations
    
    st.header(f'Welcome, User {st.session_state.user_id}!')
    st.write(get_user_top_genres(st.session_state.user_id))
    st.markdown("---")
    
    all_genres = sorted(list(set([genre for sublist in movies_df['genres'].str.split('|') for genre in sublist])))
    
    col1, col2 = st.columns(2)
    with col1:
        genre_filter = st.selectbox('Filter by Genre:', ['All'] + all_genres)
    with col2:
        sort_order = st.selectbox('Sort by:', ['Recommendation Score', 'Average Rating'])

    # Apply Filters and Sorting
    if genre_filter != 'All':
        recommended_movies_df = recommended_movies_df[recommended_movies_df['genres'].str.contains(genre_filter)]
    if sort_order == 'Average Rating':
        recommended_movies_df = recommended_movies_df.sort_values(by='avg_rating', ascending=False)
    
    # --- FIX: Reset index after filtering/sorting ---
    recommended_movies_df = recommended_movies_df.reset_index(drop=True)
    
    if recommended_movies_df.empty:
        st.warning("No recommendations match your filter.")
    else:
        st.subheader(f'Top Recommendations for You')
        cols = st.columns(5)
        for i, row in recommended_movies_df.head(10).iterrows():
            with cols[i % 5]:
                _, avg_rating_text, stars = get_average_rating(row['movieId'])
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row['genres']}</div>
                    <div class="movie-rating">{stars}<br>{avg_rating_text}</div>
                </div>
                """, unsafe_allow_html=True)
elif st.session_state.get('user_id'): # Handle case where user exists but has no recs
    st.error(f"Could not generate recommendations for User {st.session_state.user_id}.")
else:
    st.info("Please enter a User ID in the sidebar to get recommendations.")
