import streamlit as st
import pandas as pd
import numpy as np
import joblib
from justwatch import JustWatch

# ====== Load Pre-trained Assets ======
assets = joblib.load('recommender_assets.joblib')
movies = assets['movies']
cosine_sim = assets['cosine_sim']
svd_model = assets['svd_model']
user_ids = assets['user_ids']
movie_ids = assets['movie_ids']

# ====== JustWatch Poster Fetch ======
def fetch_poster(movie_name):
    try:
        justwatch = JustWatch(country='IN')  # Change to your country code if needed
        results = justwatch.search_for_item(query=movie_name)
        
        if results['items']:
            poster_path = results['items'][0].get('poster')
            if poster_path:
                return "https://images.justwatch.com" + poster_path.replace("{profile}", "s592")
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/500x750?text=Error"

# ====== Recommendation Functions ======
def hybrid_recommendations(title, user_id, top_n=10):
    if title not in movies['title'].values:
        return []
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    content_recs = movies.iloc[movie_indices]
    
    # Collaborative filtering score
    cf_scores = []
    for movieId in content_recs['movieId']:
        if movieId in movie_ids:
            pred = svd_model.predict(user_ids[user_id], movie_ids[movieId]).est
        else:
            pred = 0
        cf_scores.append(pred)
    
    content_recs = content_recs.copy()
    content_recs['cf_score'] = cf_scores
    content_recs = content_recs.sort_values('cf_score', ascending=False)
    
    return content_recs.head(top_n)

# ====== Streamlit UI ======
st.set_page_config(page_title="Hybrid Movie Recommendation System", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

user_id_input = st.text_input("Enter your user ID (numeric):", "1")
if st.button("Recommend"):
    try:
        user_id = int(user_id_input)
        recommendations = hybrid_recommendations(selected_movie, user_id, top_n=10)
        
        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            cols = st.columns(5)
            for i, row in enumerate(recommendations.itertuples()):
                poster_url = fetch_poster(row.title)
                with cols[i % 5]:
                    st.image(poster_url, caption=row.title, use_container_width=True)
    except ValueError:
        st.error("Please enter a valid numeric user ID.")
