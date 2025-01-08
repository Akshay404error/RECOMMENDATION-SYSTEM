import pandas as pd  
import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.feature_extraction.text import CountVectorizer  

# Step 1: Create sample datasets and save as CSV files  

# Sample ratings dataset  
ratings_data = {  
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],  
    'movie_id': [101, 102, 103, 101, 104, 102, 105, 103, 104],  
    'rating': [5, 3, 4, 2, 5, 4, 5, 1, 3]  
}  
ratings = pd.DataFrame(ratings_data)  
ratings.to_csv('ratings.csv', index=False)  

# Sample movies dataset  
movies_data = {  
    'movie_id': [101, 102, 103, 104, 105],  
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],  
    'genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy']  
}  
movies = pd.DataFrame(movies_data)  
movies.to_csv('movies.csv', index=False)  

# Step 2: Load datasets from CSV files  
ratings = pd.read_csv('ratings.csv')  
movies = pd.read_csv('movies.csv')  

# Step 3: Implement Collaborative Filtering  
def recommend_movies_collaborative(user_id, num_recommendations=3):  
    # Create a pivot table  
    pivot_table = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)  
    
    # Compute cosine similarity  
    similarity = cosine_similarity(pivot_table)  
    sim_df = pd.DataFrame(similarity, index=pivot_table.index, columns=pivot_table.index)  

    similar_users = sim_df[user_id].sort_values(ascending=False)  
    similar_users = similar_users[similar_users.index != user_id]  
    top_users = similar_users.head(3).index  

    recommended_movies = ratings[ratings['user_id'].isin(top_users)]  
    recommended_movies = recommended_movies.groupby('movie_id')['rating'].mean().sort_values(ascending=False)  

    recommended_movie_ids = recommended_movies.head(num_recommendations).index  
    return movies[movies['movie_id'].isin(recommended_movie_ids)]  

# Step 4: Implement Content-Based Filtering  
def recommend_movies_content_based(movie_title, num_recommendations=3):  
    count_vectorizer = CountVectorizer()  
    count_matrix = count_vectorizer.fit_transform(movies['genre'])  

    content_similarity = cosine_similarity(count_matrix)  
    content_sim_df = pd.DataFrame(content_similarity, index=movies['title'], columns=movies['title'])  

    similar_movies = content_sim_df[movie_title].sort_values(ascending=False)  
    return similar_movies.iloc[1:num_recommendations + 1]  

# Step 5: Generate recommendations  
if __name__ == "__main__":  
    # Get collaborative recommendations for user 1  
    print("Collaborative Recommendations for User 1:")  
    collaborative_recommendations = recommend_movies_collaborative(1)  
    print(collaborative_recommendations)  

    # Get content-based recommendations for 'Movie A'  
    print("\nContent-Based Recommendations for 'Movie A':")  
    content_recommendations = recommend_movies_content_based('Movie A')  
    print(content_recommendations)