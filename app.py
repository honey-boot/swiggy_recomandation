import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load saved objects and data
@st.cache_data
def load_data():
    cleaned_df = pd.read_csv("cleaned_with_clusters.csv")
    with open('column_transformer.pkl', 'rb') as f:
        ct = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    return cleaned_df, ct, scaler, kmeans

cleaned_df, ct, scaler, kmeans = load_data()

# Prepare dropdown options from training data to avoid unknown categories
cities = sorted(cleaned_df['city'].dropna().unique())
cuisines = sorted(cleaned_df['cuisine'].dropna().unique())

st.title("Swiggy Restaurant Recommendation System")

st.markdown("""
Select your preferences to get personalized restaurant recommendations.
""")

# User inputs
city = st.selectbox("Select City", options=cities)
cuisine = st.selectbox("Select Cuisine", options=cuisines)
rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
cost = st.number_input("Max Cost for Two (₹)", min_value=0.0, value=500.0, step=50.0)

# Construct user input
user_input = {
    'city': city,
    'main_city': city,  # Set internally to match model structure
    'cuisine': cuisine,
    'rating': rating,
    'cost': cost
}

def recommend_restaurants(user_input, top_n=5):
    # Prepare a dummy user DataFrame with default values as in training
    user_dict = {
        'id': 0,
        'rating_count': 100,  # default value
        'name_encoded': 0,    # dummy value
        **user_input
    }
    user_df = pd.DataFrame([user_dict])

    # Drop cluster and 'name' from cleaned_df for encoding
    base_df = cleaned_df.drop(columns=['cluster', 'name'])

    # Combine user input with dataset for consistent encoding
    combined_df = pd.concat([base_df, user_df], ignore_index=True)

    # Encode and scale
    encoded_combined = ct.transform(combined_df)
    scaled_combined = scaler.transform(encoded_combined)

    # Predict cluster for user input (last row)
    user_cluster = kmeans.predict([scaled_combined[-1]])[0]

    # Filter restaurants in that cluster
    recommended = cleaned_df[cleaned_df['cluster'] == user_cluster]

    # Filter based on rating and cost
    recommended = recommended[
        (recommended['rating'] >= user_input['rating']) &
        (recommended['cost'] <= user_input['cost'])
    ]

    # Sort by rating and popularity
    recommended = recommended.sort_values(by=['rating', 'rating_count'], ascending=False)

    return recommended[['name', 'city', 'rating', 'cost', 'cuisine']].head(top_n)

if st.button("Get Recommendations"):
    results = recommend_restaurants(user_input, top_n=10)
    if results.empty:
        st.warning("No restaurants found matching your preferences.")
    else:
        st.success(f"Top {len(results)} restaurant recommendations for you:")
        st.dataframe(results.reset_index(drop=True))