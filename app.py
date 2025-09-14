import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
df = pd.read_csv("products.csv")

# Combine relevant features for recommendation
df["combined_features"] = (
    df["product_name"].astype(str) + " " +
    df["brand"].astype(str) + " " +
    df["description"].astype(str)
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Similarity calculation
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(product_index, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip itself
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# Streamlit UI
st.title("🛍️ Product Recommendation System")

# Product selection
product_list = df["product_name"].tolist()
selected_product = st.selectbox("Choose a product:", product_list)

# Show selected product details
selected_index = df[df["product_name"] == selected_product].index[0]
st.subheader("Selected Product")
st.image(df.loc[selected_index, "image"], width=200)
st.write(f"**Name:** {df.loc[selected_index, 'product_name']}")
st.write(f"**Brand:** {df.loc[selected_index, 'brand']}")
st.write(f"**Price:** ₹{df.loc[selected_index, 'price']}")
st.write(f"**Description:** {df.loc[selected_index, 'description']}")

# Show recommendations
st.subheader("You may also like:")
recommendations = recommend(selected_index, num_recommendations=5)

for i, row in recommendations.iterrows():
    st.image(row["image"], width=150)
    st.write(f"**Name:** {row['product_name']}")
    st.write(f"**Brand:** {row['brand']}")
    st.write(f"**Price:** ₹{row['price']}")
    st.write("---")
