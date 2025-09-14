import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="E-commerce Recommender", layout="wide")

# Load dataset
df = pd.read_csv("products.csv")

# Combine features
df["combined"] = df["product_name"].astype(str) + " " + df["brand"].astype(str) + " " + df["description"].astype(str)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(idx, n=4):
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:n+1]]
    return df.iloc[top_indices]

# UI
st.title("🛍️ Product Recommendation System")

product_list = df["product_name"].tolist()
choice = st.selectbox("Choose a product:", product_list)

if choice:
    idx = df[df["product_name"] == choice].index[0]
    chosen = df.iloc[idx]

    # Selected product card
    st.subheader("Selected Product")
    st.image(chosen["image"], width=250)
    st.markdown(f"### {chosen['product_name']}")
    st.markdown(f"**Brand:** {chosen['brand']}")
    st.markdown(f"**Price:** ₹{chosen['price']}")
    st.write(chosen["description"])

    # Recommendations
    st.subheader("You may also like:")
    recs = recommend(idx, n=4)

    cols = st.columns(2)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 2]:
            st.image(row["image"], width=200)
            st.markdown(f"**{row['product_name']}**")
            st.write(f"Brand: {row['brand']}")
            st.write(f"Price: ₹{row['price']}")
            st.caption(row["description"])
            st.markdown("---")
