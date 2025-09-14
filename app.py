# app.py (defensive version)
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="E-commerce Recommender", layout="centered")

st.title("🛍️ E-commerce Recommender (Content-based)")

# ---------- Load CSV safely and normalize column names ----------
@st.cache_data
def load_products(path="products.csv"):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not load '{path}': {e}")
        st.stop()

    # clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # create lowercase map and rename to lowercase for flexible lookup
    lower_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    # candidate lists for common names
    name_cols = ["product_name", "name", "product", "title"]
    desc_cols = ["description", "desc", "details"]
    brand_cols = ["brand", "manufacturer"]
    price_cols = ["price", "cost"]
    image_cols = ["image", "image_url", "imageurl", "img", "photo", "image link"]

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    name_col = find_col(name_cols)
    desc_col = find_col(desc_cols)
    brand_col = find_col(brand_cols)
    price_col = find_col(price_cols)
    image_col = find_col(image_cols)

    missing = []
    if not name_col:
        missing.append("name (product_name/name/title)")
    if not desc_col:
        missing.append("description (description/desc/details)")
    if missing:
        st.error(f"CSV is missing required column(s): {', '.join(missing)}.\nFound columns: {list(df.columns)}")
        st.stop()

    # ensure optional columns exist; if not, create empty columns
    if not brand_col:
        df["brand"] = ""
        brand_col = "brand"
    if not price_col:
        df["price"] = ""
        price_col = "price"
    if not image_col:
        df["image"] = ""
        image_col = "image"

    # rename chosen columns to standard names used below
    df = df.rename(columns={
        name_col: "product_name",
        desc_col: "description",
        brand_col: "brand",
        price_col: "price",
        image_col: "image"
    })

    # Convert price to string to avoid display errors (keep original if numeric)
    df["price"] = df["price"].astype(str)

    return df

df = load_products("products.csv")

# ---------- Build TF-IDF + similarity ----------
@st.cache_data
def build_model(dataframe):
    df2 = dataframe.copy()
    df2["combined"] = (df2["product_name"].astype(str) + " " +
                       df2["brand"].astype(str) + " " +
                       df2["description"].astype(str))
    vect = TfidfVectorizer(stop_words="english")
    tfidf_mat = vect.fit_transform(df2["combined"])
    sim = linear_kernel(tfidf_mat, tfidf_mat)  # cosine similarity
    name_to_idx = {n: i for i, n in enumerate(df2["product_name"].tolist())}
    return df2, sim, name_to_idx

df, cosine_sim, name_to_idx = build_model(df)

# ---------- Recommendation function ----------
def recommend_by_name(product_name, top_n=5):
    """Return DataFrame of top_n recommended rows (excluding the product itself)."""
    if not isinstance(product_name, str) or product_name.strip() == "":
        return pd.DataFrame()
    if product_name not in name_to_idx:
        return pd.DataFrame()
    idx = name_to_idx[product_name]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_idxs = [i for i, s in scores[1: top_n+1]]  # skip itself
    return df.iloc[top_idxs].reset_index(drop=True)

# ---------- UI ----------
st.write("Select a product to see content-based recommendations (TF-IDF on name+brand+description).")

product_list = df["product_name"].tolist()
selected = st.selectbox("Choose product", options=product_list)

n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=8, value=4)

# show selected product info
st.subheader("Selected product")
try:
    sel_row = df[df["product_name"] == selected].iloc[0]
    if sel_row["image"]:
        try:
            st.image(sel_row["image"], width=220)
        except Exception:
            st.text("(image failed to load)")
    st.markdown(f"**{sel_row['product_name']}**")
    st.write(f"Brand: {sel_row.get('brand','')}")
    st.write(f"Price: {sel_row.get('price','')}")
    st.write(sel_row.get("description",""))
except Exception:
    st.write("Selected product display error (check CSV and column names).")

if st.button("Get Recommendations"):
    recs = recommend_by_name(selected, top_n=n)
    if recs.empty:
        st.info("No recommendations found (product name not in CSV or dataset too small).")
    else:
        for _, r in recs.iterrows():
            cols = st.columns([1, 3])
            with cols[0]:
                if r.get("image", "") != "":
                    try:
                        st.image(r["image"], width=130)
                    except Exception:
                        st.text("(image failed)")
                else:
                    st.text("(no image)")
            with cols[1]:
                st.markdown(f"**{r['product_name']}**")
                st.write(f"{r.get('brand','')}  |  ₹{r.get('price','')}")
                st.write(r.get("description",""))
            st.markdown("---")

# Debug helper (shows columns and first rows)
with st.expander("Debug: columns & sample data (for teacher/demo)"):
    st.write("Columns detected:", list(df.columns))
    st.write(df.head(6))
