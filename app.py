"""Streamlit web app: paste a review, get a sentiment prediction."""
import joblib
import streamlit as st

st.set_page_config(page_title="App Review Sentiment Analyzer", page_icon="💬")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


vectorizer, model = load_model()

st.title("💬 App Review Sentiment Analyzer")
st.write(
    "Helps software and app teams quickly see whether users like or dislike "
    "their product. Paste a review below to classify it as positive or negative."
)

examples = {
    "Choose an example...": "",
    "Positive": "Love this app, it is fast and super easy to use!",
    "Negative": "Keeps crashing after the update, completely unusable.",
}
choice = st.selectbox("Try an example", list(examples))
text = st.text_area("Paste an app review:", value=examples[choice], height=140)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter a review first.")
    else:
        p_pos = model.predict_proba(vectorizer.transform([text]))[0][1]
        if p_pos >= 0.5:
            st.success(f"Positive  (confidence: {p_pos:.0%})")
        else:
            st.error(f"Negative  (confidence: {1 - p_pos:.0%})")
        st.progress(float(p_pos), text=f"Positive probability: {p_pos:.0%}")
