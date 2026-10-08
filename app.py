"""Movie Review Sentiment Dashboard
Run with:  streamlit run app.py
Install:   pip install streamlit pandas vaderSentiment
"""
import re
from collections import Counter

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Movie Review Sentiment", layout="wide")
st.title("🎬 Movie Review Sentiment Dashboard")

STOPWORDS = set(
    "the a an and or but if of to in on at for with as is it its this that was were be been "
    "are i you he she they we my your his her their our not no so very just from by about "
    "have has had do does did will would can could there what which who when than then them "
    "movie film films movies one br really also more get got even much".split()
)


@st.cache_data
def load(file):
    return pd.read_csv(file)


@st.cache_data
def score(texts):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def label(c):
    return "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"


def top_words(texts, n=15):
    words = Counter()
    for t in texts:
        words.update(w for w in re.findall(r"[a-z']{3,}", t.lower()) if w not in STOPWORDS)
    return pd.DataFrame(words.most_common(n), columns=["word", "count"]).set_index("word")


# ---------- Load data ----------
uploaded = st.sidebar.file_uploader("Upload IMDB Dataset.csv", type="csv")
sample_size = st.sidebar.slider("Reviews to analyse", 500, 10000, 2000, step=500)

if not uploaded:
    st.info("Upload the IMDb 50K reviews CSV in the sidebar to begin.")
    st.stop()

df = load(uploaded)
text_col = st.sidebar.selectbox("Review text column", df.columns, index=0)
df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col])
df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)

# ---------- Analyse ----------
df["compound"] = score(tuple(df[text_col]))
df["predicted"] = df["compound"].apply(label)

# ---------- Dashboard ----------
counts = df["predicted"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", f"{len(df):,}")
c2.metric("Positive", f"{counts.get('Positive', 0) / len(df):.0%}")
c3.metric("Negative", f"{counts.get('Negative', 0) / len(df):.0%}")
c4.metric("Avg. score", f"{df['compound'].mean():.2f}")

left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    st.bar_chart(counts)
with right:
    st.subheader("Score spread (-1 to +1)")
    st.bar_chart(pd.cut(df["compound"], bins=20).value_counts().sort_index().rename(lambda i: str(round(i.mid, 2))))

# Accuracy check if the dataset has labels (IMDb has a 'sentiment' column)
if "sentiment" in df.columns:
    truth = df["sentiment"].str.capitalize()
    mask = df["predicted"] != "Neutral"
    acc = (truth[mask] == df["predicted"][mask]).mean()
    st.subheader("How accurate is the tool?")
    st.write(f"Agreement with the dataset's labels: **{acc:.1%}** (neutral predictions excluded).")
    st.dataframe(pd.crosstab(truth, df["predicted"]))

st.subheader("What words drive each side?")
a, b = st.columns(2)
with a:
    st.caption("Most common words in positive reviews")
    st.bar_chart(top_words(df.loc[df["predicted"] == "Positive", text_col]))
with b:
    st.caption("Most common words in negative reviews")
    st.bar_chart(top_words(df.loc[df["predicted"] == "Negative", text_col]))

st.subheader("Most extreme reviews")
p, n = st.columns(2)
with p:
    st.caption("Most positive")
    for t in df.nlargest(3, "compound")[text_col]:
        st.write("• " + t[:300] + "…")
with n:
    st.caption("Most negative")
    for t in df.nsmallest(3, "compound")[text_col]:
        st.write("• " + t[:300] + "…")

st.download_button("Download results (CSV)", df.to_csv(index=False), "sentiment_results.csv")
