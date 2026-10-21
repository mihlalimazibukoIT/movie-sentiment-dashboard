"""Movie Review Showdown — sentiment dashboard + movie review Q&A.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

The dashboard analyses an uploaded review CSV. For movie-specific Q&A across
movies outside that CSV, add a TMDB API Read Access Token in the sidebar.
TMDB's public review endpoint may have no reviews for some titles.
"""
import re
from collections import Counter

import pandas as pd
import requests
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Movie Review Showdown", page_icon="🎬", layout="wide")

STOPWORDS = set(
    "the a an and or but if of to in on at for with as is it its this that was were be been "
    "are i you he she they we my your his her their our not no so very just from by about "
    "have has had do does did will would can could there what which who when than then them "
    "movie film films movies one br really also more get got even much".split()
)
TMDB_BASE = "https://api.themoviedb.org/3"


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def score(texts):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(str(t))["compound"] for t in texts]


def label(c):
    return "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"


def top_words(texts, n=15):
    words = Counter()
    for t in texts:
        words.update(w for w in re.findall(r"[a-z']{3,}", str(t).lower()) if w not in STOPWORDS)
    return pd.DataFrame(words.most_common(n), columns=["word", "count"]).set_index("word")


def tmdb_get(path, token, params=None):
    response = requests.get(
        f"{TMDB_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
        params=params or {},
        timeout=20,
    )
    if response.status_code == 401:
        raise ValueError("TMDB rejected the token. Check that you pasted your TMDB API Read Access Token correctly.")
    response.raise_for_status()
    return response.json()


def search_movie(title, token):
    data = tmdb_get("/search/movie", token, {"query": title, "include_adult": "false"})
    results = data.get("results", [])
    return results[0] if results else None


def get_tmdb_reviews(movie_id, token):
    data = tmdb_get(f"/movie/{movie_id}/reviews", token, {"language": "en-US", "page": 1})
    return data.get("results", [])


def movie_year(movie):
    date = movie.get("release_date") or ""
    return date[:4] if date else "year unknown"


def review_summary(movie, reviews):
    if not reviews:
        return (
            f"I found **{movie.get('title', 'this movie')}** ({movie_year(movie)}) in TMDB, "
            "but TMDB currently has no English user reviews available through its review endpoint. "
            "I don't want to invent audience reactions. Try another title or add reviews to your CSV."
        )
    texts = [re.sub(r"<[^>]+>", " ", r.get("content", "")).strip() for r in reviews]
    texts = [t for t in texts if t]
    if not texts:
        return "The review source returned entries, but no readable review text was available."
    analyzer = SentimentIntensityAnalyzer()
    scored = [(t, analyzer.polarity_scores(t)["compound"]) for t in texts]
    labels = [label(s) for _, s in scored]
    counts = Counter(labels)
    total = len(scored)
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral", 0)
    top_positive = max(scored, key=lambda item: item[1])[0]
    top_negative = min(scored, key=lambda item: item[1])[0]
    overview = movie.get("overview") or "No synopsis is available from TMDB."
    return (
        f"### {movie.get('title', 'Movie')} ({movie_year(movie)})\n\n"
        f"{overview}\n\n"
        f"**Audience sentiment from {total} available TMDB review(s)**\n\n"
        f"- Positive: **{pos}/{total} ({pos/total:.0%})**\n"
        f"- Negative: **{neg}/{total} ({neg/total:.0%})**\n"
        f"- Neutral: **{neu}/{total} ({neu/total:.0%})**\n\n"
        f"**A positive review highlights:** {top_positive[:500]}{'…' if len(top_positive) > 500 else ''}\n\n"
        f"**A critical review highlights:** {top_negative[:500]}{'…' if len(top_negative) > 500 else ''}\n\n"
        "_Sentiment is estimated with VADER from the review text shown by TMDB. The available reviews "
        "may not represent all viewers, and this is not a professional rating._"
    )


def answer_dataset_question(question, df, text_col, title_col=None):
    q = question.strip().lower()
    if title_col:
        title_values = df[title_col].astype(str)
        candidates = sorted(set(title_values), key=lambda x: len(x), reverse=True)
        match = next((t for t in candidates if t.lower() in q and t.lower() not in ("nan", "")), None)
        if match:
            subset = df[title_values == match]
            if subset.empty:
                return None
            counts = subset["predicted"].value_counts()
            total = len(subset)
            common = top_words(subset[text_col], 8)
            words = ", ".join(common.index.tolist()) if not common.empty else "not enough repeated terms"
            return (
                f"### {match}\n\nBased on **{total} review(s)** in your uploaded dataset:\n\n"
                f"- Positive: {counts.get('Positive', 0)}\n"
                f"- Negative: {counts.get('Negative', 0)}\n"
                f"- Neutral: {counts.get('Neutral', 0)}\n\n"
                f"Common review terms include: **{words}**.\n\n"
                "This result only reflects the reviews in your uploaded dataset."
            )
    if any(word in q for word in ["negative", "dislike", "complaint", "bad", "worst", "critic"]):
        subset = df[df["predicted"] == "Negative"]
        heading = "Negative reviews in your dataset"
    elif any(word in q for word in ["positive", "like", "good", "best", "praise", "love"]):
        subset = df[df["predicted"] == "Positive"]
        heading = "Positive reviews in your dataset"
    else:
        subset = df
        heading = "Overall dataset sentiment"
    counts = df["predicted"].value_counts()
    total = len(df)
    if subset.empty:
        return "I couldn't find matching reviews in the current dataset."
    common = top_words(subset[text_col], 10)
    words = ", ".join(common.index.tolist()) if not common.empty else "not enough repeated terms"
    examples = subset.sort_values("compound", ascending=(heading.startswith("Negative")))[text_col].head(2).tolist()
    example_text = "\n\n".join(f"> {str(t)[:350]}{'…' if len(str(t)) > 350 else ''}" for t in examples)
    return (
        f"### {heading}\n\n"
        f"Across **{total} reviews** in your current sample, sentiment is "
        f"{counts.get('Positive', 0)/total:.0%} positive, "
        f"{counts.get('Negative', 0)/total:.0%} negative and "
        f"{counts.get('Neutral', 0)/total:.0%} neutral.\n\n"
        f"Common terms in the selected reviews: **{words}**.\n\n"
        f"Example review(s):\n\n{example_text}\n\n"
        "_These results describe your uploaded dataset, not every movie or every viewer._"
    )


st.title("🎬 Movie Review Showdown")
st.caption("Explore audience reactions, analyse sentiment, and ask questions about movies.")

with st.sidebar:
    st.header("⚙️ Data & connections")
    uploaded = st.file_uploader("Upload your movie-review CSV", type="csv", help="For example, the IMDb 50K reviews dataset.")
    sample_size = st.slider("Reviews to analyse", 500, 10000, 2000, step=500)
    st.divider()
    st.subheader("Movie-wide Q&A")
    tmdb_token = st.text_input(
        "TMDB API Read Access Token",
        type="password",
        help="Optional. Get a free API access token from your TMDB account settings. It is used only for this session.",
    )
    st.caption("With a TMDB token, the app can search movie titles and analyse available TMDB user reviews. Some films have no reviews available.")
    st.markdown("[Get a TMDB API token](https://www.themoviedb.org/settings/api)")

if not uploaded:
    st.info("Upload your movie-review CSV in the sidebar to start analysing reviews.")
    st.markdown("### What you can do here")
    a, b, c = st.columns(3)
    a.markdown("**📊 Sentiment dashboard**\n\nSee positive, negative and neutral review patterns.")
    b.markdown("**💬 Ask questions**\n\nAsk about your dataset or search a movie using TMDB.")
    c.markdown("**🥊 Movie showdown**\n\nCompare audience reactions once review data is available.")
    st.stop()

try:
    raw = load_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"I couldn't read that CSV: {exc}")
    st.stop()

if raw.empty or len(raw.columns) == 0:
    st.error("The uploaded CSV appears to be empty.")
    st.stop()

with st.sidebar:
    text_candidates = [c for c in raw.columns if any(k in c.lower() for k in ["review", "text", "content", "comment"])]
    default_text_idx = raw.columns.get_loc(text_candidates[0]) if text_candidates else 0
    text_col = st.selectbox("Review text column", list(raw.columns), index=default_text_idx)
    title_candidates = [c for c in raw.columns if any(k in c.lower() for k in ["movie_title", "title", "movie", "film_name"])]
    title_options = ["— No movie-title column —"] + list(raw.columns)
    default_title_idx = (list(raw.columns).index(title_candidates[0]) + 1) if title_candidates else 0
    title_choice = st.selectbox("Movie title column (if available)", title_options, index=default_title_idx)

df = raw.dropna(subset=[text_col]).drop_duplicates(subset=[text_col]).copy()
df[text_col] = df[text_col].astype(str)
if df.empty:
    st.error("No review text was found in the selected column.")
    st.stop()
df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
title_col = None if title_choice == "— No movie-title column —" else title_choice
df["compound"] = score(tuple(df[text_col]))
df["predicted"] = df["compound"].apply(label)

st.subheader("📊 Sentiment overview")
counts = df["predicted"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews analysed", f"{len(df):,}")
c2.metric("Positive", f"{counts.get('Positive', 0) / len(df):.0%}")
c3.metric("Negative", f"{counts.get('Negative', 0) / len(df):.0%}")
c4.metric("Average sentiment", f"{df['compound'].mean():.2f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Sentiment distribution")
    st.bar_chart(counts)
with right:
    st.markdown("#### Score spread (-1 to +1)")
    spread = pd.cut(df["compound"], bins=20, duplicates="drop").value_counts().sort_index()
    st.bar_chart(spread.rename(lambda i: str(round(i.mid, 2))))

if "sentiment" in df.columns:
    truth = df["sentiment"].astype(str).str.capitalize()
    mask = df["predicted"] != "Neutral"
    if mask.any():
        acc = (truth[mask] == df.loc[mask, "predicted"]).mean()
        st.subheader("How does the model compare with dataset labels?")
        st.write(f"Agreement with the dataset's labels: **{acc:.1%}** (neutral predictions excluded).")
        st.dataframe(pd.crosstab(truth, df["predicted"]), use_container_width=True)

st.markdown("#### What words appear most often?")
a, b = st.columns(2)
with a:
    st.caption("Positive reviews")
    words = top_words(df.loc[df["predicted"] == "Positive", text_col])
    if not words.empty:
        st.bar_chart(words)
    else:
        st.write("No positive reviews in this sample.")
with b:
    st.caption("Negative reviews")
    words = top_words(df.loc[df["predicted"] == "Negative", text_col])
    if not words.empty:
        st.bar_chart(words)
    else:
        st.write("No negative reviews in this sample.")

st.download_button("⬇️ Download sentiment results (CSV)", df.to_csv(index=False), "sentiment_results.csv", mime="text/csv")

st.divider()
st.header("💬 Ask about a movie")
st.write("Ask a question about audience reactions, what people liked or disliked, or whether a movie might suit your taste.")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("e.g. Are reviews for Dune mostly positive? What do viewers dislike?")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking the available review data…"):
            try:
                if tmdb_token.strip():
                    # Treat the main named phrase as the movie query, removing common question wording.
                    title_query = re.sub(
                        r"\b(is|are|was|were|what|why|how|should|would|could|do|does|did|tell me about|"
                        r"reviews?|review|people|viewers|audience|think|thoughts|like|dislike|about|"
                        r"worth watching|worth it|good|bad|best|worst|movie|film|the|for|of|on|in)\b",
                        " ", prompt, flags=re.I
                    )
                    title_query = re.sub(r"[?!.,]", " ", title_query)
                    title_query = " ".join(title_query.split()).strip()
                    movie = search_movie(title_query or prompt, tmdb_token.strip())
                    if movie:
                        reviews = get_tmdb_reviews(movie["id"], tmdb_token.strip())
                        answer = review_summary(movie, reviews)
                    else:
                        answer = (
                            f"I couldn't find a movie matching **{title_query or prompt}** in TMDB. "
                            "Try the full title, or check the spelling. You can also ask about the uploaded dataset."
                        )
                else:
                    answer = answer_dataset_question(prompt, df, text_col, title_col)
                    answer += (
                        "\n\n**Want to ask about a movie outside this dataset?** Add a TMDB API Read Access Token "
                        "in the sidebar. Without it, I can only answer from your uploaded reviews."
                    )
            except requests.RequestException as exc:
                answer = f"I couldn't reach TMDB right now. Please try again shortly. Technical detail: {exc}"
            except ValueError as exc:
                answer = str(exc)
            except Exception as exc:
                answer = f"Something went wrong while answering that question: {exc}"
            st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.button("🧹 Clear chat"):
    st.session_state.chat_history = []
    st.rerun()
