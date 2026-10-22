"""Movie Review Sentiment Dashboard (fun edition)
Run with:  streamlit run app.py
Install:   pip install streamlit pandas vaderSentiment
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Movie Review Showdown", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
.hero {background: linear-gradient(90deg,#ff512f,#dd2476,#6a11cb); padding: 28px 32px;
       border-radius: 18px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {margin: 6px 0 0 0; opacity: .92;}
.card {border-radius: 16px; padding: 16px 18px; color: #1a1a1a; height: 100%;}
.card h3 {margin: 0 0 4px 0; font-size: 1rem;}
.card .big {font-size: 1.6rem; font-weight: 700;}
.love {background: #d9f7e4; border: 2px solid #34c76b;}
.roast {background: #ffe1df; border: 2px solid #ff5a4f;}
.fun {background: #fff4cc; border: 2px solid #ffc833;}
.quote {border-left: 5px solid; padding: 8px 12px; margin: 8px 0; border-radius: 6px; font-size: .95rem;}
.q-love {border-color: #34c76b; background: #f2fcf6;}
.q-roast {border-color: #ff5a4f; background: #fff6f5;}
</style>
<div class="hero"><h1>🎬 Movie Review Showdown</h1>
<p>Who's swooning, who's ranting, and what exactly are they fighting about?</p></div>
""",
    unsafe_allow_html=True,
)

STOPWORDS = set(
    "the a an and or but if of to in on at for with as is it its this that was were be been "
    "are i you he she they we my your his her their our not no so very just from by about "
    "have has had do does did will would can could there what which who when than then them "
    "movie film films movies one br really also more get got even much into out all only some "
    "any these those how why too over after before while her him she's it's don't isn't".split()
)

ASPECTS = {
    "🎭 Acting": ["acting", "actor", "actress", "performance", "performances", "cast"],
    "📖 Story & plot": ["plot", "story", "script", "storyline", "writing"],
    "🎬 Directing": ["director", "directing", "direction"],
    "🎵 Music": ["music", "soundtrack", "score", "song", "songs"],
    "👀 Visuals": ["visual", "visuals", "cinematography", "effects", "scenery", "photography"],
    "🏁 Ending": ["ending", "finale", "conclusion"],
    "⏱️ Pacing": ["pacing", "slow", "boring", "dragged", "predictable"],
    "😂 Humor": ["funny", "humor", "comedy", "hilarious", "laugh", "laughs"],
    "🧑‍🤝‍🧑 Characters": ["character", "characters"],
    "💬 Dialogue": ["dialogue", "dialog", "lines"],
}


@st.cache_data
def load(file):
    return pd.read_csv(file)


@st.cache_data
def score(texts):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


def label(c):
    return "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"


def clean(t):
    return re.sub(r"<[^>]+>", " ", t)  # IMDb reviews contain <br /> tags


def top_words(texts, n=12):
    words = Counter()
    for t in texts:
        words.update(w for w in re.findall(r"[a-z']{3,}", t.lower()) if w not in STOPWORDS)
    return pd.DataFrame(words.most_common(n), columns=["word", "count"]).set_index("word")


def card(css, title, big, small=""):
    return f'<div class="card {css}"><h3>{title}</h3><div class="big">{big}</div>{small}</div>'


# ---------- Try it yourself (works for everyone, no files needed) ----------
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


st.subheader("✍️ Try it yourself")
st.caption("Type or paste any movie review (or any sentence) and see how the AI reads it.")
user_text = st.text_area(
    "Your review",
    value="I expected to hate it, but the acting was brilliant and the ending moved me.",
    height=90,
    label_visibility="collapsed",
)
if user_text.strip():
    az = get_analyzer()
    c = az.polarity_scores(user_text)["compound"]
    lab = label(c)
    css, emoji = {"Positive": ("love", "😍"), "Negative": ("roast", "😡"), "Neutral": ("fun", "😐")}[lab]
    t1, t2 = st.columns([1, 2])
    t1.markdown(card(css, f"{emoji} The AI says", lab, f"<br>score {c:+.2f} (from -1 to +1)"), unsafe_allow_html=True)
    with t2:
        st.progress((c + 1) / 2)
        seen, hits = set(), []
        for w in re.findall(r"[A-Za-z']+", user_text):
            k = w.lower()
            if k in az.lexicon and k not in seen:
                seen.add(k)
                hits.append((k, az.lexicon[k]))
        hits.sort(key=lambda x: -abs(x[1]))
        if hits:
            st.write("**Words that moved the score:** " + ", ".join(f"{w} ({v:+.1f})" for w, v in hits[:8]))
        else:
            st.write("No strongly emotional words found, so it reads as neutral.")
st.divider()

# ---------- Load data ----------
SAMPLE = Path(__file__).with_name("sample_reviews.csv")

DEMO_ROWS = [
    ("The acting was brilliant and the story kept me hooked from the first minute.", "positive"),
    ("A beautiful film with a moving soundtrack. I cried at the ending.", "positive"),
    ("Hilarious from start to finish. The whole cast is wonderful and the humor lands every time.", "positive"),
    ("Stunning visuals and a clever plot. One of the best films I have seen this year.", "positive"),
    ("Great characters, sharp dialogue and a satisfying ending. Loved it!", "positive"),
    ("The director did a fantastic job. Every scene felt intentional and the pacing was perfect.", "positive"),
    ("Warm, funny and surprisingly emotional. The performances are excellent.", "positive"),
    ("I was gripped the whole time. The story twists were smart and the music was amazing.", "positive"),
    ("A delightful movie the whole family enjoyed. Wonderful acting by the young cast.", "positive"),
    ("Powerful, honest and beautifully shot. The ending stayed with me for days.", "positive"),
    ("Brilliant script and a great cast. I laughed, I cried, I want to watch it again.", "positive"),
    ("An inspiring story with superb acting and gorgeous scenery.", "positive"),
    ("Thrilling and well paced, with characters you actually care about.", "positive"),
    ("The soundtrack alone is worth the ticket. A genuinely great film.", "positive"),
    ("Smart, touching and perfectly cast. Highly recommended.", "positive"),
    ("The plot made no sense and the ending was a complete letdown.", "negative"),
    ("Boring and predictable. The pacing dragged and I nearly fell asleep.", "negative"),
    ("Terrible acting and awful dialogue. I wanted my money back.", "negative"),
    ("The special effects looked cheap and the characters were flat and annoying.", "negative"),
    ("A waste of a good cast. The story was a mess and the script was lazy.", "negative"),
    ("Painfully slow with a pointless ending. I would not recommend it.", "negative"),
    ("The director clearly had no idea what he wanted. Confusing and dull.", "negative"),
    ("Cringeworthy humor and wooden performances. Hard to sit through.", "negative"),
    ("Loud, messy and forgettable. The music was irritating and the plot was thin.", "negative"),
    ("I hated the ending. After two hours of nothing happening, it just stopped.", "negative"),
    ("Poorly written and badly acted. One of the worst films I have watched.", "negative"),
    ("The characters made stupid decisions and the story fell apart halfway through.", "negative"),
    ("Disappointing. Great trailer, but the movie itself was boring and bland.", "negative"),
    ("Overlong, overhyped and just not funny. The dialogue was painful.", "negative"),
    ("An awful movie with a terrible script. I regret watching it.", "negative"),
]

uploaded = st.sidebar.file_uploader("📂 Upload your own CSV (optional)", type="csv")

if uploaded:
    df = load(uploaded)
    source = "your uploaded file"
elif SAMPLE.exists():
    df = load(str(SAMPLE))
    source = "the built-in sample of IMDb reviews"
else:
    df = pd.DataFrame(DEMO_ROWS, columns=["review", "sentiment"])
    source = "a tiny demo set of 30 reviews"

st.info(f"Showing results for {source}. Upload the full IMDb 50K CSV in the sidebar to analyse more reviews.")

default_idx = list(df.columns).index("review") if "review" in df.columns else 0
text_col = st.sidebar.selectbox("Review text column", df.columns, index=default_idx)
if len(df) >= 1000:
    sample_size = st.sidebar.slider("Reviews to analyse", 500, min(10000, len(df)), min(2000, len(df)), step=500)
else:
    sample_size = len(df)
df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col])
df = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
df[text_col] = df[text_col].astype(str).apply(clean)

df["compound"] = score(tuple(df[text_col]))
df["predicted"] = df["compound"].apply(label)
df["words"] = df[text_col].str.split().str.len()
df["exclaims"] = df[text_col].str.count("!")
df["shouting"] = df[text_col].str.contains(r"\b[A-Z]{4,}\b", regex=True)

pos = df[df["predicted"] == "Positive"]
neg = df[df["predicted"] == "Negative"]
neu = df[df["predicted"] == "Neutral"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎭 The Showdown", "😍 Love Letters", "😡 Roast Corner", "🎯 Is the AI right?"]
)

# ---------- Tab 1: overview ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("love", "😍 Fans", f"{len(pos)/len(df):.0%}", f"<br>{len(pos):,} positive reviews"), unsafe_allow_html=True)
    c2.markdown(card("roast", "😡 Critics", f"{len(neg)/len(df):.0%}", f"<br>{len(neg):,} negative reviews"), unsafe_allow_html=True)
    c3.markdown(card("fun", "😐 On the fence", f"{len(neu)/len(df):.0%}", f"<br>{len(neu):,} neutral reviews"), unsafe_allow_html=True)

    st.write("")
    st.subheader("🔥 Fun facts: fans vs critics")
    f1, f2, f3, f4 = st.columns(4)
    f1.markdown(card("fun", "📝 Longer rants",
                     "Critics" if neg["words"].mean() > pos["words"].mean() else "Fans",
                     f"<br>{neg['words'].mean():.0f} vs {pos['words'].mean():.0f} words (critics vs fans)"), unsafe_allow_html=True)
    f2.markdown(card("fun", "❗ Exclamation marks",
                     "Critics" if neg["exclaims"].mean() > pos["exclaims"].mean() else "Fans",
                     f"<br>{neg['exclaims'].mean():.2f} vs {pos['exclaims'].mean():.2f} per review"), unsafe_allow_html=True)
    f3.markdown(card("fun", "📢 ALL-CAPS shouting",
                     "Critics" if neg["shouting"].mean() > pos["shouting"].mean() else "Fans",
                     f"<br>{neg['shouting'].mean():.0%} vs {pos['shouting'].mean():.0%} of reviews"), unsafe_allow_html=True)
    f4.markdown(card("fun", "🌡️ Overall mood", f"{df['compound'].mean():+.2f}", "<br>scale: -1 grumpy to +1 delighted"), unsafe_allow_html=True)

    st.write("")
    st.subheader("🎞️ What are they actually talking about?")
    lower = df[text_col].str.lower()
    rows = []
    for name, kws in ASPECTS.items():
        mask = lower.str.contains(r"\b(?:" + "|".join(kws) + r")\b", regex=True)
        if mask.sum() == 0:
            continue
        rows.append({
            "aspect": name,
            "In fan reviews (%)": round(mask[df["predicted"] == "Positive"].mean() * 100, 1) if len(pos) else 0,
            "In critic reviews (%)": round(mask[df["predicted"] == "Negative"].mean() * 100, 1) if len(neg) else 0,
            "mood": df.loc[mask, "compound"].mean(),
            "mentions": int(mask.sum()),
        })
    asp = pd.DataFrame(rows).set_index("aspect")
    st.bar_chart(asp[["In fan reviews (%)", "In critic reviews (%)"]], color=["#34c76b", "#ff5a4f"])

    best, worst = asp["mood"].idxmax(), asp["mood"].idxmin()
    a1, a2 = st.columns(2)
    a1.markdown(card("love", "🏆 Most loved topic", best, f"<br>avg mood {asp.loc[best,'mood']:+.2f} across {asp.loc[best,'mentions']} reviews"), unsafe_allow_html=True)
    a2.markdown(card("roast", "💣 Most roasted topic", worst, f"<br>avg mood {asp.loc[worst,'mood']:+.2f} across {asp.loc[worst,'mentions']} reviews"), unsafe_allow_html=True)

    st.subheader("📈 Mood spread")
    spread = pd.cut(df["compound"], bins=20).value_counts().sort_index()
    spread.index = [str(round(i.mid, 2)) for i in spread.index]
    st.bar_chart(spread, color="#6a11cb")

# ---------- Tab 2 / 3: love & roast ----------
def side(tab, data, emoji, css, qcss, title, ascending):
    with tab:
        st.subheader(f"{emoji} {title}")
        st.caption(f"Most common words ({len(data):,} reviews)")
        st.bar_chart(top_words(data[text_col]), color="#34c76b" if css == "love" else "#ff5a4f")
        st.caption("The most extreme ones")
        picks = data.nsmallest(5, "compound") if ascending else data.nlargest(5, "compound")
        for _, r in picks.iterrows():
            st.markdown(f'<div class="quote {qcss}">“{r[text_col][:350]}…”<br><b>score {r["compound"]:+.2f}</b></div>', unsafe_allow_html=True)

side(tab2, pos, "😍", "love", "q-love", "Love Letters: what the fans are gushing about", False)
side(tab3, neg, "😡", "roast", "q-roast", "Roast Corner: what the critics can't stand", True)

# ---------- Tab 4: accuracy ----------
with tab4:
    st.subheader("🎯 How well does the AI judge match the humans?")
    if "sentiment" in df.columns:
        truth = df["sentiment"].str.capitalize()
        mask = df["predicted"] != "Neutral"
        acc = (truth[mask] == df["predicted"][mask]).mean()
        st.metric("Agreement with the dataset's labels", f"{acc:.1%}")
        st.write("Rows are what the reviewer chose, columns are what the AI predicted:")
        st.dataframe(pd.crosstab(truth, df["predicted"]))
        st.caption("Neutral predictions are excluded from the accuracy figure. Mixed or sarcastic reviews are where VADER slips.")
    else:
        st.write("This dataset has no 'sentiment' label column, so accuracy can't be checked.")

st.download_button("⬇️ Download results (CSV)", df.drop(columns=["shouting"]).to_csv(index=False), "sentiment_results.csv")
