"""Duolingo spaced-repetition explorer (Settles & Meeder 2016 data).

Run:  uv run streamlit run app.py --server.port 8502
Needs prepare.py to have been run first.
"""

import re
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

PQ = Path(__file__).resolve().parents[2] / "data" / "duolingo_hlr" / "parquet"
TRACES = PQ / "traces.parquet"
LANGS = {"de": "German", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"}
DELTA_ORDER = ["<10 min", "10-60 min", "1-6 h", "6-24 h", "1-2 d", "2-4 d", "4-7 d",
               "1-2 wk", "2-4 wk", "1-3 mo", "3 mo+"]
SHORT = DELTA_ORDER[:4]   # under a day
LONG = DELTA_ORDER[7:]    # over a week
HIST_ORDER = ["1-2", "3-5", "6-10", "11-25", "26+"]

st.set_page_config(page_title="Spaced repetition explorer", layout="wide")


def label(lexeme: str) -> str:
    """'lernt/lernen<vblex><pri>' -> 'lernt (lernen · vblex pri)'."""
    surface, rest = lexeme.split("/", 1)
    lemma = rest.split("<", 1)[0]
    tags = " ".join(t for t in re.findall(r"<([^>]*)>", rest) if not t.startswith("*"))
    if surface.startswith("<"):
        surface = lemma
    return f"{surface} ({lemma} · {tags})" if surface != lemma else f"{surface} ({tags})"


@st.cache_resource
def load():
    words = pd.read_parquet(PQ / "words.parquet")
    wc = pd.read_parquet(PQ / "word_curve.parquet")
    wc["gap"] = wc.delta_bin.map(lambda b: "short" if b in SHORT else "long" if b in LONG else None)
    g = wc.dropna(subset=["gap"]).assign(hits=lambda d: d.recall * d.n)
    g = g.groupby(["lexeme_id", "gap"])[["hits", "n"]].sum()
    g = (g.hits / g.n).unstack().join(g.n.unstack().add_prefix("n_"))
    words = words.merge(g, left_on="lexeme_id", right_index=True, how="left")
    words["drop"] = words["short"] - words["long"]
    words["label"] = words.lexeme_string.map(label)
    return {
        "words": words,
        "word_curve": wc,
        "curve": pd.read_parquet(PQ / "curve.parquet"),
        "users": pd.read_parquet(PQ / "users.parquet"),
    }


@st.cache_data(max_entries=50)
def user_rows(user: str) -> pd.DataFrame:
    return duckdb.execute(f"SELECT * FROM '{TRACES}' WHERE user = ? ORDER BY ts", [user]).df()


def weighted(df, by):
    return df.assign(hits=df.recall * df.n).groupby(by, observed=True)[["hits", "n"]].sum() \
             .assign(recall=lambda d: d.hits / d.n).drop(columns="hits").reset_index()


D = load()
st.sidebar.markdown(
    f"**{len(D['users']):,} learners · {D['users'].rows.sum():,} word reviews**  \n"
    "Two weeks of Duolingo, 28 Feb – 12 Mar 2013. Each row is one word practised by one learner "
    "in one session.")
st.sidebar.markdown(
    "**Read this first**  \n"
    "Duolingo decides *when* each word comes back, based on how well it thinks you know it. "
    "So a word reviewed after 3 months is usually one the system judged well known, which "
    "makes forgetting look slower than it really is. Treat curves as 'recall at the moment "
    "Duolingo chose to test', not as pure memory decay.")
st.sidebar.markdown(
    "**Recall** = share of a session's exercises where the word was answered right "
    "(most sessions test a word 1-3 times).")

tab_student, tab_word, tab_patterns = st.tabs(["Student", "Word", "Patterns"])

# ---------------------------------------------------------------- student
with tab_student:
    users = D["users"]
    active = users[users.sessions >= 10]
    if "hlr_user" not in st.session_state:
        st.session_state.hlr_user = active.sort_values("rows", ascending=False).user.iloc[100]
    c1, c2 = st.columns([3, 1])
    if c2.button("🎲 Random active learner", use_container_width=True,
                 help="Picks from learners with 10+ sessions"):
        st.session_state.hlr_user = active.user.sample(1).iloc[0]
    typed = c1.text_input("Learner id", st.session_state.hlr_user)
    if typed != st.session_state.hlr_user and typed in set(users.user):
        st.session_state.hlr_user = typed
    user = st.session_state.hlr_user
    ur = users.set_index("user").loc[user]
    rows = user_rows(user)

    m = st.columns(5)
    m[0].metric("Learning", ", ".join(LANGS.get(l, l) for l in ur.langs.split(",")))
    m[1].metric("App language", LANGS.get(ur.ui_lang, ur.ui_lang))
    m[2].metric("Sessions", int(ur.sessions))
    m[3].metric("Distinct words", int(ur.words))
    m[4].metric("Average recall", f"{ur.recall:.1%}")

    sess = rows.groupby("ts").agg(time=("time", "first"), words=("lexeme_id", "size"),
                                  recall=("recall", "mean"), lang=("lang", "first")).reset_index()
    st.altair_chart(
        alt.Chart(sess).mark_circle(opacity=0.8).encode(
            x=alt.X("time:T", title="Session time"),
            y=alt.Y("recall:Q", title="Average recall in session", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
            size=alt.Size("words:Q", title="Words reviewed"),
            color=alt.Color("lang:N", title="Language"),
            tooltip=["time:T", "words", alt.Tooltip("recall:Q", format=".0%")],
        ).properties(height=240), use_container_width=True)

    st.subheader("Sessions")
    st.caption("Pick a session to see every word reviewed in it: how long since the learner last saw it, "
               "their track record before this session, and how they did now.")
    opts = sess.ts.tolist()
    ts = st.select_slider("Session", options=opts, value=opts[0],
                          format_func=lambda t: pd.to_datetime(t, unit="s").strftime("%b %d %H:%M"))
    s = rows[rows.ts == ts].copy()
    s["word"] = s.lexeme_string.map(label)
    s["last seen"] = s.delta_days.map(
        lambda d: f"{d * 1440:.0f} min ago" if d < 1 / 24 else f"{d * 24:.1f} h ago" if d < 1 else f"{d:.1f} days ago")
    s["before"] = s.history_correct.astype(str) + " / " + s.history_seen.astype(str) + " right"
    s["now"] = s.session_correct.astype(str) + " / " + s.session_seen.astype(str) + " right"
    st.dataframe(
        s[["word", "last seen", "before", "now", "recall"]].sort_values("recall"),
        hide_index=True, use_container_width=True,
        column_config={"recall": st.column_config.ProgressColumn("recall now", min_value=0, max_value=1,
                                                                 format="%.2f")})

    with st.expander("One word's history for this learner"):
        wopts = rows.drop_duplicates("lexeme_id").set_index("lexeme_id").lexeme_string
        lid = st.selectbox("Word", wopts.index, format_func=lambda i: label(wopts[i]))
        h = rows[rows.lexeme_id == lid][["time", "delta_days", "history_seen", "history_correct",
                                         "session_seen", "session_correct", "recall"]]
        st.dataframe(h, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- word
with tab_word:
    words = D["words"]
    c1, c2 = st.columns([1, 3])
    lang = c1.selectbox("Language learned", list(LANGS), format_func=LANGS.get, index=3)
    wl = words[(words.lang == lang) & (words.rows >= 50)].sort_values("rows", ascending=False)
    wlabels = dict(zip(wl.lexeme_id, wl.label))
    lid = c2.selectbox("Word", wl.lexeme_id, format_func=wlabels.get)
    wr = wl.set_index("lexeme_id").loc[lid]
    m = st.columns(5)
    m[0].metric("Reviews", f"{int(wr.rows):,}")
    m[1].metric("Learners", f"{int(wr.users):,}")
    m[2].metric("Recall", f"{wr.recall:.1%}")
    m[3].metric("Recall, gap < 1 day", f"{wr.short:.1%}" if pd.notna(wr.short) else "–")
    m[4].metric("Recall, gap > 1 week", f"{wr.long:.1%}" if pd.notna(wr.long) else "–")

    wc = D["word_curve"][D["word_curve"].lexeme_id == lid].assign(series="this word")
    base = weighted(D["curve"][D["curve"].lang == lang], "delta_bin").assign(series=f"all {LANGS[lang]} words")
    both = pd.concat([wc[["delta_bin", "n", "recall", "series"]], base])
    st.altair_chart(
        alt.Chart(both).mark_line(point=True).encode(
            x=alt.X("delta_bin:N", sort=DELTA_ORDER, title="Time since the learner last practised it"),
            y=alt.Y("recall:Q", title="Recall", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=["series", "delta_bin", "n", alt.Tooltip("recall:Q", format=".1%")],
        ).properties(height=320), use_container_width=True)
    st.caption("Points for this word with small n (hover) are noisy.")

# ---------------------------------------------------------------- patterns
with tab_patterns:
    curve = D["curve"]
    st.subheader("Forgetting curve by language")
    by_lang = weighted(curve, ["lang", "delta_bin"])
    by_lang["language"] = by_lang.lang.map(LANGS)
    st.altair_chart(
        alt.Chart(by_lang).mark_line(point=True).encode(
            x=alt.X("delta_bin:N", sort=DELTA_ORDER, title="Time since last practice"),
            y=alt.Y("recall:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False), title="Recall"),
            color=alt.Color("language:N"),
            tooltip=["language", "delta_bin", "n", alt.Tooltip("recall:Q", format=".1%")],
        ).properties(height=300), use_container_width=True)

    st.subheader("Does more practice slow forgetting?")
    st.caption("Same curve, split by how many times the learner had already seen the word. If practice "
               "strengthens memory, lines with more prior practice should sit higher at long gaps.")
    lang_p = st.selectbox("Language", ["all"] + list(LANGS),
                          format_func=lambda l: "All languages" if l == "all" else LANGS[l], key="plang")
    cv = curve if lang_p == "all" else curve[curve.lang == lang_p]
    by_hist = weighted(cv, ["hist_bin", "delta_bin"])
    st.altair_chart(
        alt.Chart(by_hist).mark_line(point=True).encode(
            x=alt.X("delta_bin:N", sort=DELTA_ORDER, title="Time since last practice"),
            y=alt.Y("recall:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False), title="Recall"),
            color=alt.Color("hist_bin:N", sort=HIST_ORDER, title="Times seen before"),
            tooltip=["hist_bin", "delta_bin", "n", alt.Tooltip("recall:Q", format=".1%")],
        ).properties(height=300), use_container_width=True)

    st.subheader("Which words are forgotten fastest?")
    st.caption("Drop = recall after a short gap (< 1 day) minus recall after a long gap (> 1 week). "
               "Half-life is a rough fit and often capped: the curves are too flat to pin it down.")
    c1, c2, c3 = st.columns(3)
    lang_w = c1.selectbox("Language", list(LANGS), format_func=LANGS.get, key="wlang")
    min_n = c2.slider("Min reviews in each gap group", 20, 500, 100, 10)
    pos = c3.multiselect("Part of speech", sorted(D["words"].pos.dropna().unique()))
    w = D["words"][(D["words"].lang == lang_w) & (D["words"].n_short >= min_n) & (D["words"].n_long >= min_n)]
    if pos:
        w = w[w.pos.isin(pos)]
    cols = ["label", "pos", "rows", "recall", "short", "long", "drop", "half_life_days"]
    cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in ["recall", "short", "long", "drop"]}
    cfg["half_life_days"] = st.column_config.NumberColumn("half-life (days, rough)", format="%.0f")
    cfg["short"] = st.column_config.NumberColumn("recall < 1 day", format="%.3f")
    cfg["long"] = st.column_config.NumberColumn("recall > 1 week", format="%.3f")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Biggest drop** ({len(w):,} words qualify)")
    c1.dataframe(w.nlargest(25, "drop")[cols], hide_index=True, column_config=cfg)
    c2.markdown("**Smallest drop**")
    c2.dataframe(w.nsmallest(25, "drop")[cols], hide_index=True, column_config=cfg)

    st.subheader("By part of speech")
    ps = (D["words"][D["words"].n_short.fillna(0) + D["words"].n_long.fillna(0) > 0]
          .assign(sh=lambda d: d.short * d.n_short, lo=lambda d: d.long * d.n_long)
          .groupby(["lang", "pos"])[["sh", "n_short", "lo", "n_long"]].sum())
    ps = ps[(ps.n_short > 2000) & (ps.n_long > 2000)]
    ps = pd.DataFrame({"recall < 1 day": ps.sh / ps.n_short, "recall > 1 week": ps.lo / ps.n_long}).reset_index()
    ps["drop"] = ps["recall < 1 day"] - ps["recall > 1 week"]
    ps["language"] = ps.lang.map(LANGS)
    st.altair_chart(
        alt.Chart(ps).mark_bar().encode(
            x=alt.X("drop:Q", title="Recall drop, short → long gap", axis=alt.Axis(format="%")),
            y=alt.Y("pos:N", sort="-x", title="Part of speech"),
            color="language:N", yOffset="language:N",
            tooltip=["language", "pos", alt.Tooltip("drop:Q", format=".1%"),
                     alt.Tooltip("recall < 1 day:Q", format=".1%"), alt.Tooltip("recall > 1 week:Q", format=".1%")],
        ).properties(height=500), use_container_width=True)
