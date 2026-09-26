"""Overview tab for the spaced-repetition explorer: what the dataset is and how to read it."""

import re
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

FIELDS = pd.DataFrame([
    ("p_recall", "recall",
     "Share of this session's exercises containing the word that the learner got right "
     "(= session_correct / session_seen)."),
    ("timestamp", "ts / time", "When the session happened (UNIX seconds; converted to a date here)."),
    ("delta", "delta_days",
     "Time since the learner last practised this word, in seconds (converted to days here). "
     "The 'gap' in every forgetting curve."),
    ("user_id", "user", "Anonymous learner id."),
    ("learning_language", "lang", "Language being learned."),
    ("ui_language", "ui_lang", "Language of the app interface; presumably the learner's native language."),
    ("lexeme_id", "lexeme_id", "Duolingo's internal id for the word (a hash)."),
    ("lexeme_string", "lexeme_string",
     "The word in Duolingo's tag format: surface form / lemma <part of speech> <grammar tags> (see below)."),
    ("history_seen", "history_seen", "Times the learner had seen this word before this session."),
    ("history_correct", "history_correct", "Times they had got it right before this session."),
    ("session_seen", "session_seen", "Times the word came up in this session."),
    ("session_correct", "session_correct", "Times they got it right in this session."),
], columns=["original column", "name here", "meaning"])

LANGS = {"de": "German", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"}


@st.cache_data
def facts(traces: str):
    con = duckdb.connect()
    t = f"'{traces}'"
    pairs = con.execute(f"""
        SELECT lang, ui_lang, count(*) AS reviews, count(DISTINCT user) AS learners,
               count(DISTINCT lexeme_id) AS words, avg(recall) AS recall
        FROM {t} GROUP BY ALL ORDER BY reviews DESC""").df()
    seen = con.execute(f"""
        SELECT least(session_seen, 6) AS times, count(*) AS n FROM {t} GROUP BY 1 ORDER BY 1""").df()
    recall = con.execute(f"""
        SELECT round(recall, 2) AS recall, count(*) AS n FROM {t} GROUP BY 1 ORDER BY 1""").df()
    gaps = con.execute(f"""
        SELECT delta_bin, count(*) AS n FROM {t} GROUP BY 1""").df()
    hist = con.execute(f"""
        SELECT min(history_seen) AS min_hist, median(history_seen) AS med_hist,
               max(history_seen) AS max_hist, median(delta_days) AS med_gap
        FROM {t}""").df().iloc[0]
    per_day = con.execute(f"""
        SELECT date_trunc('day', time) AS day, count(*) AS reviews, count(DISTINCT user) AS learners
        FROM {t} GROUP BY 1 ORDER BY 1""").df()
    example = con.execute(f"""
        SELECT * FROM {t} WHERE lang = 'es' AND session_seen = 3 AND history_seen BETWEEN 5 AND 12
               AND delta_days BETWEEN 1 AND 5 LIMIT 1""").df().iloc[0]
    return pairs, seen, recall, gaps, hist, per_day, example


@st.cache_data
def tag_glossary(words: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(HERE / "lexeme_reference.txt", sep=r"\s{2,}", engine="python",
                      names=["tag", "kind", "meaning"])
    counts = {}
    for s, n in zip(words.lexeme_string, words.rows):
        for tag in re.findall(r"<([^>*]+)>", s.split("/", 1)[1]):
            counts[tag] = counts.get(tag, 0) + n
    ref["reviews"] = ref.tag.map(counts).fillna(0).astype(int)
    return ref[ref.reviews > 0].sort_values(["kind", "reviews"], ascending=[True, False])


def render(D, traces, delta_order):
    users, words = D["users"], D["words"]
    pairs, seen, recall, gaps, hist, per_day, ex = facts(str(traces))

    st.header("Duolingo spaced repetition (2016): what's in this dataset")
    st.markdown(
        "**12.9 million word reviews** by **115k Duolingo learners** over two weeks in 2013. Each row "
        "says: *this learner practised this word in this session; it had been this long since they last "
        "practised it; this is their track record with it; this is how they did now.* Duolingo released "
        "it with a paper introducing **half-life regression**, the model it built to decide when each "
        "word should come back for review.")

    m = st.columns(6)
    m[0].metric("Word reviews", f"{users.rows.sum() / 1e6:.1f}M")
    m[1].metric("Learners", f"{len(users):,}")
    m[2].metric("Sessions", f"{users.sessions.sum():,}")
    m[3].metric("Distinct words", f"{len(words):,}")
    m[4].metric("Language pairs", len(pairs))
    m[5].metric("Average recall", f"{(users.recall * users.rows).sum() / users.rows.sum():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Source")
        st.markdown(
            "- **Paper:** B. Settles and B. Meeder (2016). *A Trainable Spaced Repetition Model for "
            "Language Learning.* Proceedings of ACL, pages 1848–1858. "
            "[doi:10.18653/v1/P16-1174](https://doi.org/10.18653/v1/P16-1174)\n"
            "- **Data:** Harvard Dataverse, [doi:10.7910/DVN/N8XJME](https://doi.org/10.7910/DVN/N8XJME), "
            "one gzipped CSV (379 MB). No signup needed.\n"
            "- **Code:** [github.com/duolingo/halflife-regression]"
            "(https://github.com/duolingo/halflife-regression) (MIT), with the model, baselines and a "
            "reference for the word tags.\n"
            "- **License:** CC BY-NC 4.0: free for non-commercial use with credit.")
    with c2:
        st.subheader("The idea behind it")
        st.markdown(
            "- **Spaced repetition:** review something just before you'd forget it. Each review makes "
            "the memory last longer, so gaps between reviews can grow.\n"
            "- **Half-life:** how long until the chance of remembering a word drops to 50%. The model "
            "assumes recall = 2^(−gap / half-life): right after practice ≈ 100%, after one half-life 50%, "
            "after two 25%.\n"
            "- **Half-life regression:** learn each learner-word half-life from features like how often "
            "they've seen it and got it right, plus the word itself. The paper reports testing it in "
            "Duolingo's practice sessions.")

    st.divider()
    st.subheader("Who is in it")
    c1, c2 = st.columns([3, 2])
    with c1:
        p = pairs.copy()
        p["course"] = p.lang.map(LANGS) + " for " + p.ui_lang.map(LANGS) + " speakers"
        st.dataframe(p[["course", "reviews", "learners", "words", "recall"]], hide_index=True,
                     use_container_width=True,
                     column_config={"recall": st.column_config.NumberColumn(format="%.3f")})
        st.caption("A course = language learned × app language. 'Words' counts distinct word tags.")
    with c2:
        st.altair_chart(alt.Chart(per_day).mark_bar().encode(
            x=alt.X("day:T", title=None), y=alt.Y("reviews:Q", title="Reviews per day"),
            tooltip=["day:T", "reviews", "learners"]).properties(height=250), use_container_width=True)
        st.caption(f"{per_day.day.min():%d %b %Y} – {per_day.day.max():%d %b %Y}. "
                   "Only learners and words active in this window appear.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sessions per learner in the two weeks**")
        st.altair_chart(alt.Chart(users.assign(s=users.sessions.clip(upper=60))).mark_bar().encode(
            x=alt.X("s:Q", bin=alt.Bin(step=2), title="Sessions (60+ grouped)"),
            y=alt.Y("count():Q", title="Learners")).properties(height=220), use_container_width=True)
        st.caption(f"Median {users.sessions.median():.0f} sessions. Most learners appear only briefly.")
    with c2:
        st.markdown("**Nothing else is known about learners:** no age, country, device, level or goals. "
                    "Only the language pair and the review history.")

    st.divider()
    st.subheader("What one row means")
    st.dataframe(FIELDS, hide_index=True, use_container_width=True,
                 column_config={"meaning": st.column_config.TextColumn(width="large")})
    st.markdown("**A real row, read out loud**")
    gap = ex.delta_days
    gap_txt = f"{gap:.1f} days"
    st.code(
        f"p_recall={ex.recall:.3f}  timestamp={ex.ts}  delta={int(round(gap * 86400))}  user_id={ex.user}\n"
        f"learning_language={ex.lang}  ui_language={ex.ui_lang}  lexeme_string={ex.lexeme_string}\n"
        f"history_seen={ex.history_seen}  history_correct={ex.history_correct}  "
        f"session_seen={ex.session_seen}  session_correct={ex.session_correct}", language=None)
    st.markdown(
        f"On **{ex.time:%d %b %Y at %H:%M}**, learner **{ex.user}** ({LANGS[ex.ui_lang]} speaker learning "
        f"{LANGS[ex.lang]}) practised **{ex.surface}** (a form of *{ex.lemma}*). They had last practised it "
        f"**{gap_txt} earlier**. Before this session they had seen it **{ex.history_seen} times** and got it "
        f"right **{ex.history_correct}** of those. In this session it came up **{ex.session_seen} times** and "
        f"they got **{ex.session_correct}** right, so recall = {ex.session_correct}/{ex.session_seen} "
        f"= **{ex.recall:.2f}**.")

    st.subheader("How to read the word tags")
    st.markdown(
        "Format: `surface form / lemma <part of speech> <grammar tags…>`. Examples from the Duolingo docs: "
        "`escribe/escribir<vblex><pri><p3><sg>` = *escribe*, a form of the verb *escribir*, present "
        "indicative, 3rd person singular. `es/ser<vbser><pri><p3><sg>` = *es*, the verb 'to be'.\n\n"
        "Tags starting with `*` are **wildcards**: `<*sf>` means *any form* of the word (all conjugations "
        "count as the same item), `<*numb>` means singular or plural. So some rows track a specific "
        "form and others a whole word family.")
    gl = tag_glossary(words)
    kinds = ["POS", "tense", "person", "number", "gender", "case", "def", "adjective", "animacy", "other"]
    k = st.segmented_control("Tag type", [x for x in kinds if x in set(gl.kind)], default="POS")
    st.dataframe(gl[gl.kind == k][["tag", "meaning", "reviews"]], hide_index=True,
                 use_container_width=True)
    st.caption("From Duolingo's lexeme_reference.txt; 'reviews' counts rows whose word carries the tag.")

    st.divider()
    st.subheader("Quirks to know before reading any chart")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. 'Recall' is mostly all-or-nothing**")
        st.altair_chart(alt.Chart(seen.assign(t=seen.times.map(lambda x: "6+" if x >= 6 else str(x))))
                        .mark_bar().encode(x=alt.X("t:N", title="Times the word came up in the session"),
                                           y=alt.Y("n:Q", title="Rows")).properties(height=200),
                        use_container_width=True)
        one = seen.loc[seen.times == 1, "n"].sum() / seen.n.sum()
        perfect = recall.loc[recall.recall == 1, "n"].sum() / recall.n.sum()
        st.caption(f"{one:.0%} of rows test the word once, so recall is just 0 or 1 there. "
                   f"{perfect:.0%} of all rows are a perfect 1.0.")
    with c2:
        st.markdown("**2. Every row is a repeat**")
        st.markdown(
            f"history_seen is at least **{int(hist.min_hist)}** in every row (median {hist.med_hist:.0f}, "
            f"max {int(hist.max_hist):,}). The first time a learner meets a word is never in the data, "
            "so this shows *reviewing*, not first learning.")
        st.markdown("**Gap since last practice**")
        g = gaps.set_index("delta_bin").reindex(delta_order).reset_index()
        st.altair_chart(alt.Chart(g).mark_bar().encode(
            x=alt.X("delta_bin:N", sort=delta_order, title=None), y=alt.Y("n:Q", title="Rows")
        ).properties(height=160), use_container_width=True)
        st.caption(f"Median gap {hist.med_gap:.1f} days. Gaps can be years: they count back to before "
                   "the two-week window.")
    with c3:
        st.markdown("**3. Duolingo chose when to test**")
        st.markdown(
            "Reviews weren't random: Duolingo's scheduler brought words back when it guessed they needed practice, and learners chose when to do "
            "sessions. Words reviewed after long gaps are ones the system expected to be remembered. "
            "So **forgetting curves here look flatter than real memory decay**, and differences between "
            "words partly reflect the scheduler, not just memory.")

    st.subheader("What this explorer did to it")
    st.markdown(
        "`prepare.py` loads the CSV, converts times to dates and gaps to days, splits `lexeme_string` into "
        "surface form, lemma and part of speech, groups gaps and prior practice into bins, sorts by learner "
        "and time, and saves per-learner, per-word and per-curve summaries. For each word it also fits "
        "recall = a × 2^(−gap / half-life); because curves are so flat, those half-lives are rough and "
        "often hit the cap, so the app ranks words by the plain recall drop between short (< 1 day) and "
        "long (> 1 week) gaps.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Good for")
        st.markdown(
            "- Comparing words, parts of speech and languages on how well they're retained\n"
            "- Seeing how a real learner's reviews are spread over time\n"
            "- Testing spaced-repetition models (what the paper did)\n"
            "- How recall relates to prior practice and accuracy history")
    with c2:
        st.subheader("Not good for")
        st.markdown(
            "- True forgetting speed (scheduler picks the gaps)\n"
            "- First exposure to words (never in the data)\n"
            "- Anything about who learners are (no demographics)\n"
            "- Long-term progress (two-week window)\n"
            "- Which exercise or sentence the word appeared in (not recorded)")
