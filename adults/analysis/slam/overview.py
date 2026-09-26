"""Overview tab for the SLAM explorer: what the dataset is and how to read it."""

import altair as alt
import pandas as pd
import streamlit as st

FORMATS = pd.DataFrame([
    ("reverse_translate", "Translate (type it)",
     "Shown a sentence in their native language, the learner types the translation in the language "
     "they are learning. Free typing, so spelling slips and word-choice differences count.",
     "Prompt “Yo soy un niño.” → learner types “I am a boy.”"),
    ("reverse_tap", "Translate (tap word bank)",
     "Same idea, but the learner builds the answer by tapping words from a shuffled word bank. "
     "No spelling, and the right words are on screen, so it is the easiest format.",
     "Prompt “Es sopa.” → taps [It] [is] [soup]"),
    ("listen", "Listen and type",
     "The learner hears a sentence in the language being learned and types what they heard. "
     "There is no written prompt, so the prompt field is empty.",
     "Audio “Monday afternoon” → types it"),
], columns=["format", "name", "what the learner does", "example"])

SESSIONS = pd.DataFrame([
    ("lesson", "A normal lesson in the course. Mostly new material, introduced in the order of Duolingo's skill tree."),
    ("practice", "A review session the learner chose (or was nudged into) to refresh skills they already studied."),
    ("test", "A short test to skip ahead ('test out') of skills the learner may already know."),
], columns=["session", "meaning"])

EX_FIELDS = pd.DataFrame([
    ("prompt", "What the learner was shown, in their native language. Empty for listen exercises.", "Yo soy un niño."),
    ("user", "Anonymous learner id. Same id across the train/dev/test files.", "XEinXf5+"),
    ("countries", "Country (or countries, separated by |) the learner used the app from.", "CO, or CA|US"),
    ("days", "Days since the learner started the course, with decimals. 0.5 = 12 hours in.", "13.693"),
    ("client", "Device: android, ios or web.", "web"),
    ("session", "Session type: lesson, practice or test (see below).", "lesson"),
    ("format", "Exercise type (see below).", "reverse_translate"),
    ("time", "Seconds the learner spent on the exercise. Sometimes missing (null).", "9"),
], columns=["field", "meaning", "example"])

TOK_FIELDS = pd.DataFrame([
    ("id", "Unique id: 8-char session id + 2-digit exercise number + 2-digit word position.", "DRihrVmh0102"),
    ("token", "One word of the correct answer.", "am"),
    ("part of speech", "Universal Dependencies part-of-speech tag (see glossary).", "VERB"),
    ("morphology", "Grammatical features of this word form, separated by |.",
     "Mood=Ind|Number=Sing|Person=1|Tense=Pres"),
    ("dependency label", "The word's grammatical role in the sentence.", "cop (linking verb)"),
    ("dependency head", "Position of the word this one attaches to (0 = the sentence root).", "4"),
    ("label", "1 = the learner got this word wrong, 0 = right. The thing to predict in the shared task.", "0"),
], columns=["field", "meaning", "example"])

UPOS = pd.DataFrame([
    ("NOUN", "noun", "boy, sopa"), ("VERB", "verb", "am, cocino"), ("PRON", "pronoun", "I, ella"),
    ("DET", "determiner / article", "a, the, el"), ("ADJ", "adjective", "good, grande"),
    ("ADP", "preposition", "from, de"), ("ADV", "adverb", "very, también"),
    ("PROPN", "proper noun", "Mexico, Pedro"), ("CONJ / CCONJ", "coordinating conjunction", "and, y"),
    ("SCONJ", "subordinating conjunction", "because, que"), ("NUM", "number", "three, tres"),
    ("INTJ", "interjection", "hello, hola"), ("AUX", "auxiliary verb", "have (in 'have eaten')"),
    ("PART", "particle", "not, to (in 'to go')"),
], columns=["tag", "meaning", "examples"])

DEPS = pd.DataFrame([
    ("ROOT", "main word of the sentence"), ("nsubj", "subject"), ("dobj / obj", "direct object"),
    ("det", "determiner of a noun"), ("amod", "adjective modifying a noun"),
    ("case", "preposition attached to its noun"), ("cop", "linking verb ('is' in 'she is a girl')"),
    ("nmod", "noun modifying another word"), ("advmod", "adverb modifier"),
    ("conj / cc", "conjoined item / the conjunction itself"), ("aux", "auxiliary verb"),
    ("compound", "part of a compound ('Monday' in 'Monday afternoon')"),
], columns=["label", "meaning"])

TRACK_INFO = pd.DataFrame([
    ("en_es", "English", "Spanish"), ("es_en", "Spanish", "English"), ("fr_en", "French", "English"),
], columns=["track", "learning", "native language"])


@st.cache_data
def track_summary(pq, tracks):
    """Built from the small per-track tables, so it doesn't load every track's words into memory."""
    rows = []
    for t in tracks:
        users = pd.read_parquet(pq / f"{t}_users.parquet")
        words = pd.read_parquet(pq / f"{t}_words.parquet")
        ex = pd.read_parquet(pq / f"{t}_exercises.parquet", columns=["user", "countries"])
        rows.append({
            "track": t,
            "learners": len(users),
            "exercises": len(ex),
            "words answered": int(users.n_tokens.sum()),
            "distinct words": len(words),
            "countries": ex.countries.astype(str).str.split("|").explode().nunique(),
            "error rate": (users.error_rate * users.n_tokens).sum() / users.n_tokens.sum(),
            "median exercises / learner": int(ex.groupby("user", observed=True).size().median()),
        })
    return TRACK_INFO.merge(pd.DataFrame(rows), on="track")


def raw_example(D, pq, track) -> str:
    """Rebuild the raw text block for one real exercise, as it appears in the files."""
    ex = D["ex"]
    ok = ex[(ex.format == "reverse_translate") & ex.prompt.notna()
            & ~ex.prompt.astype(str).str.contains(r"[¿?¡!]")]
    e = ok.iloc[len(ok) // 2]
    toks = pd.read_parquet(pq / f"{track}_tokens.parquet", filters=[("ex_id", "==", e.ex_id)])
    lines = [f"# prompt:{e.prompt}",
             f"# user:{e.user}  countries:{e.countries}  days:{e.days:.3f}  client:{e.client}  "
             f"session:{e.session}  format:{e.format}  time:{'null' if pd.isna(e.time) else int(e.time)}"]
    toks = toks.sort_values("pos_in_ex")
    w = {c: toks[c].astype(str).str.len().max() + 2 for c in ["token", "pos", "morph", "dep", "head"]}
    for t in toks.itertuples():
        lines.append(f"{t.token_id}  {t.token:<{w['token']}}{t.pos:<{w['pos']}}{t.morph:<{w['morph']}}"
                     f"{t.dep:<{w['dep']}}{t.head!s:<{w['head']}}{t.wrong}")
    return "\n".join(lines)


def render(D, track, pq, tracks):
    ex, tok = D["ex"], D["tok"]

    st.header("Duolingo SLAM 2018: what's in this dataset")
    st.markdown(
        "Every word of every exercise that about 6,400 Duolingo learners answered during their **first "
        "~30 days** on the app, marked **right or wrong**. Duolingo released it for a research "
        "competition, the *Second Language Acquisition Modeling* (SLAM) shared task, where teams had to "
        "predict which words a learner would get wrong in the future from their history so far.")

    st.subheader("At a glance")
    summ = track_summary(pq, tuple(tracks))
    st.dataframe(summ, hide_index=True, use_container_width=True, column_config={
        "error rate": st.column_config.NumberColumn(format="%.3f"),
        "learners": st.column_config.NumberColumn(format="%d"),
        "exercises": st.column_config.NumberColumn(format="%d"),
        "words answered": st.column_config.NumberColumn(format="%d"),
    })
    st.caption("Track names follow the files: *language learned*_*native language*. "
               "Error rate = share of words answered wrong.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Source")
        st.markdown(
            "- **Paper:** B. Settles, C. Brust, E. Gustafson, M. Hagiwara, N. Madnani (2018). "
            "*Second Language Acquisition Modeling.* NAACL-HLT Workshop on Innovative Use of NLP for "
            "Building Educational Applications (BEA).\n"
            "- **Data:** Harvard Dataverse, [doi:10.7910/DVN/8SWHNO](https://doi.org/10.7910/DVN/8SWHNO), "
            "version 4.0 (Feb 2019). Download requires a short guestbook form.\n"
            "- **License:** CC BY-NC 4.0: free for non-commercial use with credit.\n"
            "- **Shared task:** attracted 15 teams. The paper describes 7M+ words from 6k+ learners.")
    with c2:
        st.subheader("Who is in it")
        st.markdown(
            "- New Duolingo users, followed for roughly their **first 30 days**.\n"
            "- Three courses (tracks), listed in the table above.\n"
            "- Nothing about the learners except an anonymous id, the country they used the app from, "
            "and their device. No age, education, goals or native-language proficiency.\n"
            "- Self-selected: people who chose to install Duolingo and kept at it long enough to appear.")

    st.divider()
    st.subheader(f"The learners in this track ({track})")
    c1, c2, c3 = st.columns(3)
    with c1:
        countries = ex.countries.astype(str).str.split("|").explode()
        top = (countries.value_counts(normalize=True).head(12).rename("share")
               .rename_axis("country").reset_index())
        st.markdown("**Top countries** (share of exercises)")
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("country:N", sort="-x", title=None),
            tooltip=["country", alt.Tooltip("share:Q", format=".1%")]).properties(height=280),
            use_container_width=True)
    with c2:
        per = ex.groupby("user", observed=True).agg(exercises=("ex_id", "size"), last_day=("days", "max"))
        st.markdown("**Exercises per learner**")
        st.altair_chart(alt.Chart(per).mark_bar().encode(
            x=alt.X("exercises:Q", bin=alt.Bin(maxbins=40), title="Exercises in the ~30 days"),
            y=alt.Y("count():Q", title="Learners")).properties(height=280), use_container_width=True)
    with c3:
        dev = ex.client.value_counts(normalize=True).rename("share").rename_axis("device").reset_index()
        st.markdown("**Device** (share of exercises)")
        st.altair_chart(alt.Chart(dev).mark_arc(innerRadius=50).encode(
            theta="share:Q", color="device:N",
            tooltip=["device", alt.Tooltip("share:Q", format=".1%")]).properties(height=280),
            use_container_width=True)

    st.divider()
    st.subheader("How the data is organised")
    st.markdown(
        "**Learner → session → exercise → word.** A learner does sessions; each session has several "
        "exercises (one sentence each); each exercise lists the words of the correct answer, each marked "
        "right or wrong.")
    st.markdown("**Exercise types**")
    fmt = tok.groupby("format", observed=True).agg(share=("wrong", "size"), error_rate=("wrong", "mean"))
    fmt["share"] = fmt.share / fmt.share.sum()
    f = FORMATS.merge(fmt.reset_index(), on="format", how="left")
    st.dataframe(f, hide_index=True, use_container_width=True, column_config={
        "share": st.column_config.NumberColumn("share of words", format="%.2f"),
        "error_rate": st.column_config.NumberColumn("error rate", format="%.3f"),
        "what the learner does": st.column_config.TextColumn(width="large"),
    })
    st.markdown("**Session types**")
    ses = ex.session.value_counts(normalize=True).rename("share of exercises").reset_index()
    st.dataframe(SESSIONS.merge(ses, on="session"), hide_index=True, use_container_width=True,
                 column_config={"share of exercises": st.column_config.NumberColumn(format="%.3f"),
                                "meaning": st.column_config.TextColumn(width="large")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fields for each exercise**")
        st.dataframe(EX_FIELDS, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Fields for each word**")
        st.dataframe(TOK_FIELDS, hide_index=True, use_container_width=True)
        st.caption("Part of speech, morphology and dependencies were produced automatically by a parser, "
                   "not by hand, so they are sometimes wrong (for example on questions).")

    st.markdown("**What one exercise looks like in the raw files** (real example from this track)")
    st.code(raw_example(D, pq, track), language=None)
    st.caption("First line: what the learner saw. Second: exercise metadata. Then one line per word of the "
               "correct answer; the last number is the right (0) / wrong (1) label.")

    st.divider()
    st.subheader("How 'right' and 'wrong' are decided")
    st.markdown(
        "- Duolingo accepts many correct answers for most exercises. The learner's response is matched to "
        "the **correct answer closest to what they typed**, and that answer is what appears in the data.\n"
        "- Each word of that answer is marked **wrong** if the learner's response was missing it, misspelled "
        "it or had a different word there; otherwise **right**.\n"
        "- **The learner's actual typed answer is not in the dataset.** You see which words they missed, "
        "never what they wrote instead, and extra words they added are invisible.\n"
        "- The dataset changelog notes a punctuation-alignment bug that affected ~7% of tokens in earlier "
        "versions; version 4.0 (used here) fixes the parses.")

    st.subheader("Train / dev / test split")
    sp = ex.split.value_counts(normalize=True)
    st.markdown(
        f"The files were split **by time within each learner**: roughly the first {sp.get('train', 0):.0%} "
        f"of each learner's exercises are *train*, the next {sp.get('dev', 0):.0%} *dev* and the last "
        f"{sp.get('test', 0):.0%} *test* (checked: for every learner, train ends before dev starts and dev "
        "before test). Answers for dev and test sit in separate `.key` files, because the competition hid "
        "them. **This explorer joins all three back together**, so each learner has one continuous history.")

    st.subheader("What this explorer did to it")
    st.markdown(
        "1. `prepare.py`: parsed the raw text files into tables (one row per exercise, one per word), "
        "joined in the `.key` answers, and sorted each learner's exercises by time.\n"
        "2. `patterns.py`: fitted a baseline model predicting each answer from learner ability (overall "
        "and per exercise type), word difficulty and exercise type; kept the leftover 'better or worse "
        "than expected' part; and correlated it between words across learners (the *Patterns* tab).\n"
        "3. Words are compared lower-cased and as written (*soy* and *es* are different words, not both "
        "*ser*).")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Good for")
        st.markdown(
            "- Which words, grammar and exercise types trip beginners up\n"
            "- How accuracy changes over the first month\n"
            "- Differences between devices, countries, exercise types\n"
            "- Whether mastery of one word goes with another\n"
            "- Predicting a learner's next mistakes (the original task)")
    with c2:
        st.subheader("Not good for")
        st.markdown(
            "- Anything past the first ~30 days (long-term retention, fluency)\n"
            "- What learners actually wrote, or error types (no responses)\n"
            "- Effects of age, education or motivation (not recorded)\n"
            "- Children specifically (age is not recorded)\n"
            "- Causal claims: Duolingo decides the order of material, so what's learned together is "
            "partly just what's taught together")

    st.subheader("Glossary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Part-of-speech tags**")
        counts = tok.pos.value_counts(normalize=True).rename("share")
        u = UPOS.assign(share=UPOS.tag.str.split(" / ").map(lambda ts: sum(counts.get(x, 0) for x in ts)))
        st.dataframe(u, hide_index=True, use_container_width=True,
                     column_config={"share": st.column_config.NumberColumn("share of words", format="%.3f")})
    with c2:
        st.markdown("**Common dependency labels**")
        st.dataframe(DEPS, hide_index=True, use_container_width=True)
