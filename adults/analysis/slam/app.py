"""SLAM explorer: look at one learner, one word, or links between words.

Run:  uv run streamlit run app.py
Needs prepare.py and patterns.py to have been run first.
"""

import html
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PQ = Path(__file__).resolve().parents[2] / "data" / "slam" / "parquet"
TRACKS = {
    "en_es": "Spanish speakers learning English",
    "es_en": "English speakers learning Spanish",
    "fr_en": "English speakers learning French",
}
FORMAT_LABEL = {
    "reverse_translate": "translate (type it)",
    "reverse_tap": "translate (tap word bank)",
    "listen": "listen & type",
}

st.set_page_config(page_title="SLAM explorer", layout="wide")


@st.cache_resource
def load(track: str):
    ex = pd.read_parquet(PQ / f"{track}_exercises.parquet")
    tok = pd.read_parquet(PQ / f"{track}_tokens.parquet",
                          columns=["ex_id", "pos_in_ex", "token", "word", "pos", "wrong"])
    tok = tok.merge(ex[["ex_id", "user", "days", "format", "prompt"]], on="ex_id")
    return {
        "ex": ex,
        "tok": tok,
        "tok_by_user": tok.groupby("user", observed=True).indices,
        "users": pd.read_parquet(PQ / f"{track}_users.parquet"),
        "words": pd.read_parquet(PQ / f"{track}_words.parquet"),
        "uw": pd.read_parquet(PQ / f"{track}_user_word.parquet"),
        "pairs": pd.read_parquet(PQ / f"{track}_pairs.parquet"),
    }


def pairs_for(pairs: pd.DataFrame, word: str) -> pd.DataFrame:
    a = pairs[pairs.word_a == word].rename(columns={"word_b": "other"})
    b = pairs[pairs.word_b == word].rename(columns={"word_a": "other"})
    cols = ["other", "n_shared", "link", "z", "same_ex_share", "same_first_session"]
    return pd.concat([a[cols], b[cols]])


def pair_filters(key: str):
    c1, c2, c3 = st.columns(3)
    min_shared = c1.slider("Min learners who saw both", 30, 1000, 200, 10, key=f"{key}_n")
    max_ex = c2.slider("Max same-exercise share", 0.0, 1.0, 0.05, 0.05, key=f"{key}_ex",
                       help="How often the two words sit in the same sentence. A botched sentence "
                            "makes all its words wrong together, which fakes a link.")
    max_fs = c3.slider("Max 'taught together' share", 0.0, 1.0, 1.0, 0.05, key=f"{key}_fs",
                       help="Share of learners who met both words for the first time in the same "
                            "session. High = Duolingo teaches them in the same lesson.")
    return lambda p: p[(p.n_shared >= min_shared) & (p.same_ex_share <= max_ex)
                       & (p.same_first_session <= max_fs)]


def pair_table(p: pd.DataFrame, n=50):
    show = p.sort_values("z", ascending=False).head(n).copy()
    show["same_ex_share"] = (show["same_ex_share"] * 100).round(0)
    show["same_first_session"] = (show["same_first_session"] * 100).round(0)
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        column_config={
            "link": st.column_config.NumberColumn("link", format="%.3f",
                                                  help="Correlation of 'better than expected' across learners"),
            "z": st.column_config.NumberColumn("strength (z)", format="%.1f",
                                               help="Link scaled by sample size. Above ~4.5 is unlikely to be noise "
                                                    "given how many pairs are tested."),
            "n_shared": "learners (both)",
            "same_ex_share": st.column_config.NumberColumn("same sentence %", format="%d"),
            "same_first_session": st.column_config.NumberColumn("taught together %", format="%d"),
        })


# ---------------------------------------------------------------- sidebar
track = st.sidebar.selectbox("Track", list(TRACKS), format_func=lambda t: f"{t}: {TRACKS[t]}")
D = load(track)
st.sidebar.caption(
    f"{D['users'].shape[0]:,} learners · {len(D['ex']):,} exercises · "
    f"{len(D['tok']):,} words answered · first ~30 days on Duolingo (2017)")
st.sidebar.markdown(
    "**How to read 'better than expected'**  \n"
    "A model predicts each answer from the learner's overall ability (per exercise type), "
    "the word's difficulty and the exercise type. A learner is *better than expected* on a word "
    "when they get it right more often than that predicts. Links between words use this, "
    "so 'strong learners know everything' doesn't show up as a pattern.")

tab_student, tab_word, tab_patterns = st.tabs(["Student", "Word", "Patterns"])

# ---------------------------------------------------------------- student
with tab_student:
    users = D["users"].sort_values("n_tokens", ascending=False)
    if "student" not in st.session_state or st.session_state.get("student_track") != track:
        st.session_state.student = users.user.iloc[0]
        st.session_state.student_track = track
    c1, c2 = st.columns([3, 1])
    if c2.button("🎲 Random student", use_container_width=True):
        st.session_state.student = users.user.sample(1).iloc[0]
    ulist = users.user.tolist()
    n_answered = dict(zip(users.user, users.n_tokens))
    user = c1.selectbox("Learner", ulist, index=ulist.index(st.session_state.student),
                        format_func=lambda u: f"{u}  ({n_answered[u]:,} words answered)")
    st.session_state.student = user

    uex = D["ex"][D["ex"].user == user].sort_values("days")
    utok = D["tok"].iloc[D["tok_by_user"][user]].sort_values(["days", "ex_id", "pos_in_ex"])
    urow = D["users"].set_index("user").loc[user]

    m = st.columns(6)
    m[0].metric("Country", uex.countries.iloc[0])
    m[1].metric("Device", uex.client.mode().iloc[0])
    m[2].metric("Days active", f"{uex.days.astype(int).nunique()} of {int(np.ceil(uex.days.max()))}")
    m[3].metric("Exercises", f"{len(uex):,}")
    m[4].metric("Error rate", f"{urow.error_rate:.1%}")
    pct = (D["users"].ability < urow.ability).mean()
    m[5].metric("Ability percentile", f"{pct:.0%}")

    daily = utok.assign(day=utok.days.astype(int)).groupby(["day", "format"], observed=True).agg(
        words=("wrong", "size"), error_rate=("wrong", "mean")).reset_index()
    daily["format"] = daily["format"].map(FORMAT_LABEL)
    st.altair_chart(
        alt.Chart(daily).mark_circle(opacity=0.8).encode(
            x=alt.X("day:Q", title="Day since starting", scale=alt.Scale(domain=[0, 30])),
            y=alt.Y("error_rate:Q", title="Share of words wrong", axis=alt.Axis(format="%")),
            size=alt.Size("words:Q", title="Words answered"),
            color=alt.Color("format:N", title="Exercise type"),
            tooltip=["day", "format", "words", alt.Tooltip("error_rate:Q", format=".1%")],
        ).properties(height=260),
        use_container_width=True)

    st.subheader("Exercise log")
    days = sorted(uex.days.astype(int).unique())
    day = st.select_slider("Day", options=days, value=days[0])
    day_ex = uex[uex.days.astype(int) == day]
    day_tok = utok[utok.days.astype(int) == day]
    st.caption(f"{len(day_ex)} exercises on day {day}. Prompt is what the learner was shown "
               "(listen exercises have no written prompt). The learner's own typed answer is not in the "
               "dataset: the last column is the correct answer closest to what they typed, with each word "
               "green if their answer had it and red if it was missing, misspelled or replaced.")
    rows = []
    for ex_id, g in day_tok.groupby("ex_id", sort=False):
        e = day_ex[day_ex.ex_id == ex_id].iloc[0]
        words = " ".join(
            f'<span class="{"bad" if w else "ok"}">{html.escape(t)}</span>'
            for t, w in zip(g.token, g.wrong))
        prompt = html.escape(e.prompt) if isinstance(e.prompt, str) else "<i>(audio)</i>"
        t = f"{int(e.time)}s" if pd.notna(e.time) else "–"
        rows.append(f"<tr><td>{e.days:.2f}</td><td>{FORMAT_LABEL.get(e.format, e.format)}</td>"
                    f"<td>{e.ex_id[:8]}</td><td>{t}</td><td>{prompt}</td><td>{words}</td></tr>")
    st.html(
        "<style>.log{width:100%;border-collapse:collapse;font-size:14px}"
        ".log td,.log th{padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.25);text-align:left}"
        ".ok{background:rgba(34,160,90,.18);padding:1px 4px;border-radius:4px}"
        ".bad{background:rgba(220,50,50,.35);padding:1px 4px;border-radius:4px;font-weight:600}</style>"
        "<table class='log'><tr><th>day</th><th>type</th><th>session</th><th>time</th>"
        "<th>prompt</th><th>closest correct answer (red = learner missed it)</th></tr>" + "".join(rows) + "</table>")

    with st.expander("This learner's hardest and easiest words (relative to expectation)"):
        uw = D["uw"][(D["uw"].user == user) & (D["uw"].n_seen >= 3)]
        c1, c2 = st.columns(2)
        cols = ["word", "n_seen", "n_wrong", "mean_resid"]
        c1.markdown("**Worse than expected**")
        c1.dataframe(uw.nlargest(15, "mean_resid")[cols], hide_index=True)
        c2.markdown("**Better than expected**")
        c2.dataframe(uw.nsmallest(15, "mean_resid")[cols], hide_index=True)

# ---------------------------------------------------------------- word
with tab_word:
    words = D["words"].sort_values("n_users", ascending=False)
    wlist = words.word.tolist()
    default = {"en_es": "morning", "es_en": "mañana", "fr_en": "matin"}[track]
    word = st.selectbox("Word", wlist, index=wlist.index(default) if default in wlist else 0)
    wr = words.set_index("word").loc[word]
    m = st.columns(4)
    m[0].metric("Learners who saw it", f"{int(wr.n_users):,}")
    m[1].metric("Times answered", f"{int(wr.n_tokens):,}")
    m[2].metric("Error rate", f"{wr.error_rate:.1%}")
    m[3].metric("Median first-seen day", f"{wr.median_first_day:.1f}")

    wt = D["tok"][D["tok"].word == word]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Error rate by exercise type**")
        byf = wt.groupby("format", observed=True).wrong.agg(["size", "mean"]).reset_index()
        byf["format"] = byf["format"].map(FORMAT_LABEL)
        st.dataframe(byf.rename(columns={"size": "answers", "mean": "error rate"}),
                     hide_index=True, column_config={"error rate": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.markdown("**Prompts it appears in**")
        st.caption("A high error rate on one prompt often means that prompt has several valid "
                   "translations and the learner picked a different one from the reference.")
        byp = wt.groupby(wt.prompt.fillna("(audio)")).wrong.agg(["size", "mean"])
        st.dataframe(byp.sort_values("size", ascending=False).head(15)
                     .rename(columns={"size": "answers", "mean": "error rate"}),
                     column_config={"error rate": st.column_config.NumberColumn(format="%.3f")})

    st.markdown("**Words whose mastery goes with this one**")
    filt = pair_filters("word")
    pw = filt(pairs_for(D["pairs"], word))
    if pw.empty:
        st.info("No pairs pass the filters (this word may be too rare for pair links).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Strongest positive links")
            pair_table(pw, 20)
        with c2:
            st.caption("Strongest negative links")
            pair_table(pw.assign(z=-pw.z), 20)

# ---------------------------------------------------------------- patterns
with tab_patterns:
    st.markdown(
        "Every pair of common words, ranked by how strongly *doing better than expected* on one goes "
        "with doing better than expected on the other. Use the filters to strip out the two usual "
        "confounds: words in the same sentence, and words taught in the same lesson.")
    filt = pair_filters("pat")
    q = st.text_input("Only pairs containing (optional)", "")
    p = filt(D["pairs"])
    if q:
        p = p[(p.word_a == q.lower()) | (p.word_b == q.lower())]
    st.caption(f"{len(p):,} pairs pass the filters · {(p.z > 4.5).sum():,} with strength above 4.5")
    pair_table(p, 100)

    st.divider()
    st.subheader("Look inside one pair")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Word A", wlist, index=wlist.index(default) if default in wlist else 0, key="pa")
    b_default = {"en_es": "evening", "es_en": "noche", "fr_en": "soir"}[track]
    b = c2.selectbox("Word B", wlist, index=wlist.index(b_default) if b_default in wlist else 1, key="pb")
    uw = D["uw"]
    ua = uw[uw.word == a].set_index("user")
    ub = uw[uw.word == b].set_index("user")
    both = ua.join(ub, lsuffix="_a", rsuffix="_b", how="inner")
    if len(both) < 20:
        st.info(f"Only {len(both)} learners saw both words, which is too few to say anything.")
    else:
        both["a_group"] = np.where(both.mean_resid_a < 0, f"better than expected on '{a}'",
                                   f"worse than expected on '{a}'")
        g = both.groupby("a_group").agg(
            learners=("n_seen_b", "size"),
            b_error_rate=("n_wrong_b", "sum"), b_seen=("n_seen_b", "sum"),
            b_expected=("mean_pred_b", "mean"), b_first_wrong=("first_wrong_b", "mean"))
        g["b_error_rate"] = g.b_error_rate / g.b_seen
        g = g.drop(columns="b_seen")
        st.caption(f"{len(both):,} learners saw both. Split them by how they did on '{a}', then look "
                   f"at '{b}'. If the pattern is real, the two groups differ on '{b}' by more than the "
                   "model's expectation (which already accounts for ability).")
        st.dataframe(g, column_config={
            "b_error_rate": st.column_config.NumberColumn(f"error rate on '{b}'", format="%.3f"),
            "b_expected": st.column_config.NumberColumn(f"expected on '{b}'", format="%.3f"),
            "b_first_wrong": st.column_config.NumberColumn(f"wrong first time on '{b}'", format="%.3f"),
        })
        r = both[["mean_resid_a", "mean_resid_b"]].corr().iloc[0, 1]
        st.metric("Link", f"{r:.3f}")
        chart_df = both.reset_index()[["user", "mean_resid_a", "mean_resid_b", "n_seen_a", "n_seen_b"]]
        st.altair_chart(
            alt.Chart(chart_df).mark_circle(opacity=0.35, size=25).encode(
                x=alt.X("mean_resid_a:Q", title=f"'{a}': worse than expected →"),
                y=alt.Y("mean_resid_b:Q", title=f"'{b}': worse than expected →"),
                tooltip=["user", "n_seen_a", "n_seen_b"],
            ).properties(height=380),
            use_container_width=True)
