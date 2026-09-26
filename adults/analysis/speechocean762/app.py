"""speechocean762 explorer: Mandarin speakers reading English aloud.

Run:  uv run streamlit run app.py --server.port 8503
Needs prepare.py to have been run first.
"""

import html
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PQ = Path(__file__).resolve().parents[2] / "data" / "speechocean762" / "parquet"
GROUPS = ["child (6-12)", "teen (13-15)", "adult (19+)"]
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
# rough plain-English hints for ARPAbet symbols
HINT = {
    "AA": "f-a-ther", "AE": "c-a-t", "AH": "b-u-t / a-bout", "AO": "c-au-ght", "AW": "c-ow",
    "AY": "b-i-te", "EH": "b-e-d", "ER": "b-ir-d", "EY": "b-ai-t", "IH": "b-i-t", "IY": "b-ee-t",
    "OW": "b-oa-t", "OY": "b-oy", "UH": "b-oo-k", "UW": "b-oo-t", "B": "b", "CH": "ch-ur-ch",
    "D": "d", "DH": "th-is", "F": "f", "G": "g", "HH": "h", "JH": "j-udge", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "si-ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t",
    "TH": "th-in", "V": "v", "W": "w", "Y": "y-es", "Z": "z", "ZH": "mea-s-ure",
}

st.set_page_config(page_title="Pronunciation explorer", layout="wide")
st.html("""<style>
.w{display:inline-block;margin:4px 10px 10px 0;padding:4px 8px;border-radius:6px;
   border:1px solid rgba(128,128,128,.3);vertical-align:top}
.w b{display:block;font-size:15px}
.w small{opacity:.7}
.p{display:inline-block;padding:1px 4px;margin:1px;border-radius:4px;font-family:monospace;font-size:13px}
.s2{background:rgba(34,160,90,.18)} .s1{background:rgba(230,160,20,.35)} .s0{background:rgba(220,50,50,.45);font-weight:700}
.ins{background:rgba(120,80,220,.35);font-style:italic}
</style>""")


def phone_name(p: str) -> str:
    return f"{p}  ({HINT[p]})" if p in HINT else p


@st.cache_resource
def load():
    spk = pd.read_parquet(PQ / "speakers.parquet")
    utt = pd.read_parquet(PQ / "utterances.parquet").merge(spk[["speaker", "age", "gender", "group"]])
    words = pd.read_parquet(PQ / "words.parquet").merge(utt[["utt", "speaker", "group", "age"]])
    ph = pd.read_parquet(PQ / "phones.parquet").merge(utt[["utt", "speaker", "group", "age"]])
    ph["kind"] = np.where(ph.base.isin(VOWELS), "vowel", "consonant")
    ins = pd.read_parquet(PQ / "inserted.parquet").merge(utt[["utt", "speaker", "group"]])
    return spk, utt, words, ph, ins


def matched(ph: pd.DataFrame, min_each=3) -> pd.DataFrame:
    """Keep only (word, phone position) slots read by both children and adults, so the two groups
    are compared on the same material."""
    k = ph[ph.group.isin(["child (6-12)", "adult (19+)"])]
    c = k.groupby(["word", "phone_idx", "group"], observed=True).size().unstack()
    keep = c[(c.get("child (6-12)", 0) >= min_each) & (c.get("adult (19+)", 0) >= min_each)].index
    return k.set_index(["word", "phone_idx"]).loc[lambda d: d.index.isin(keep)].reset_index()


def render_utt(u_words: pd.DataFrame, u_ph: pd.DataFrame, u_ins: pd.DataFrame) -> str:
    out = []
    for _, w in u_words.iterrows():
        ps = u_ph[u_ph.word_idx == w.word_idx].sort_values("phone_idx")
        ins = u_ins[u_ins.word_idx == w.word_idx]
        # an inserted phone counts if at least 2 of 5 experts heard it
        ins = ins.groupby(["after_idx", "base"]).size().loc[lambda s: s >= 2].reset_index()
        parts = []
        for i, p in enumerate(list(ps.itertuples()) + [None]):
            for _, x in ins[ins.after_idx == i].iterrows():
                parts.append(f'<span class="p ins" title="extra sound heard">+{x.base}</span>')
            if p is not None:
                cls = "s2" if p.score >= 1.5 else "s1" if p.score >= 0.5 else "s0"
                tip = f"score {p.score:.1f}/2 · experts: {p.n_ok} ok, {p.n_accent} accented, {p.n_wrong} wrong"
                parts.append(f'<span class="p {cls}" title="{tip}">{p.base}</span>')
        stress = "" if w.stress == 10 else " · stress wrong"
        out.append(f'<span class="w"><b>{html.escape(w.word)}</b>{"".join(parts)}<br>'
                   f"<small>accuracy {w.accuracy}/10{stress}</small></span>")
    return "".join(out)


spk, utt, words, ph, ins = load()
st.sidebar.markdown(
    f"**{len(spk)} speakers · {len(utt):,} recordings**  \n"
    "Native Mandarin speakers reading English sentences aloud. Five experts scored every "
    "sentence, word and sound.")
st.sidebar.markdown(
    "**Sound scores**: 2 = correct, 1 = right but heavily accented, 0 = wrong or missing. "
    "Colours: <span style='background:rgba(34,160,90,.25);padding:0 4px'>correct</span> "
    "<span style='background:rgba(230,160,20,.4);padding:0 4px'>accented</span> "
    "<span style='background:rgba(220,50,50,.5);padding:0 4px'>wrong</span> "
    "<span style='background:rgba(120,80,220,.4);padding:0 4px'>+extra sound</span>",
    unsafe_allow_html=True)
st.sidebar.markdown(
    "**Children vs adults**: the two groups read almost entirely *different* sentences "
    "(children's are shorter and simpler). Comparisons marked *matched* use only words read by both.")

tab_spk, tab_sound, tab_pat = st.tabs(["Speaker", "Sound", "Patterns"])

# ---------------------------------------------------------------- speaker
with tab_spk:
    c1, c2, c3 = st.columns([1, 2, 1])
    grp = c1.selectbox("Group", ["all"] + GROUPS)
    pool = spk if grp == "all" else spk[spk.group == grp]
    pool = pool.sort_values("speaker")
    if "so_spk" not in st.session_state or st.session_state.so_spk not in set(pool.speaker):
        st.session_state.so_spk = pool.speaker.iloc[0]
    if c3.button("🎲 Random speaker", use_container_width=True):
        st.session_state.so_spk = pool.speaker.sample(1).iloc[0]
    slist = pool.speaker.tolist()
    info = pool.set_index("speaker")
    s = c2.selectbox("Speaker", slist, index=slist.index(st.session_state.so_spk),
                     format_func=lambda x: f"{x} · age {info.at[x, 'age']} · {info.at[x, 'gender']}")
    st.session_state.so_spk = s
    sr = info.loc[s]
    m = st.columns(6)
    m[0].metric("Age", int(sr.age))
    m[1].metric("Gender", {"m": "male", "f": "female"}.get(sr.gender, sr.gender))
    for i, k in enumerate(["accuracy", "fluency", "prosodic", "total"]):
        m[i + 2].metric(f"Avg {k}", f"{sr[f'mean_{k}']:.1f}/10")

    su = utt[utt.speaker == s].sort_values("utt")
    st.caption("Hover a sound to see how the five experts marked it.")
    for _, u in su.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**{u.text}**")
            c1.audio(u.wav)
            spread = f" (experts ranged {u.expert_total_min:.0f}–{u.expert_total_max:.0f})" \
                if u.expert_total_max - u.expert_total_min >= 2 else ""
            c1.caption(f"accuracy {u.accuracy} · fluency {u.fluency} · rhythm {u.prosodic} · "
                       f"completeness {u.completeness:g}/10 · **total {u.total}**{spread}")
            c2.html(render_utt(words[words.utt == u.utt], ph[ph.utt == u.utt], ins[ins.utt == u.utt]))

# ---------------------------------------------------------------- sound
with tab_sound:
    counts = ph.base.value_counts()
    plist = counts.index.tolist()
    p = st.selectbox("Sound", plist, index=plist.index("TH"), format_func=phone_name)
    pp = ph[ph.base == p]
    m = st.columns(4)
    m[0].metric("Times expected", f"{len(pp):,}")
    m[1].metric("Average score", f"{pp.score.mean():.2f} / 2")
    m[2].metric("Expert votes 'wrong'", f"{pp.n_wrong.sum() / pp.n_experts.sum():.1%}")
    m[3].metric("Expert votes 'accented'", f"{pp.n_accent.sum() / pp.n_experts.sum():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By position in the word**")
        pos = pp.groupby("position").agg(n=("score", "size"), score=("score", "mean")).reset_index()
        st.altair_chart(alt.Chart(pos).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 2]), title="Average score"),
            y=alt.Y("position:N", sort=["first", "middle", "last", "only"], title=None),
            tooltip=["position", "n", alt.Tooltip("score:Q", format=".2f")]).properties(height=160),
            use_container_width=True)
    with c2:
        st.markdown("**Children vs adults (matched)**")
        mp = matched(pp, 2)
        if mp.empty:
            st.info("Not enough words with this sound read by both groups.")
        else:
            g = mp.groupby("group", observed=True).agg(n=("score", "size"), score=("score", "mean")).reset_index()
            st.altair_chart(alt.Chart(g).mark_bar().encode(
                x=alt.X("score:Q", scale=alt.Scale(domain=[0, 2]), title="Average score"),
                y=alt.Y("group:N", title=None), color=alt.Color("group:N", legend=None),
                tooltip=["group", "n", alt.Tooltip("score:Q", format=".2f")]).properties(height=160),
                use_container_width=True)
            st.caption(f"Only the {mp.word.nunique()} words containing {p} that both groups read.")

    st.markdown("**Words where this sound goes wrong most**")
    ww = pp.groupby("word").agg(times=("score", "size"), score=("score", "mean"),
                                wrong=("n_wrong", "sum"), votes=("n_experts", "sum"))
    ww["wrong share"] = ww.wrong / ww.votes
    st.dataframe(ww[ww.times >= 5].sort_values("score").head(20)[["times", "score", "wrong share"]],
                 column_config={"score": st.column_config.NumberColumn(format="%.2f"),
                                "wrong share": st.column_config.NumberColumn(format="%.2f")})

    st.markdown("**Hear it go wrong**")
    bad = pp[pp.score < 0.5].merge(utt[["utt", "text", "wav"]])
    if bad.empty:
        st.info("No recordings where this sound was scored wrong.")
    else:
        pick = bad.sample(min(4, len(bad)), random_state=st.session_state.get("hear_seed", 0))
        if st.button("Other examples"):
            st.session_state.hear_seed = st.session_state.get("hear_seed", 0) + 1
            st.rerun()
        for _, r in pick.iterrows():
            c1, c2 = st.columns([1, 2])
            c1.audio(r.wav)
            c2.markdown(f"**{r.word}** in “{r.text}” · {r.group}, age {r.age}")

# ---------------------------------------------------------------- patterns
with tab_pat:
    st.subheader("Hardest sounds: children vs adults (matched)")
    st.caption("Each dot is a sound, scored only on words both groups read. Dots below the diagonal: "
               "children do worse than adults; above: better.")
    mp = matched(ph)
    g = mp.groupby(["base", "group"], observed=True).score.mean().unstack()
    g["n"] = mp.groupby("base").size()
    g = g[g.n >= 100].reset_index()
    g["kind"] = np.where(g.base.isin(VOWELS), "vowel", "consonant")
    g["hint"] = g.base.map(HINT)
    lo = float(min(g["child (6-12)"].min(), g["adult (19+)"].min())) - 0.02
    diag = pd.DataFrame({"x": [lo, 2], "y": [lo, 2]})
    pts = alt.Chart(g).mark_circle(size=70).encode(
        x=alt.X("adult (19+):Q", scale=alt.Scale(domain=[lo, 2]), title="Adult average score"),
        y=alt.Y("child (6-12):Q", scale=alt.Scale(domain=[lo, 2]), title="Child average score"),
        color="kind:N", tooltip=["base", "hint", "n", alt.Tooltip("adult (19+):Q", format=".2f"),
                                 alt.Tooltip("child (6-12):Q", format=".2f")])
    txt = pts.mark_text(dx=10, dy=-6, fontSize=11).encode(text="base:N")
    line = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[lo, 2])), y=alt.Y("y:Q", scale=alt.Scale(domain=[lo, 2])))
    st.altair_chart((line + pts + txt).properties(height=460), use_container_width=True)
    ov = mp.groupby("group", observed=True).score.mean()
    st.caption(f"Matched overall: children {ov.get('child (6-12)', np.nan):.3f}, "
               f"adults {ov.get('adult (19+)', np.nan):.3f} (out of 2), over {len(mp):,} sounds.")

    st.subheader("Word endings")
    st.caption("Mandarin syllables rarely end in a consonant, so a classic pattern is dropping or softening "
               "final consonants, or adding a vowel after them.")
    c1, c2 = st.columns(2)
    with c1:
        pk = ph[ph.position != "only"].groupby(["kind", "position"]).score.mean().reset_index()
        st.altair_chart(alt.Chart(pk).mark_bar().encode(
            x=alt.X("position:N", sort=["first", "middle", "last"], title="Position in word"),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[1.6, 2]), title="Average score"),
            color="kind:N", xOffset="kind:N",
            tooltip=["kind", "position", alt.Tooltip("score:Q", format=".3f")]).properties(height=260),
            use_container_width=True)
    with c2:
        iv = ins.groupby(["utt", "word_idx", "after_idx", "base", "where"]).size().loc[lambda s: s >= 2] \
            .reset_index().groupby(["base", "where"]).size().rename("n").reset_index()
        top = iv.groupby("base").n.sum().nlargest(10).index
        st.altair_chart(alt.Chart(iv[iv.base.isin(top)]).mark_bar().encode(
            x=alt.X("n:Q", title="Times heard (≥2 of 5 experts agree)"),
            y=alt.Y("base:N", sort="-x", title="Extra sound"),
            color=alt.Color("where:N", title="Where in word"),
            tooltip=["base", "where", "n"]).properties(height=260),
            use_container_width=True)

    st.subheader("Age and overall score (matched words)")
    mw = words[words.group.isin(["child (6-12)", "adult (19+)"])]
    shared = mw.groupby("word").group.nunique().loc[lambda s: s == 2].index
    sa = words[words.word.isin(shared)].groupby(["speaker", "age", "group"], observed=True) \
        .accuracy.mean().reset_index()
    st.altair_chart(alt.Chart(sa).mark_circle(opacity=0.6).encode(
        x=alt.X("age:Q", title="Age"), y=alt.Y("accuracy:Q", title="Avg word accuracy on shared words",
                                               scale=alt.Scale(zero=False)),
        color="group:N", tooltip=["speaker", "age", alt.Tooltip("accuracy:Q", format=".2f")]
    ).properties(height=300), use_container_width=True)

    st.subheader("How much do the experts agree?")
    utt["spread"] = utt.expert_total_max - utt.expert_total_min
    st.altair_chart(alt.Chart(utt).mark_bar().encode(
        x=alt.X("spread:Q", bin=alt.Bin(step=0.5), title="Highest minus lowest expert total score (0-10)"),
        y=alt.Y("count():Q", title="Recordings")).properties(height=220), use_container_width=True)
    st.caption(f"Median spread {utt.spread.median():.1f} points: sentence scores are fuzzy by about that much.")
