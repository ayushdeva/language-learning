"""EdNet explorer: Korean students practising for the TOEIC on Santa.

Run:  uv run streamlit run app.py --server.port 8504
Needs prepare.py to have been run first.
"""

from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

import overview

DATA = Path(__file__).resolve().parents[2] / "data" / "ednet"
PQ = DATA / "parquet"
KT1 = PQ / "kt1.parquet"
PART_NAME = {1: "1 Photographs", 2: "2 Question–response", 3: "3 Conversations", 4: "4 Talks",
             5: "5 Incomplete sentences", 6: "6 Text completion", 7: "7 Reading comprehension"}
STAGES = {1: "1–10", 2: "11–30", 3: "31–100", 4: "101–300", 5: "301–1k", 6: "1k–3k", 7: "3k+"}

st.set_page_config(page_title="EdNet explorer", layout="wide")


@st.cache_resource
def load():
    q = pd.read_parquet(PQ / "questions.parquet")
    per_month = duckdb.execute(f"""
        SELECT date_trunc('month', time) AS month, count(*) AS answers, count(DISTINCT user) AS students
        FROM '{KT1}' GROUP BY 1 ORDER BY 1""").df()
    return {
        "questions": q,
        "users": pd.read_parquet(PQ / "users.parquet"),
        "tags": pd.read_parquet(PQ / "tags.parquet"),
        "pairs": pd.read_parquet(PQ / "tag_pairs.parquet"),
        "curve": pd.read_parquet(PQ / "curve.parquet"),
        "timing": pd.read_parquet(PQ / "timing.parquet"),
        "per_month": per_month,
    }


@st.cache_data(max_entries=50)
def student_rows(user: int) -> pd.DataFrame:
    return duckdb.execute(f"SELECT * FROM '{KT1}' WHERE user = ? ORDER BY n_th", [user]).df()


@st.cache_data(max_entries=50)
def question_rows(qid: int) -> pd.DataFrame:
    return duckdb.execute(f"""
        SELECT a.user_answer, a.correct, a.elapsed_s, a.n_th, u.ability
        FROM '{KT1}' a JOIN '{PQ / "users.parquet"}' u USING (user) WHERE a.qid = ?""", [qid]).df()


def weighted(df, by, cols=("accuracy",)):
    g = df.assign(**{f"_{c}": df[c] * df.n for c in cols}).groupby(list(by), observed=True)
    out = g[[f"_{c}" for c in cols] + ["n"]].sum()
    for c in cols:
        out[c] = out[f"_{c}"] / out.n
    return out[list(cols) + ["n"]].reset_index()


D = load()
Q, U = D["questions"], D["users"]
st.sidebar.markdown(
    f"**{len(U):,} students · {int(U.answers.sum()):,} answers · {len(Q):,} questions**  \n"
    "Santa, a Korean TOEIC-prep app. Every multiple-choice answer, with the option chosen and time spent.")
st.sidebar.markdown(
    "**Ability** = how much more often a student is right than the questions they got would predict "
    "(+0.10 = right 10 points more often than average on the same questions).")
st.sidebar.markdown(
    "**Keep in mind**: Santa picks most questions for each student; dates are shifted for privacy; skill "
    "tags are numbers without names.")

tab_ov, tab_student, tab_q, tab_tags, tab_pat = st.tabs(
    ["Overview", "Student", "Question", "Skill tags", "Patterns"])

with tab_ov:
    overview.render(D, DATA / "EdNet-KT1.zip")

# ---------------------------------------------------------------- student
with tab_student:
    pool = U[U.answers >= 100]
    if "ed_user" not in st.session_state:
        st.session_state.ed_user = int(pool.sort_values("answers").user.iloc[len(pool) // 2])
    c1, c2, c3 = st.columns([2, 1, 1])
    typed = c1.text_input("Student id (number after 'u')", str(st.session_state.ed_user))
    if typed.lstrip("u").isdigit() and int(typed.lstrip("u")) in set(U.user):
        st.session_state.ed_user = int(typed.lstrip("u"))
    if c2.button("🎲 Random (100+ answers)", use_container_width=True):
        st.session_state.ed_user = int(pool.user.sample(1).iloc[0])
        st.rerun()
    if c3.button("🎲 Random heavy user (3k+)", use_container_width=True):
        st.session_state.ed_user = int(U[U.answers >= 3000].user.sample(1).iloc[0])
        st.rerun()
    user = st.session_state.ed_user
    ur = U.set_index("user").loc[user]
    rows = student_rows(user)

    m = st.columns(6)
    m[0].metric("Answers", f"{int(ur.answers):,}")
    m[1].metric("Bundles", f"{int(ur.bundles):,}")
    m[2].metric("Accuracy", f"{ur.accuracy:.1%}")
    m[3].metric("Ability", f"{ur.ability:+.3f}", help=f"Percentile {(U.ability < ur.ability).mean():.0%}")
    m[4].metric("Active days", int(ur.active_days))
    m[5].metric("Span", f"{(ur.last_time - ur.first_time).days} days")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Accuracy as they practise** (rolling average over 50 answers)")
        r = rows[["n_th", "correct", "part"]].copy()
        r["rolling"] = r.correct.rolling(50, min_periods=10).mean()
        st.altair_chart(alt.Chart(r).mark_line().encode(
            x=alt.X("n_th:Q", title="Answer number"),
            y=alt.Y("rolling:Q", title="Accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=["n_th", alt.Tooltip("rolling:Q", format=".0%")]).properties(height=260),
            use_container_width=True)
    with c2:
        st.markdown("**By part: this student vs everyone**")
        mine = rows.groupby("part").correct.agg(["mean", "size"]).rename(columns={"mean": "accuracy"})
        allp = Q.groupby("part").apply(lambda g: (g.accuracy * g.answers).sum() / g.answers.sum(),
                                       include_groups=False).rename("accuracy")
        cmp_ = pd.concat([mine.accuracy.rename("this student"), allp.rename("everyone")], axis=1) \
            .reset_index().melt("part", var_name="who", value_name="accuracy")
        cmp_["part"] = cmp_.part.map(PART_NAME)
        st.altair_chart(alt.Chart(cmp_).mark_bar().encode(
            y=alt.Y("part:N", title=None), x=alt.X("accuracy:Q", axis=alt.Axis(format="%")),
            color=alt.Color("who:N", title=None), yOffset="who:N",
            tooltip=["part", "who", alt.Tooltip("accuracy:Q", format=".1%")]).properties(height=260),
            use_container_width=True)
        st.caption("Different students get different questions, so this isn't a like-for-like comparison.")

    st.subheader("Answer log")
    bundles = rows.solving_id.unique()
    page = st.slider("Bundles", 1, len(bundles), (1, min(20, len(bundles))), key="bpage")
    sel = rows[rows.solving_id.isin(bundles[page[0] - 1:page[1]])].copy()
    sel["question"] = "q" + sel.qid.astype(str)
    sel["result"] = np.where(sel.correct == 1, "✅", "❌")
    sel["gap"] = sel.time.diff().dt.total_seconds().div(3600)
    sel["gap"] = sel.gap.map(lambda h: "" if pd.isna(h) or h < 1 else f"{h:.0f} h later" if h < 48 else f"{h / 24:.0f} days later")
    st.dataframe(
        sel[["n_th", "solving_id", "gap", "part", "bundle_id", "question", "user_answer", "correct_answer",
             "result", "elapsed_s"]],
        hide_index=True, use_container_width=True, height=500,
        column_config={"n_th": "#", "solving_id": "bundle #", "gap": "break before",
                       "user_answer": "answered", "correct_answer": "correct",
                       "elapsed_s": st.column_config.NumberColumn("seconds (bundle)", format="%.0f")})

# ---------------------------------------------------------------- question
with tab_q:
    qq = Q[Q.answers >= 100].sort_values("answers", ascending=False)
    c1, c2, c3 = st.columns([2, 1, 1])
    part_f = c1.selectbox("Part", ["all"] + list(PART_NAME), format_func=lambda p: "All parts" if p == "all"
                          else PART_NAME[p])
    pool = qq if part_f == "all" else qq[qq.part == part_f]
    if "ed_q" not in st.session_state or st.session_state.ed_q not in set(pool.qid):
        st.session_state.ed_q = int(pool.qid.iloc[0])
    if c3.button("🎲 Random question", use_container_width=True):
        st.session_state.ed_q = int(pool.qid.sample(1).iloc[0])
        st.rerun()
    typed = c2.text_input("Question id", f"q{st.session_state.ed_q}")
    if typed.lstrip("q").isdigit() and int(typed.lstrip("q")) in set(Q.qid):
        st.session_state.ed_q = int(typed.lstrip("q"))
    qr = Q.set_index("qid").loc[st.session_state.ed_q]
    st.caption("EdNet doesn't release question content (text, options, passages, audio or images): only the "
               "id, bundle, correct letter, part and skill tags. What students chose is all there is to go on.")

    m = st.columns(6)
    m[0].metric("Part", PART_NAME[qr.part])
    m[1].metric("Correct option", qr.correct_answer)
    m[2].metric("Answers", f"{int(qr.answers):,}" if pd.notna(qr.answers) else "0")
    m[3].metric("Accuracy", f"{qr.accuracy:.1%}" if pd.notna(qr.accuracy) else "–")
    m[4].metric("Skill tags", qr.tags.replace(";", ", "))
    m[5].metric("Bundle", f"{qr.bundle_id} ({(Q.bundle_id == qr.bundle_id).sum()} q)")

    if pd.isna(qr.answers):
        st.info("Nobody answered this question in KT1.")
    else:
        rows_q = question_rows(st.session_state.ed_q)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Which options students picked**")
            n_opt = 3 if qr.part == 2 else 4
            ch = pd.DataFrame({"option": list("abcd")[:n_opt],
                               "n": [qr[f"n_{o}"] for o in "abcd"][:n_opt]})
            ch["share"] = ch.n / ch.n.sum()
            ch["is"] = np.where(ch.option == qr.correct_answer, "correct", "wrong")
            st.altair_chart(alt.Chart(ch).mark_bar().encode(
                x=alt.X("option:N", title=None), y=alt.Y("share:Q", axis=alt.Axis(format="%")),
                color=alt.Color("is:N", scale=alt.Scale(domain=["correct", "wrong"],
                                                        range=["#2ca25f", "#de2d26"]), title=None),
                tooltip=["option", "n", alt.Tooltip("share:Q", format=".1%")]).properties(height=260),
                use_container_width=True)
            wrong = ch[ch["is"] == "wrong"]
            if len(wrong) and wrong.n.sum():
                top = wrong.nlargest(1, "n").iloc[0]
                st.caption(f"Most popular wrong option: **{top.option}**, chosen by "
                           f"{top.n / wrong.n.sum():.0%} of students who got it wrong. An even split would be "
                           f"{1 / len(wrong):.0%}.")
        with c2:
            st.markdown("**Who gets it right** (accuracy by student ability)")
            rq = rows_q.dropna(subset=["ability"]).copy()
            cuts = np.quantile(U.ability, np.linspace(0, 1, 11))
            rq["decile"] = np.clip(np.searchsorted(cuts, rq.ability, side="right"), 1, 10)
            dec = rq.groupby("decile").correct.agg(["mean", "size"]).reset_index()
            st.altair_chart(alt.Chart(dec).mark_line(point=True).encode(
                x=alt.X("decile:O", title="Student ability decile (1 = weakest, 10 = strongest)"),
                y=alt.Y("mean:Q", title="Accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["decile", "size", alt.Tooltip("mean:Q", format=".1%")]).properties(height=260),
                use_container_width=True)
            st.caption("A good question separates strong from weak students: a steep line. A flat line means "
                       "it doesn't reflect ability (lucky guesses, or a confusing question).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Other questions in the same bundle**")
            sib = Q[Q.bundle_id == qr.bundle_id][["question_id", "correct_answer", "tags", "answers", "accuracy"]]
            st.dataframe(sib, hide_index=True, use_container_width=True,
                         column_config={"accuracy": st.column_config.NumberColumn(format="%.3f")})
        with c2:
            st.markdown("**Time spent on the bundle: right vs wrong**")
            t = rows_q[(rows_q.elapsed_s > 0) & (rows_q.elapsed_s < rows_q.elapsed_s.quantile(0.98))].copy()
            t["result"] = np.where(t.correct == 1, "right", "wrong")
            st.altair_chart(alt.Chart(t.sample(min(len(t), 20000), random_state=0)).mark_bar(opacity=0.6)
                            .encode(x=alt.X("elapsed_s:Q", bin=alt.Bin(maxbins=40), title="Seconds"),
                                    y=alt.Y("count():Q", stack=None, title="Answers"),
                                    color=alt.Color("result:N", scale=alt.Scale(domain=["right", "wrong"],
                                                                                range=["#2ca25f", "#de2d26"]))
                                    ).properties(height=220), use_container_width=True)

# ---------------------------------------------------------------- tags
with tab_tags:
    tags = D["tags"].sort_values("answers", ascending=False)
    st.markdown("Skill tags are expert labels attached to questions (a question can have several). They "
                "were released as numbers only, so what each tag means is unknown; the parts and questions "
                "it appears on are the best clue.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(tags, hide_index=True, use_container_width=True, height=420, column_config={
            "accuracy": st.column_config.NumberColumn(format="%.3f"),
            "parts": "parts it appears in"})
    with c2:
        st.altair_chart(alt.Chart(tags[tags.answers > 10000]).mark_circle(opacity=0.6).encode(
            x=alt.X("answers:Q", scale=alt.Scale(type="log"), title="Answers (log)"),
            y=alt.Y("accuracy:Q", axis=alt.Axis(format="%"), title="Accuracy"),
            color=alt.Color("parts:N", legend=None),
            tooltip=["tag", "questions", "parts", "answers", alt.Tooltip("accuracy:Q", format=".1%")]
        ).properties(height=420), use_container_width=True)
    tag = st.selectbox("Look at one tag", tags.tag.tolist())
    tr = tags.set_index("tag").loc[tag]
    st.caption(f"Tag {tag}: {int(tr.questions)} questions in part(s) {tr.parts}, accuracy {tr.accuracy:.1%}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Its questions**")
        tq = Q[Q.tags.fillna("").str.split(";").map(lambda ts: str(tag) in ts)]
        st.dataframe(tq[["question_id", "part", "tags", "answers", "accuracy"]].sort_values("answers",
                                                                                            ascending=False),
                     hide_index=True, use_container_width=True, height=350,
                     column_config={"accuracy": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.markdown("**Tags whose mastery goes with it**")
        p = D["pairs"]
        a = p[p.tag_a == tag].rename(columns={"tag_b": "other"})
        b = p[p.tag_b == tag].rename(columns={"tag_a": "other"})
        pt = pd.concat([a, b])[["other", "n_shared", "link", "z", "same_question_share"]]
        st.dataframe(pt.sort_values("z", ascending=False).head(25), hide_index=True, use_container_width=True,
                     height=350, column_config={
                         "link": st.column_config.NumberColumn(format="%.3f"),
                         "z": st.column_config.NumberColumn("strength (z)", format="%.1f"),
                         "same_question_share": st.column_config.NumberColumn("on same questions",
                                                                              format="%.2f")})

# ---------------------------------------------------------------- patterns
with tab_pat:
    st.subheader("Accuracy with practice")
    st.caption("Accuracy at each stage of a student's history. 'Students with 1k+ answers' follows only "
               "students who stayed long, so the line isn't just the stronger students surviving. "
               "'Beyond question difficulty' subtracts each question's average accuracy, in case Santa "
               "serves harder questions later.")
    c1, c2 = st.columns([1, 3])
    who = c1.radio("Students", ["All students", "Students with 1k+ answers"])
    measure = c1.radio("Measure", ["Accuracy", "Beyond question difficulty"])
    by_part = c1.checkbox("Split by part")
    cv = D["curve"] if who == "All students" else D["curve"][D["curve"].stayer]
    col = "accuracy" if measure == "Accuracy" else "resid"
    g = weighted(cv, ["stage", "part"] if by_part else ["stage"], cols=(col,))
    g["stage_label"] = g.stage.map(STAGES)
    enc = dict(x=alt.X("stage_label:N", sort=list(STAGES.values()), title="Answer number"),
               y=alt.Y(f"{col}:Q", axis=alt.Axis(format="%" if col == "accuracy" else "+.0%"),
                       scale=alt.Scale(zero=False), title=measure),
               tooltip=["stage_label", "n", alt.Tooltip(f"{col}:Q", format=".1%")])
    if by_part:
        g["part"] = g.part.map(PART_NAME)
        enc["color"] = alt.Color("part:N")
    c2.altair_chart(alt.Chart(g).mark_line(point=True).encode(**enc).properties(height=320),
                    use_container_width=True)

    st.subheader("Time spent vs accuracy")
    tm = D["timing"].copy()
    tm["part"] = tm.part.map(PART_NAME)
    tm = tm[tm.n >= 500]
    st.altair_chart(alt.Chart(tm).mark_line().encode(
        x=alt.X("secs:Q", title="Seconds on the bundle (120 = 120+)"),
        y=alt.Y("accuracy:Q", axis=alt.Axis(format="%")), color="part:N",
        tooltip=["part", "secs", "n", alt.Tooltip("accuracy:Q", format=".1%")]).properties(height=300),
        use_container_width=True)
    st.caption("Time is per bundle, so multi-question parts (3, 4, 6, 7) naturally take longer.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hardest parts")
        pp = Q.dropna(subset=["answers"]).groupby("part").apply(
            lambda g: pd.Series({"accuracy": (g.accuracy * g.answers).sum() / g.answers.sum(),
                                 "answers": g.answers.sum()}), include_groups=False).reset_index()
        pp["part"] = pp.part.map(PART_NAME)
        pp["guess"] = np.where(pp.part.str.startswith("2"), 1 / 3, 1 / 4)
        base = alt.Chart(pp).encode(y=alt.Y("part:N", title=None))
        st.altair_chart((base.mark_bar().encode(x=alt.X("accuracy:Q", axis=alt.Axis(format="%"),
                                                        scale=alt.Scale(domain=[0, 1])),
                                                tooltip=["part", "answers", alt.Tooltip("accuracy:Q", format=".1%")])
                         + base.mark_tick(color="black", thickness=2, size=18).encode(x="guess:Q"))
                        .properties(height=260), use_container_width=True)
        st.caption("Black tick = accuracy from pure guessing.")
    with c2:
        st.subheader("Answer-letter bias")
        lt = []
        for part, g in Q.dropna(subset=["answers"]).groupby("part"):
            chosen = g[["n_a", "n_b", "n_c", "n_d"]].sum()
            corr = g.correct_answer.value_counts()
            for o in "abcd":
                lt.append({"part": PART_NAME[part], "option": o, "kind": "chosen by students",
                           "share": chosen[f"n_{o}"] / chosen.sum()})
                lt.append({"part": PART_NAME[part], "option": o, "kind": "correct answer",
                           "share": corr.get(o, 0) / corr.sum()})
        lt = pd.DataFrame(lt)
        st.altair_chart(alt.Chart(lt).mark_bar().encode(
            x=alt.X("option:N", title=None), y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("kind:N", title=None), xOffset="kind:N",
            facet=alt.Facet("part:N", columns=4, title=None),
            tooltip=["part", "option", "kind", alt.Tooltip("share:Q", format=".1%")]
        ).properties(height=110, width=110))
        st.caption("Are correct answers spread evenly over a–d, and do students favour a letter?")

    st.subheader("Most tempting wrong options")
    st.caption("Questions where one wrong option pulls most of the wrong answers: often a classic trap.")
    qd = Q[Q.answers >= 1000].copy()
    opts = ["n_a", "n_b", "n_c", "n_d"]
    wrong_counts = qd[opts].to_numpy().astype(float)
    wrong_counts[np.arange(len(qd)), qd.correct_answer.map("abcd".index).to_numpy()] = 0
    qd["top_wrong"] = np.array(list("abcd"))[wrong_counts.argmax(1)]
    qd["pull"] = wrong_counts.max(1) / wrong_counts.sum(1)
    st.dataframe(qd.nlargest(30, "pull")[["question_id", "part", "tags", "correct_answer", "top_wrong",
                                          "pull", "accuracy", "answers"]],
                 hide_index=True, use_container_width=True, column_config={
                     "pull": st.column_config.NumberColumn("share of wrong answers on it", format="%.2f"),
                     "accuracy": st.column_config.NumberColumn(format="%.3f")})

    st.subheader("Skill tags that go together")
    st.caption("Like SLAM's word links: does doing better than expected on one tag go with doing better on "
               "another, after removing question difficulty and student ability? Tags sharing questions are "
               "linked mechanically, so filter those out.")
    c1, c2 = st.columns(2)
    max_same = c1.slider("Max share on same questions", 0.0, 1.0, 0.0, 0.05)
    min_n = c2.slider("Min students", 300, 50000, 2000, 100)
    p = D["pairs"]
    p = p[(p.same_question_share <= max_same) & (p.n_shared >= min_n)]
    ti = D["tags"].set_index("tag")
    p = p.assign(parts_a=p.tag_a.map(ti.parts), parts_b=p.tag_b.map(ti.parts))
    st.dataframe(p.nlargest(40, "z")[["tag_a", "parts_a", "tag_b", "parts_b", "n_shared", "link", "z",
                                      "same_question_share"]],
                 hide_index=True, use_container_width=True, column_config={
                     "link": st.column_config.NumberColumn(format="%.3f"),
                     "z": st.column_config.NumberColumn("strength (z)", format="%.1f")})
