"""Overview tab for the EdNet explorer: what the dataset is, and what the levels we didn't download hold."""

import zipfile

import altair as alt
import pandas as pd
import streamlit as st

PARTS = pd.DataFrame([
    (1, "Listening", "Photographs", "Look at a photo, hear four statements, pick the one that describes it.", 4),
    (2, "Listening", "Question–response", "Hear a question or remark and three responses; pick the best reply.", 3),
    (3, "Listening", "Conversations", "Hear a short conversation, answer several questions about it.", 4),
    (4, "Listening", "Talks", "Hear a short talk (announcement, message…), answer several questions.", 4),
    (5, "Reading", "Incomplete sentences", "Pick the word or phrase that fills the gap in a sentence.", 4),
    (6, "Reading", "Text completion", "Fill several gaps in a short text (words, phrases or sentences).", 4),
    (7, "Reading", "Reading comprehension", "Read one or more passages, answer several questions.", 4),
], columns=["part", "section", "name", "what the student does", "options"])

KT1_FIELDS = pd.DataFrame([
    ("timestamp", "When the question was given, Unix time in milliseconds. **Shifted by a fixed amount "
     "for privacy**, so the dates are not real, though gaps between events are."),
    ("solving_id", "Counter of bundle attempts for this student: 1, 2, 3… Questions sharing a bundle "
     "share this number."),
    ("question_id", "Question, q{number}. Links to the questions table for the correct answer, part and tags."),
    ("user_answer", "The option the student submitted: a, b, c or d."),
    ("elapsed_time", "Time spent, in milliseconds. For bundles, every question in the bundle carries the same "
     "value, so it describes the bundle rather than each question."),
], columns=["field", "meaning"])

CONTENT_TABLES = pd.DataFrame([
    ("questions.csv", "13,169 rows",
     "question_id, bundle_id, explanation_id, correct_answer (a–d), part (1–7), tags (skill numbers "
     "separated by ;), deployed_at (when it went live)"),
    ("lectures.csv", "1,021 rows",
     "lecture_id, part (0 = none), tags (one skill number), video_length (ms), deployed_at. -1 = unknown."),
    ("payments.csv", "190 rows",
     "payment_item_id, type (pass = time-limited full access, paygo = a number of questions), duration (ms; "
     "the column is misspelt 'duaration' in the file), number_of_questions"),
    ("coupons.csv", "91 rows", "coupon_id, coupon_type, duration (ms of access granted)"),
], columns=["file", "size", "columns"])

LEVELS = pd.DataFrame([
    ("KT1", "Downloaded", "1.1 GB zip · 5.6 GB · 784,309 files", "784k",
     "One row per question answered: which option, how long. Since Apr 2017 (shifted dates)."),
    ("KT2", "Not downloaded", "530 MB zip · 3.1 GB · 297,444 files", "297k",
     "Every *action* around answering, from 27 Aug 2018: entering a bundle, each option clicked (including "
     "changes of mind before submitting), submitting. Adds the source (which part of the app) and platform "
     "(mobile or web). KT1 can be rebuilt from it."),
    ("KT3", "Not downloaded", "727 MB zip · 4.3 GB · 297,915 files", "298k",
     "KT2 plus learning activities: opening and closing expert explanations of answers, and starting and "
     "stopping video lectures. Lets you measure time spent studying and whether it helps."),
    ("KT4", "Not downloaded", "1.1 GB zip · 6.4 GB · 297,915 files", "298k",
     "KT3 plus every remaining UI action: erasing / un-erasing answer options, playing and pausing audio "
     "and video (with the cursor position), payments, refunds and coupon use."),
], columns=["level", "status", "size", "students", "what it records"])

KT2_ACTIONS = pd.DataFrame([
    ("enter", "KT2+", "Student opens a question bundle (item = bundle id). In KT3+ also opening an explanation "
     "(item = e…) or starting a lecture (item = l…)."),
    ("respond", "KT2+", "Student picks an option for a question. Can happen several times per question; the "
     "last one before submit is the answer."),
    ("submit", "KT2+", "Student submits the bundle."),
    ("quit", "KT3+", "Student leaves an explanation or stops a lecture."),
    ("erase_choice / undo_erase_choice", "KT4", "Crossing out an option, or bringing it back."),
    ("play_audio / pause_audio", "KT4", "Listening audio played or paused, with cursor_time."),
    ("play_video / pause_video", "KT4", "Lecture video played or paused, with cursor_time."),
    ("pay / refund", "KT4", "Buying or refunding a payment item (item = p…)."),
    ("enroll_coupon", "KT4", "Entering a coupon code (item = c…)."),
], columns=["action_type", "level", "meaning"])

SOURCES = pd.DataFrame([
    ("diagnosis", "Placement questions when a student starts."),
    ("sprint", "Student picks a TOEIC part and practises only that part."),
    ("tutor", "'All parts' mode: Santa picks questions from any part."),
    ("todays_recommendation::…", "Daily questions and lectures Santa recommends from its prediction of what "
     "the student knows."),
    ("adaptive_offer", "Lectures and questions offered when a student keeps getting a skill wrong, or their "
     "accuracy on it drops."),
    ("in_review", "Redoing questions already solved."),
    ("after_sprint / after_review / my_note", "(KT3+) where an explanation was opened from."),
    ("archive", "(KT3+) the lecture library."),
], columns=["source", "meaning"])


@st.cache_data
def raw_example(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist()[1:200]:
            text = z.read(name).decode()
            if 12 <= text.count("\n") <= 400:
                return f"{name}\n" + "\n".join(text.splitlines()[:12])
    return ""


def render(D, zip_path):
    q, users = D["questions"], D["users"]
    n_answers = int(users.answers.sum())

    st.header("EdNet (KT1): what's in this dataset")
    st.markdown(
        "Every multiple-choice question answered by **784k students** on **Santa**, a Korean app that "
        "prepares people for the **TOEIC** (an English test widely used in Korea for jobs and university "
        "admission). Released by the company behind Santa (Riiid) as a benchmark for *knowledge tracing*: "
        "predicting whether a student will get the next question right from everything they did before.")

    m = st.columns(6)
    m[0].metric("Students", f"{len(users):,}")
    m[1].metric("Answers", f"{n_answers / 1e6:.1f}M")
    m[2].metric("Questions", f"{len(q):,}")
    m[3].metric("Bundles", f"{q.bundle_id.nunique():,}")
    m[4].metric("Skill tags", D["tags"].shape[0])
    m[5].metric("Overall accuracy", f"{(users.accuracy * users.answers).sum() / n_answers:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Source")
        st.markdown(
            "- **Paper:** Y. Choi, Y. Lee, D. Shin, J. Cho, S. Park, S. Lee, J. Baek, C. Bae, B. Kim, J. Heo "
            "(arXiv, 2019). *EdNet: A Large-Scale Hierarchical Dataset in Education.* "
            "[arXiv:1912.03072](https://arxiv.org/abs/1912.03072)\n"
            "- **Data & docs:** [github.com/riiid/ednet](https://github.com/riiid/ednet), files on Google "
            "Drive. No signup needed.\n"
            "- **License:** CC BY-NC 4.0, for research: non-commercial use with credit.\n"
            "- **Related:** Riiid later ran a Kaggle competition, *Riiid Answer Correctness Prediction*, "
            "on data from the same app.")
    with c2:
        st.subheader("Who is in it")
        st.markdown(
            "- Users of Santa in **Korea**, on Android, iOS and web, studying for the TOEIC.\n"
            "- Presumably mostly Korean-speaking adults (students and job seekers), but **nothing about "
            "students is recorded**: no age, level, target score or background.\n"
            "- Students run from one question to tens of thousands (see below).\n"
            "- Free users got a daily quota of questions from some parts; paying users got everything "
            "(payments appear only in KT4), so heavy users skew towards paying customers.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Answers per student**")
        u = users.assign(bucket=pd.cut(users.answers, [0, 10, 30, 100, 300, 1000, 3000, 10**7],
                                       labels=["1–10", "11–30", "31–100", "101–300", "301–1k", "1k–3k", "3k+"]))
        b = u.groupby("bucket", observed=True).agg(students=("user", "size"), answers=("answers", "sum"))
        b = (b / b.sum()).reset_index().melt("bucket", var_name="share of", value_name="share")
        st.altair_chart(alt.Chart(b).mark_bar().encode(
            x=alt.X("bucket:N", sort=None, title="Answers per student"),
            y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("share of:N", title="Share of"), xOffset="share of:N",
            tooltip=["bucket", "share of", alt.Tooltip("share:Q", format=".1%")]).properties(height=260),
            use_container_width=True)
        st.caption(f"Median {users.answers.median():.0f} answers per student, but most answers come from the "
                   "small group of heavy users.")
    with c2:
        st.markdown("**Activity over (shifted) time**")
        per = D["per_month"]
        st.altair_chart(alt.Chart(per).mark_bar().encode(
            x=alt.X("month:T", title=None), y=alt.Y("answers:Q", title="Answers per month"),
            tooltip=["month:T", "answers", "students"]).properties(height=260), use_container_width=True)
        st.caption("Dates are shifted by a fixed amount, so treat them as relative: the gaps are real, the "
                   "calendar isn't.")

    st.divider()
    st.subheader("The TOEIC parts")
    pc = q.groupby("part").agg(questions=("qid", "size"), bundles=("bundle_id", "nunique"),
                               answers=("answers", "sum"),
                               accuracy=("accuracy", lambda s: (s * q.loc[s.index, "answers"]).sum()
                                         / q.loc[s.index, "answers"].sum()))
    st.dataframe(PARTS.merge(pc.reset_index(), on="part"), hide_index=True, use_container_width=True,
                 column_config={"accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "what the student does": st.column_config.TextColumn(width="large")})
    st.caption("Part descriptions follow the standard TOEIC Listening & Reading test. Every part has four "
               "options except part 2, which has three, so guessing gives 25% (33% in part 2).")

    st.subheader("Bundles")
    st.markdown(
        "Questions come in **bundles** that share a passage, picture or recording: one conversation in part 3 "
        "comes with several questions. A student gets the whole bundle, answers every question in it, then "
        "submits. Bundles are the unit of `solving_id` and of `elapsed_time`.")
    bs = q.groupby(["part", "bundle_id"]).size().rename("questions").reset_index()
    st.dataframe(bs.groupby("part").questions.agg(["mean", "min", "max"]).round(2).T,
                 use_container_width=True)
    st.caption("Questions per bundle, by part.")

    st.divider()
    st.subheader("KT1: what one row means")
    st.dataframe(KT1_FIELDS, hide_index=True, use_container_width=True,
                 column_config={"meaning": st.column_config.TextColumn(width="large")})
    st.markdown("**The start of a real student file** (one file per student, `KT1/u{id}.csv`)")
    st.code(raw_example(str(zip_path)), language=None)
    st.markdown(
        "Right or wrong is **not in KT1 itself**: you compare `user_answer` with `correct_answer` from the "
        "questions table. This explorer does that join for every answer.")

    st.subheader("Contents: the tables that describe the material")
    st.dataframe(CONTENT_TABLES, hide_index=True, use_container_width=True,
                 column_config={"columns": st.column_config.TextColumn(width="large")})
    st.markdown(
        "- **Tags** are expert-assigned skill labels shared by questions and lectures (293 in total: 188 "
        "appear on questions and 259 on lectures, overlapping. This explorer uses the 188 question tags). "
        "**They are released as bare numbers without names**, so the app can tell you which tags are hard or "
        "go together, but not what tag 179 is about.\n"
        "- Every bundle has an **explanation** (expert commentary) with the same number: b1707 ↔ e1707.\n"
        "- A *scores* table was announced but never released.")

    st.divider()
    st.subheader("The levels: what KT2, KT3 and KT4 add (for later)")
    st.markdown(
        "EdNet comes in four nested levels. Each level after KT1 switches from *one row per answer* to "
        "*one row per action in the app*, and each adds more kinds of action. They cover a smaller set of "
        "students: only those active after the app started logging full behaviour (27 Aug 2018).")
    st.dataframe(LEVELS, hide_index=True, use_container_width=True,
                 column_config={"what it records": st.column_config.TextColumn(width="large")})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Action types (KT2–KT4)**")
        st.dataframe(KT2_ACTIONS, hide_index=True, use_container_width=True,
                     column_config={"meaning": st.column_config.TextColumn(width="large")})
        st.markdown(
            "**Columns (KT2–KT4):** `timestamp`, `action_type`, `item_id` (b… bundle, q… question, e… "
            "explanation, l… lecture, p… payment, c… coupon), `source`, `user_answer` (on respond and "
            "erase actions), `platform` (mobile / web), and in KT4 `cursor_time` (media position, ms).")
    with c2:
        st.markdown("**Sources: where in the app the action happened (KT2–KT4)**")
        st.dataframe(SOURCES, hide_index=True, use_container_width=True,
                     column_config={"meaning": st.column_config.TextColumn(width="large")})
    st.markdown(
        "**Questions they could answer that KT1 can't:**\n"
        "- Do students who change their answer before submitting do better or worse? (KT2)\n"
        "- Does it matter whether a question was self-chosen (sprint), recommended, or a review? (KT2)\n"
        "- Does reading the explanation after a wrong answer, or watching a lecture, improve the next "
        "attempt on that skill? How long do people study explanations? (KT3)\n"
        "- Does replaying the audio help on listening questions? Does crossing out options help? "
        "Do paying users behave differently? (KT4)\n\n"
        "Download links: [KT2](http://bit.ly/ednet-kt2) · [KT3](http://bit.ly/ednet-kt3) · "
        "[KT4](http://bit.ly/ednet-kt4). Full docs: the [EdNet README](https://github.com/riiid/ednet).")

    st.divider()
    st.subheader("What this explorer did to it")
    st.markdown(
        "`prepare.py` reads the 784k student files straight out of the zip (no unpacking), joins every answer "
        "to its question to mark it right or wrong, numbers each student's answers in order, and builds "
        "summaries per question, student, tag, practice stage and time spent. A student's **ability** is how "
        "much more often they're right than the questions' average accuracy predicts (0 = average). Tag "
        "*links* use what's left after removing question difficulty and ability, like the SLAM explorer.")

    st.subheader("Quirks to know")
    st.markdown(
        "- **The app chooses most questions.** Santa recommends questions from its own model of the student, "
        "so who sees which question isn't random, and question accuracy partly reflects who was sent it.\n"
        "- **Survivorship:** students who keep going tend to be stronger or more motivated, so average "
        "accuracy at the 1,000th question describes different people than at the 10th.\n"
        "- **A common starting sequence:** about 27% of students begin with exactly the same questions "
        "(q8098, q8074, q176, …), which looks like a fixed placement test. Their accuracy is low (≈49% on the "
        "first 10 answers vs ≈65% afterwards), and with a median of 11 answers many students never get far "
        "past it. KT1 doesn't record where a question came from; KT2's `source` column (e.g. `diagnosis`) would.\n"
        "- **Bundle timing:** `elapsed_time` is per bundle, repeated on each question.\n"
        "- **Odd answers:** about 27,600 answers aren't a–d (blank or other values); they count as wrong.\n"
        "- **Row count:** 95,293,926 answers here; the EdNet README says 95,294,926 (a 1,000 difference, "
        "likely a typo, as every student file was read).\n"
        "- **Unnamed tags**, **shifted dates**, and **no student background** (see above).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Good for")
        st.markdown(
            "- Which TOEIC parts, questions and skills are hard\n"
            "- Which wrong options attract students (every answer choice is recorded)\n"
            "- How accuracy changes with practice, at huge scale\n"
            "- Time spent vs accuracy\n"
            "- Which skills go together\n"
            "- Knowledge-tracing models (the original purpose)")
    with c2:
        st.subheader("Not good for")
        st.markdown(
            "- Speaking or writing (multiple choice only)\n"
            "- Real dates or seasonality (timestamps shifted)\n"
            "- Anything about who students are\n"
            "- What a skill tag means\n"
            "- Effects of lectures and explanations (needs KT3)")
