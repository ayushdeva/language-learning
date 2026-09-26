"""Overview tab for the pronunciation explorer: what the dataset is and how to read it."""

import altair as alt
import pandas as pd
import streamlit as st

SENTENCE = pd.DataFrame([
    ("accuracy", "0–10", "How correct the pronunciation is overall. 9–10 excellent, no obvious mistakes; "
     "7–8 good, a few mistakes; 5–6 understandable but many mistakes and accent; 3–4 poor, serious "
     "mistakes; 0–2 only one or two words recognisable."),
    ("fluency", "0–10", "Smoothness. 8–10 no noticeable pauses or stammering; 6–7 a few pauses or "
     "repetitions; 4–5 many; 0–3 very broken up."),
    ("prosodic", "0–10", "Intonation, rhythm and speed ('prosody'). 9–10 native-like cadence; 7–8 nearly "
     "correct; 5–6 unstable speed, poor rhythm; 3–4 too fast or slow, no rhythm; 0–2 can't read the "
     "sentence through."),
    ("completeness", "0–10", "Share of the words pronounced well enough to count (the dataset README says "
     "0–1 but the files use 0–10)."),
    ("total", "0–10", "Overall score, combining the above."),
], columns=["score", "range", "what the experts judged"])

WORD = pd.DataFrame([
    ("accuracy", "0–10", "10 perfect; 7–9 most sounds right but accented; 4–6 under 30% of sounds wrong; "
     "2–3 over 30% wrong, or said as a different word ('bag' → 'bike'); 1 hard to make out; 0 silent."),
    ("stress", "5 or 10", "10 = stress on the right syllable (or one-syllable word); 5 = wrong stress."),
    ("total", "0–10", "Overall word score."),
], columns=["score", "range", "what the experts judged"])

PHONE = pd.DataFrame([
    (2, "correct", "B EH R", "plain"),
    (1, "right sound, heavy accent", "B {EH} R", "{ }"),
    (0, "wrong or missing", "B (EH) R", "( )"),
    ("–", "extra sound inserted", "B EH [L] R", "[ ]"),
], columns=["score", "meaning", "example in expert notes", "mark"])

ARPABET = pd.DataFrame([
    ("AA", "vowel", "father"), ("AE", "vowel", "cat"), ("AH", "vowel", "but, about"),
    ("AO", "vowel", "caught"), ("AW", "vowel", "cow"), ("AY", "vowel", "bite"), ("EH", "vowel", "bed"),
    ("ER", "vowel", "bird"), ("EY", "vowel", "bait"), ("IH", "vowel", "bit"), ("IY", "vowel", "beet"),
    ("OW", "vowel", "boat"), ("OY", "vowel", "boy"), ("UH", "vowel", "book"), ("UW", "vowel", "boot"),
    ("B", "consonant", "buy"), ("CH", "consonant", "church"), ("D", "consonant", "die"),
    ("DH", "consonant", "this"), ("F", "consonant", "fan"), ("G", "consonant", "go"),
    ("HH", "consonant", "hat"), ("JH", "consonant", "judge"), ("K", "consonant", "cat"),
    ("L", "consonant", "let"), ("M", "consonant", "man"), ("N", "consonant", "no"),
    ("NG", "consonant", "sing"), ("P", "consonant", "pan"), ("R", "consonant", "red"),
    ("S", "consonant", "sit"), ("SH", "consonant", "she"), ("T", "consonant", "top"),
    ("TH", "consonant", "thin"), ("V", "consonant", "van"), ("W", "consonant", "win"),
    ("Y", "consonant", "yes"), ("Z", "consonant", "zoo"), ("ZH", "consonant", "measure"),
], columns=["symbol", "type", "as in"])


def render(spk, utt, words, ph, ins, vowels):
    st.header("speechocean762: what's in this dataset")
    st.markdown(
        "**5,000 recordings of English sentences read aloud by 250 native Mandarin speakers**, about half "
        "of them children. **Five experts** independently scored every recording: the whole sentence, "
        "each word, and each individual speech sound. It was released as a free benchmark for "
        "*automatic pronunciation scoring*: software that listens to a learner and says what they got wrong.")

    m = st.columns(6)
    m[0].metric("Speakers", len(spk))
    m[1].metric("Recordings", f"{len(utt):,}")
    m[2].metric("Words scored", f"{len(words):,}")
    m[3].metric("Sounds scored", f"{len(ph):,}")
    m[4].metric("Distinct sentences", f"{utt.text.nunique():,}")
    m[5].metric("Distinct words", f"{words.word.nunique():,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Source")
        st.markdown(
            "- **Paper:** J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, Y. Wang "
            "(2021). *speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation "
            "Assessment.* Proc. Interspeech 2021. [arXiv:2104.01378](https://arxiv.org/abs/2104.01378)\n"
            "- **Data:** OpenSLR resource 101, [openslr.org/101](https://www.openslr.org/101/), one 521 MB "
            "archive. No signup needed.\n"
            "- **License:** CC BY 4.0: free for any use, including commercial, with credit.\n"
            "- **Baseline:** a scoring system published in the Kaldi speech toolkit, described in the paper.")
    with c2:
        st.subheader("Who is in it")
        g = spk.groupby("group", observed=True).agg(speakers=("speaker", "size"),
                                                    ages=("age", lambda a: f"{a.min()}–{a.max()}"),
                                                    female=("gender", lambda x: (x == "f").mean()))
        st.dataframe(g.reset_index(), hide_index=True, use_container_width=True,
                     column_config={"female": st.column_config.NumberColumn("share female", format="%.2f")})
        st.markdown(
            "- Everyone's first language is **Mandarin Chinese**.\n"
            "- Only age and gender are recorded: nothing on English level, years of study or region.\n"
            f"- Split into **train** and **test** halves of {int((spk.split == 'train').sum())} speakers and "
            f"{int((utt.split == 'train').sum()):,} recordings each (for training and testing scoring "
            "software). This explorer uses both together.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Speakers by age**")
        st.altair_chart(alt.Chart(spk).mark_bar().encode(
            x=alt.X("age:O", title="Age"), y=alt.Y("count():Q", title="Speakers"),
            color=alt.Color("gender:N", title="Gender"), tooltip=["age", "gender", "count()"]
        ).properties(height=240), use_container_width=True)
        st.caption("No speakers aged 16–18. Every speaker has exactly 20 recordings.")
    with c2:
        st.markdown("**Sentence length by group**")
        u = utt.assign(words=utt.text.str.split().str.len())
        st.altair_chart(alt.Chart(u).mark_bar(opacity=0.7).encode(
            x=alt.X("words:O", title="Words in sentence"), y=alt.Y("count():Q", title="Recordings",
                                                                  stack=None),
            color=alt.Color("group:N", title=None)).properties(height=240), use_container_width=True)
        st.caption("Children were given shorter sentences, and the groups share almost no sentences.")

    st.divider()
    st.subheader("What they read")
    st.markdown("A random sample of sentences from each group:")
    c1, c2 = st.columns(2)
    for col, grp in zip([c1, c2], ["child (6-12)", "adult (19+)"]):
        s = utt[utt.group == grp].text.drop_duplicates().sample(8, random_state=1)
        col.markdown(f"**{grp}**\n" + "\n".join(f"- {t.capitalize()}" for t in s))
    shared = set(utt[utt.group == "child (6-12)"].text) & set(utt[utt.group == "adult (19+)"].text)
    st.caption(f"Sentences read by both children and adults: {len(shared)}. Some sentences are read by "
               f"several speakers within a group ({utt.text.value_counts().gt(1).sum():,} sentences appear "
               "more than once).")

    st.divider()
    st.subheader("How it was scored")
    st.markdown(
        "Five experts scored each recording independently with the same rubric, at three levels. The "
        "released scores are the **average or median of the five**; each expert's own marks are kept "
        "in a second file, which this explorer uses for the 'how many experts said wrong' counts.")
    st.markdown("**Sentence level**")
    st.dataframe(SENTENCE, hide_index=True, use_container_width=True,
                 column_config={"what the experts judged": st.column_config.TextColumn(width="large")})
    st.markdown("**Word level**")
    st.dataframe(WORD, hide_index=True, use_container_width=True,
                 column_config={"what the experts judged": st.column_config.TextColumn(width="large")})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sound (phone) level**")
        st.dataframe(PHONE, hide_index=True, use_container_width=True)
        st.caption("Experts marked each expected sound and could add sounds they heard that shouldn't be "
                   "there. 'B EH [L] R' = an extra L in 'bear'.")
    with c2:
        st.markdown("**Score distributions**")
        long = utt.melt(value_vars=["accuracy", "fluency", "prosodic", "total"], var_name="score")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("value:Q", bin=alt.Bin(step=1), title="Sentence score"),
            y=alt.Y("count():Q", title=None), color=alt.Color("score:N", legend=None),
            row=alt.Row("score:N", title=None)).properties(height=50, width=380))
        st.caption("About three-quarters of sentence scores are 7, 8 or 9, and few recordings score very low, "
                   "so most differences between speakers are small.")

    st.subheader("How speech sounds are written")
    st.markdown(
        "Sounds use **ARPAbet**, the symbol set of the CMU Pronouncing Dictionary. Vowels carry a stress "
        "digit: **0** unstressed, **1** main stress, **2** secondary stress (e.g. ABILITY = "
        "AH0 B IH1 L AH0 T IY0). The Sound tab ignores the digit and groups e.g. AH0 and AH1 as AH.")
    counts = ph.base.value_counts()
    a = ARPABET.assign(times=ARPABET.symbol.map(counts).fillna(0).astype(int),
                       avg_score=ARPABET.symbol.map(ph.groupby("base").score.mean()))
    c1, c2 = st.columns(2)
    for col, kind in zip([c1, c2], ["vowel", "consonant"]):
        col.markdown(f"**{kind.capitalize()}s**")
        col.dataframe(a[a.type == kind].drop(columns="type"), hide_index=True, use_container_width=True,
                      column_config={"avg_score": st.column_config.NumberColumn("avg score (0-2)",
                                                                              format="%.2f")})

    st.divider()
    st.subheader("Files and what this explorer did")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**In the download**\n"
            "- `WAVE/SPEAKERxxxx/*.WAV`: the recordings (16 kHz, mono, 16-bit)\n"
            "- `train/` and `test/`: speaker ages and genders, recording lists and transcripts "
            "(Kaldi toolkit format)\n"
            "- `resource/scores.json`: combined scores at all three levels\n"
            "- `resource/scores-detail.json`: each of the five experts' marks\n"
            "- `resource/lexicon.txt`: pronunciation dictionary for the words used")
    with c2:
        st.markdown(
            "**What `prepare.py` does**\n"
            "- Flattens everything into five tables: speakers, recordings, words, sounds and inserted "
            "sounds\n"
            "- For each sound, counts how many of the five experts marked it correct, accented or wrong\n"
            "- Records where each sound sits in its word (first, middle, last)\n"
            "- Groups speakers: child 6–12, teen 13–15, adult 19+\n"
            f"- An extra sound is shown in the Speaker tab if at least 2 of 5 experts heard it "
            f"({len(ins):,} individual expert marks in total)")

    st.divider()
    st.subheader("Quirks to know")
    spread = (utt.expert_total_max - utt.expert_total_min).median()
    st.markdown(
        "- **Children and adults read different sentences**, and children's are shorter. Comparing "
        "their raw scores compares the material as much as the people. The app's 'matched' views only use "
        "words both groups read.\n"
        f"- **Experts disagree:** on sentence totals the highest and lowest expert differ by a median "
        f"{spread:.1f} points out of 10. Treat small score differences as noise.\n"
        "- **Graders may treat children differently.** A child's voice is easy to recognise; any leniency "
        "towards children can't be separated from real skill.\n"
        "- **Read speech only:** people read given sentences aloud, which is easier than speaking freely.\n"
        "- **Completeness scale:** 0–10 in the data, not 0–1 as the README says.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Good for")
        st.markdown(
            "- Which English sounds Mandarin speakers find hard, and where in words\n"
            "- Hearing specific mistakes (every score links to audio)\n"
            "- Children vs adults on the same words\n"
            "- How consistent human pronunciation judges are\n"
            "- Building or testing automatic pronunciation scoring")
    with c2:
        st.subheader("Not good for")
        st.markdown(
            "- Learners with other first languages\n"
            "- Progress over time (one set of 20 recordings per speaker, no follow-up)\n"
            "- Spontaneous speech or conversation\n"
            "- Why a speaker is good or bad (no background on their learning)")
