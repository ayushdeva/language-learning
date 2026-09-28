"""Summary page: the four datasets side by side, what each holds and what it's good for.

Static text only, so it renders even if a dataset's prepare.py hasn't been run.
"""

import streamlit as st

st.title("The four datasets at a glance")
st.caption("One row per dataset. Each has its own page in the sidebar with an Overview tab and explorers.")

st.markdown("""
| Dataset | What's in it | Good for |
|---|---|---|
| **Duolingo SLAM 2018**<br>~6,400 new Duolingo users, first ~30 days, 3 courses | Every word of every exercise, marked right or wrong, with exercise type, time, device and country | Which words and grammar learners get wrong; predicting future mistakes; how errors change over the first month |
| **Duolingo spaced repetition**<br>12.9M reviews, 115k learners, 2 weeks, 6 languages | One row per word per session: time since last practice, past record with the word, recall now | Forgetting curves; how practice history affects recall; testing review-scheduling models |
| **speechocean762**<br>5,000 recordings, 250 Mandarin speakers (half children) | English sentences read aloud, scored by 5 experts per sentence, word and sound | Pronunciation scoring; which sounds are hard for Mandarin speakers; children vs adults |
| **EdNet (KT1)**<br>95M answers, 784k Korean students, 13k questions | Every multiple-choice answer on a TOEIC-prep app: option picked, correct or not, time spent (no question text) | Predicting the next answer (knowledge tracing); question difficulty; progress over months |
""", unsafe_allow_html=True)

st.caption("All four are free for research: SLAM, spaced repetition and EdNet are CC BY-NC 4.0 (non-commercial); "
           "speechocean762 is CC BY 4.0 (commercial use allowed with credit).")
