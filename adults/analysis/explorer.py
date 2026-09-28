"""Language-learning dataset explorer: one app, one page per dataset.

Run from adults/analysis:  uv run streamlit run explorer.py
Each dataset folder has its own prepare.py (see its README) that must have been run first.
"""

import streamlit as st

st.set_page_config(page_title="Language learning explorer", layout="wide")

pages = [
    st.Page("slam/app.py", title="Duolingo SLAM", icon="✍️", url_path="slam"),
    st.Page("duolingo_hlr/app.py", title="Duolingo spaced repetition", icon="🔁", url_path="spaced-repetition"),
    st.Page("speechocean762/app.py", title="SpeechOcean762", icon="🗣️", url_path="speechocean"),
    st.Page("ednet/app.py", title="EdNet (TOEIC)", icon="📝", url_path="ednet"),
]
summary = st.Page("summary.py", title="Summary", icon="📋", url_path="summary", default=True)
st.navigation({"Start here": [summary], "Datasets": pages}).run()
