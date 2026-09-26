# Duolingo SLAM 2018 explorer

Data: Duolingo Second Language Acquisition Modeling shared task (Settles et al. 2018),
Harvard Dataverse doi:10.7910/DVN/8SWHNO, CC BY-NC 4.0. Raw files in `adults/data/slam/dataverse_files/`.

Tracks are named `<language learned>_<native language>`:
`en_es` Spanish speakers learning English, `es_en` English speakers learning Spanish,
`fr_en` English speakers learning French. Each covers learners' first ~30 days.

```bash
uv run python prepare.py    # raw text -> parquet (~40s)
uv run python patterns.py   # baseline model + word-pair links (~40s)
uv run streamlit run app.py
```

## Findings so far
- morning/evening: no link once learner ability and word difficulty are accounted for (0.007).
- Strongest real links are grammatical families: la/los/las, yo/tú, niños/niñas, l'/j', chiens/chiennes.
- Some strong links (evening/harbor, pepper/career) centre on single prompts with several valid translations
  ("Buenas noches", "El puerto", "mi carrera"). The reference is the correct answer closest to what the learner
  typed, so these likely reflect answers Duolingo did not accept (e.g. British spellings?), not shared vocabulary.
- Learners' actual typed answers are not in the dataset, only per-word right/wrong on the closest correct answer.
