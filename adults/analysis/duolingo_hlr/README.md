# Duolingo spaced-repetition explorer

Data: Settles & Meeder (2016), "A Trainable Spaced Repetition Model for Language Learning", ACL.
Harvard Dataverse doi:10.7910/DVN/N8XJME, CC BY-NC 4.0. 12.9M word reviews by 115k learners,
28 Feb - 12 Mar 2013, six languages. Raw file in `adults/data/duolingo_hlr/`.

```bash
uv run python prepare.py                        # csv.gz -> parquet + summaries (~20s)
uv run streamlit run app.py --server.port 8502
```

## Findings so far
- Forgetting curves are very flat: recall ~91% right after practice, ~86% after 3+ months.
  Duolingo chooses when to review each word, so long gaps mostly happen for words it already judged
  well known. That selection makes memory look far more durable than it is.
- Because of that, per-word half-lives are poorly identified (median fit is years, 23% hit the cap).
  The app ranks words by the plain recall drop between short (<1 day) and long (>1 week) gaps instead.
- More prior practice only slightly flattens the curve at long gaps (3 mo+: 84.6% with 1-2 prior
  views vs ~86.5% with 6+), consistent with the same selection effect.

`lexeme_reference.txt` (word-tag glossary used by the Overview tab) is copied from
[duolingo/halflife-regression](https://github.com/duolingo/halflife-regression),
Copyright (c) 2016 Duolingo, Inc., MIT License.
