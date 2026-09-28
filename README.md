# Language learning

Exploring how people learn languages, in two tracks:

- `children/` - language learning for children
- `adults/` - language learning for adults

Each track has `literature/` (paper notes), `data/` (raw downloads, not in git) and `analysis/`.

## Explorer (adults/analysis)

One Streamlit app with a page per dataset; pick the dataset in the left panel.

```bash
cd adults/analysis
uv run streamlit run explorer.py        # http://localhost:8501
```

| Page | Dataset |
|---|---|
| [Duolingo SLAM](adults/analysis/slam) | Duolingo SLAM 2018: word-level right/wrong, learners' first 30 days |
| [Duolingo spaced repetition](adults/analysis/duolingo_hlr) | Duolingo half-life regression 2016: 13M word reviews |
| [SpeechOcean762](adults/analysis/speechocean762) | Mandarin speakers reading English aloud, expert-scored |
| [EdNet (TOEIC)](adults/analysis/ednet) | EdNet KT1: 95M TOEIC answers by 784k Korean students |

Datasets are not in the repo. Download each into `adults/data/<name>/` as described in its folder's README,
then run that folder's `prepare.py` from `adults/analysis` (e.g. `uv run python slam/prepare.py`; SLAM also
needs `uv run python slam/patterns.py`). All pages share one environment (`adults/analysis/pyproject.toml`).
