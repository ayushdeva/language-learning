# Language learning

Exploring how people learn languages, in two tracks:

- `children/` - language learning for children
- `adults/` - language learning for adults

Each track has `literature/` (paper notes), `data/` (raw downloads, not in git) and `analysis/`.

## Explorers (adults/analysis)

| App | Dataset | Run |
|---|---|---|
| [slam](adults/analysis/slam) | Duolingo SLAM 2018: word-level right/wrong, first 30 days | `uv run streamlit run app.py` |
| [duolingo_hlr](adults/analysis/duolingo_hlr) | Duolingo spaced repetition 2016: 13M word reviews | `uv run streamlit run app.py --server.port 8502` |
| [speechocean762](adults/analysis/speechocean762) | Mandarin speakers reading English, expert-scored | `uv run streamlit run app.py --server.port 8503` |
| [ednet](adults/analysis/ednet) | EdNet KT1: 95M TOEIC answers by 784k Korean students | `uv run streamlit run app.py --server.port 8504` |

Datasets are not in the repo. Download them into `adults/data/<name>/` as described in each README,
then run that folder's `prepare.py` (and `patterns.py` for slam).
