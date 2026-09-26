"""Turn the Duolingo half-life regression dataset (Settles & Meeder 2016) into
Parquet plus a few small summary tables for the explorer.

Each raw row = one word (lexeme) practised by one learner in one session:
  p_recall          share of this session's exercises with the word answered right
  delta             seconds since the learner last practised this word
  history_seen/correct   times seen / answered right before this session
  session_seen/correct   times seen / answered right in this session

Outputs (adults/data/duolingo_hlr/parquet/):
  traces.parquet   all rows, sorted by learner and time, with parsed word fields
  curve.parquet    recall by time-since-last-practice x amount of prior practice
  words.parquet    per word: counts, recall, fitted half-life
  users.parquet    per learner summary
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "duolingo_hlr"
RAW = DATA / "settles.acl16.learning_traces.13m.csv.gz"
OUT = DATA / "parquet"

# time since last practice, in days
DELTA_EDGES = [0, 10 / 1440, 1 / 24, 6 / 24, 1, 2, 4, 7, 14, 30, 90, 1e9]
DELTA_LABELS = ["<10 min", "10-60 min", "1-6 h", "6-24 h", "1-2 d", "2-4 d", "4-7 d",
                "1-2 wk", "2-4 wk", "1-3 mo", "3 mo+"]
# how many times the word was seen before this session
HIST_EDGES = [0, 2, 5, 10, 25, 1e9]
HIST_LABELS = ["1-2", "3-5", "6-10", "11-25", "26+"]


def case(col: str, edges, labels) -> str:
    parts = [f"WHEN {col} <= {hi} THEN '{lab}'" for hi, lab in zip(edges[1:], labels)]
    return "CASE " + " ".join(parts) + " END"


def fit_half_life(g: pd.DataFrame) -> pd.Series:
    """Fit p = a * 2^(-delta/h) over binned points, weighted by count.

    a = recall right after practice (well below 1 on Duolingo: typos, new
    contexts), h = days for recall to halve from there.
    """
    hs = np.logspace(-1, 4, 100)
    as_ = np.linspace(0.5, 1.0, 26)
    d = g["mid_days"].to_numpy()[:, None, None]
    p = g["recall"].to_numpy()[:, None, None]
    w = g["n"].to_numpy()[:, None, None]
    pred = as_[None, :, None] * 2 ** (-d / hs[None, None, :])
    err = (w * (p - pred) ** 2).sum(axis=0)
    ia, ih = np.unravel_index(err.argmin(), err.shape)
    return pd.Series({"recall_at_0": as_[ia], "half_life_days": hs[ih]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    print("reading csv ...")
    con.execute(f"""
        CREATE TABLE t AS
        SELECT
            user_id AS user, learning_language AS lang, ui_language AS ui_lang,
            timestamp AS ts, to_timestamp(timestamp) AS time,
            delta / 86400.0 AS delta_days, p_recall AS recall,
            lexeme_id, lexeme_string,
            split_part(lexeme_string, '/', 1) AS surface,
            regexp_extract(lexeme_string, '/([^<]*)', 1) AS lemma,
            regexp_extract(lexeme_string, '/[^<]*<([^>]*)>', 1) AS pos,
            history_seen, history_correct, session_seen, session_correct
        FROM read_csv_auto('{RAW}')
    """)
    con.execute(f"ALTER TABLE t ADD COLUMN delta_bin VARCHAR")
    con.execute(f"UPDATE t SET delta_bin = {case('delta_days', DELTA_EDGES, DELTA_LABELS)}")
    con.execute(f"ALTER TABLE t ADD COLUMN hist_bin VARCHAR")
    con.execute(f"UPDATE t SET hist_bin = {case('history_seen', HIST_EDGES, HIST_LABELS)}")
    n = con.execute("SELECT count(*) FROM t").fetchone()[0]
    print(f"{n:,} rows")

    con.execute(f"""COPY (SELECT * FROM t ORDER BY user, ts)
                    TO '{OUT / "traces.parquet"}' (FORMAT parquet, ROW_GROUP_SIZE 100000)""")

    con.execute(f"""COPY (
        SELECT lang, delta_bin, hist_bin, count(*) AS n, avg(recall) AS recall,
               median(delta_days) AS mid_days
        FROM t GROUP BY ALL) TO '{OUT / "curve.parquet"}' (FORMAT parquet)""")

    con.execute(f"""COPY (
        SELECT user, any_value(ui_lang) AS ui_lang, string_agg(DISTINCT lang, ',') AS langs,
               count(DISTINCT ts) AS sessions, count(*) AS rows,
               count(DISTINCT lexeme_id) AS words, avg(recall) AS recall,
               min(time) AS first_time, max(time) AS last_time
        FROM t GROUP BY user) TO '{OUT / "users.parquet"}' (FORMAT parquet)""")

    words = con.execute("""
        SELECT lang, lexeme_id, any_value(lexeme_string) AS lexeme_string,
               any_value(surface) AS surface, any_value(lemma) AS lemma, any_value(pos) AS pos,
               count(*) AS rows, count(DISTINCT user) AS users, avg(recall) AS recall
        FROM t GROUP BY lang, lexeme_id""").df()
    binned = con.execute("""
        SELECT lexeme_id, delta_bin, count(*) AS n, avg(recall) AS recall,
               median(delta_days) AS mid_days
        FROM t GROUP BY lexeme_id, delta_bin""").df()
    print("fitting per-word half-lives ...")
    enough = words.loc[words.rows >= 200, "lexeme_id"]
    hl = (binned[binned.lexeme_id.isin(enough)]
          .groupby("lexeme_id").apply(fit_half_life, include_groups=False))
    words = words.merge(hl, left_on="lexeme_id", right_index=True, how="left")
    words.to_parquet(OUT / "words.parquet", index=False)
    binned.to_parquet(OUT / "word_curve.parquet", index=False)
    print("done")


if __name__ == "__main__":
    main()
