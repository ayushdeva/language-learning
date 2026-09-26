"""Find word pairs whose mastery goes together, beyond general ability.

Step 1 - baseline model. A logistic regression predicts whether each token is
answered wrong from: who the learner is (ability), which word it is
(difficulty), the exercise format (listen / translate / tap) and a per-learner
ability for each format. This captures "strong learners get everything right",
"some words are hard" and "some learners are much better at reading than
listening" (without the last term, words mostly heard in listening exercises
link up with each other spuriously).

Step 2 - residuals. For every token, residual = actual - predicted. A learner
with a negative average residual on a word did *better* on it than their
ability and the word's difficulty predict.

Step 3 - pair links. For each pair of common words, correlate the per-learner
average residuals across learners who saw both. A positive link means: learners
who do unexpectedly well on word A also tend to do unexpectedly well on word B.

Two confounds are reported alongside each pair rather than hidden:
  same_ex_share      how often the two words appear in the same exercise
                     (one botched sentence makes both words "wrong" together)
  same_first_session how often learners met both words for the first time in
                     the same session (Duolingo teaches them together)

Outputs (in adults/data/slam/parquet/):
  {track}_users.parquet      learner ability
  {track}_words.parquet      word difficulty and counts
  {track}_user_word.parquet  per learner x word summary
  {track}_pairs.parquet      pairwise links between common words
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "data" / "slam" / "parquet"
TRACKS = ["en_es", "es_en", "fr_en"]
MIN_USERS_PER_WORD = 100  # only words seen by at least this many learners get pair links
MAX_WORDS = 1200
MIN_SHARED = 30  # a pair needs at least this many learners who saw both words


def fit_baseline(m: pd.DataFrame):
    u = m["user"].cat.codes.to_numpy()
    w = m["word_code"].to_numpy()
    f = m["format"].cat.codes.to_numpy()
    nu, nw, nf = u.max() + 1, w.max() + 1, f.max() + 1
    uf = u * nf + f  # learner x format: some learners listen much better than they read
    rows = np.arange(len(m))
    one = np.ones(len(m))
    X = sp.hstack([
        sp.csr_matrix((one, (rows, u)), shape=(len(m), nu)),
        sp.csr_matrix((one, (rows, w)), shape=(len(m), nw)),
        sp.csr_matrix((one, (rows, f)), shape=(len(m), nf)),
        sp.csr_matrix((one, (rows, uf)), shape=(len(m), nu * nf)),
    ]).tocsr()
    clf = LogisticRegression(C=1.0, max_iter=300)
    clf.fit(X, m["wrong"].to_numpy())
    p = clf.predict_proba(X)[:, 1]
    coef = clf.coef_[0]
    # positive coef = more errors, so ability is the negated user coefficient
    return p, -coef[:nu], coef[nu:nu + nw]


def pairwise_corr(X: np.ndarray, M: np.ndarray):
    """Pearson correlation between columns using only rows where both are present."""
    X = np.where(M > 0, X, 0.0)
    n = M.T @ M
    sx = X.T @ M  # sx[a, b] = sum of x_a over rows that also have b
    sxx = (X * X).T @ M
    sxy = X.T @ X
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sx.T / n
        cov = sxy / n - mx * my
        va = sxx / n - mx ** 2
        vb = sxx.T / n - my ** 2
        r = cov / np.sqrt(va * vb)
    return n, r


def run(track: str):
    ex = pd.read_parquet(PQ / f"{track}_exercises.parquet")
    tok = pd.read_parquet(PQ / f"{track}_tokens.parquet", columns=["ex_id", "word", "wrong"])
    m = tok.merge(ex[["ex_id", "user", "format", "days"]], on="ex_id")
    m["session"] = m["ex_id"].str[:8]
    m["word"] = m["word"].astype("category")
    m["word_code"] = m["word"].cat.codes
    words = m["word"].cat.categories

    print(f"[{track}] fitting baseline on {len(m):,} tokens ...")
    p, ability, difficulty = fit_baseline(m)
    m["pred"] = p
    m["resid"] = m["wrong"] - p

    users = pd.DataFrame({"user": m["user"].cat.categories, "ability": ability})
    ustats = m.groupby("user", observed=True).agg(
        n_tokens=("wrong", "size"), error_rate=("wrong", "mean"),
        n_words=("word_code", "nunique"), last_day=("days", "max"))
    users = users.merge(ustats, left_on="user", right_index=True)
    users.to_parquet(PQ / f"{track}_users.parquet", index=False)

    # per learner x word
    m = m.sort_values(["user", "days"], kind="stable")
    uw = m.groupby(["user", "word_code"], observed=True).agg(
        n_seen=("wrong", "size"), n_wrong=("wrong", "sum"),
        mean_resid=("resid", "mean"), mean_pred=("pred", "mean"),
        first_day=("days", "first"), first_wrong=("wrong", "first"),
        first_session=("session", "first"),
    ).reset_index()
    uw["word"] = words[uw["word_code"]]

    wstats = m.groupby("word_code").agg(
        n_tokens=("wrong", "size"), error_rate=("wrong", "mean"),
        n_users=("user", "nunique"), median_first_day=("days", "median"))
    wdf = pd.DataFrame({"word": words, "difficulty": difficulty}).join(wstats)
    wdf.to_parquet(PQ / f"{track}_words.parquet", index=False)
    uw.drop(columns=["first_session"]).to_parquet(PQ / f"{track}_user_word.parquet", index=False)

    # ---- pair links among common words
    common = wdf[wdf.n_users >= MIN_USERS_PER_WORD].nlargest(MAX_WORDS, "n_users")
    codes = common.index.to_numpy()
    col = pd.Series(np.arange(len(codes)), index=codes)
    sub = uw[uw.word_code.isin(codes)].copy()
    sub["c"] = col[sub.word_code].to_numpy()
    ucode = sub["user"].cat.codes.to_numpy()
    nU, nW = ucode.max() + 1, len(codes)
    X = np.zeros((nU, nW))
    M = np.zeros((nU, nW))
    X[ucode, sub.c] = sub.mean_resid
    M[ucode, sub.c] = 1
    print(f"[{track}] pair links over {nW} words x {nU} learners ...")
    n, r = pairwise_corr(X, M)

    # confound 1: same exercise
    mt = m[m.word_code.isin(codes)]
    ex_codes = mt["ex_id"].astype("category").cat.codes.to_numpy()
    B = sp.csr_matrix((np.ones(len(mt)), (ex_codes, col[mt.word_code].to_numpy())),
                      shape=(ex_codes.max() + 1, nW))
    B.data[:] = 1
    B = B.tocsr()
    B.sum_duplicates()
    B.data[:] = 1
    co_ex = (B.T @ B).toarray()
    ex_count = np.diag(co_ex)
    same_ex_share = co_ex / np.minimum.outer(ex_count, ex_count)

    # confound 2: first met in the same session
    fs = sub["user"].astype(str) + "|" + sub["first_session"]
    fs_codes = fs.astype("category").cat.codes.to_numpy()
    A = sp.csr_matrix((np.ones(len(sub)), (fs_codes, sub.c)), shape=(fs_codes.max() + 1, nW))
    same_first = (A.T @ A).toarray()

    ia, ib = np.triu_indices(nW, k=1)
    keep = n[ia, ib] >= MIN_SHARED
    ia, ib = ia[keep], ib[keep]
    pairs = pd.DataFrame({
        "word_a": words[codes[ia]], "word_b": words[codes[ib]],
        "n_shared": n[ia, ib].astype(int),
        "link": r[ia, ib],
        "same_ex_share": same_ex_share[ia, ib],
        "same_first_session": same_first[ia, ib] / n[ia, ib],
    }).dropna(subset=["link"])
    # rough standard error of a correlation, to separate signal from noise
    pairs["z"] = pairs["link"] * np.sqrt(pairs["n_shared"] - 3)
    pairs.to_parquet(PQ / f"{track}_pairs.parquet", index=False)
    print(f"[{track}] {len(pairs):,} pairs saved")


if __name__ == "__main__":
    for t in sys.argv[1:] or TRACKS:
        run(t)
