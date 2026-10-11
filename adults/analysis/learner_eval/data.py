"""Load SLAM into a stream of tasks per learner, using SLAM's own train/dev/test split.

A Task is one exercise: the tokens of the (closest) correct answer, with their
word, lemma and morphology. The outcome is a 0/1 "wrong" label per token.
Each learner's exercises are labelled train, then dev, then test in time order (the last ~10% are test).
Run slam/prepare.py first; this reads its parquet output.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simplemma

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "slam" / "parquet"
TRACKS = ["en_es", "es_en", "fr_en"]


@dataclass
class Task:
    ex_id: str
    user: str
    days: float          # time since the learner started, in days
    format: str          # listen / reverse_tap / reverse_translate
    split: str           # train / dev / test
    words: list[str]
    lemmas: list[str]
    morph: list[tuple[str, ...]]   # per token, e.g. ("NOUN:Number=Plur", "NOUN:Gender=Masc")
    pos: list[str]       # per token part of speech
    dep: list[str]       # per token dependency label
    client: str          # ios / android / web
    session: str         # lesson / practice / test


@dataclass
class Dataset:
    track: str
    tasks: list[Task]            # grouped by learner, chronological within learner
    outcomes: list[np.ndarray]   # per task: 1 = token wrong

    def only(self, *splits: str) -> "Dataset":
        keep = [i for i, t in enumerate(self.tasks) if t.split in splits]
        return Dataset(self.track, [self.tasks[i] for i in keep], [self.outcomes[i] for i in keep])


def morph_kcs(pos: str, morph: str) -> tuple[str, ...]:
    """'Gender=Masc|Number=Sing|fPOS=NOUN++' -> ('NOUN:Gender=Masc', 'NOUN:Number=Sing')"""
    return tuple(f"{pos}:{f}" for f in morph.split("|") if "=" in f and not f.startswith("fPOS="))


def load(track: str, formats: tuple[str, ...] = ("reverse_translate",)) -> Dataset:
    ex = pd.read_parquet(PARQUET / f"{track}_exercises.parquet",
                         columns=["ex_id", "user", "days", "format", "split", "ex_num", "client", "session"])
    ex = ex[ex["format"].isin(formats)]
    tok = pd.read_parquet(PARQUET / f"{track}_tokens.parquet",
                          columns=["ex_id", "pos_in_ex", "word", "pos", "morph", "dep", "wrong"])
    tok = tok.merge(ex, on="ex_id").sort_values(["user", "ex_num", "pos_in_ex"], kind="stable")

    lang = track.split("_")[0]
    lemma_of = {w: simplemma.lemmatize(w, lang=lang) for w in tok["word"].unique()}
    pm = tok["pos"].astype(str) + "\t" + tok["morph"]
    kcs_of = {k: morph_kcs(*k.split("\t")) for k in pm.unique()}

    words = tok["word"].tolist()
    lemmas = [lemma_of[w] for w in words]
    morph = [kcs_of[k] for k in pm]
    pos, dep = tok["pos"].astype(str).tolist(), tok["dep"].astype(str).tolist()
    wrong = tok["wrong"].to_numpy()

    ex_ids = tok["ex_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ex_ids[1:] != ex_ids[:-1]])
    ends = np.r_[starts[1:], len(tok)]
    users, days = tok["user"].astype(str).to_numpy(), tok["days"].to_numpy()
    fmts, splits = tok["format"].astype(str).to_numpy(), tok["split"].astype(str).to_numpy()
    clients, sessions = tok["client"].astype(str).to_numpy(), tok["session"].astype(str).to_numpy()

    tasks, outcomes = [], []
    for s, e in zip(starts, ends):
        tasks.append(Task(ex_ids[s], users[s], float(days[s]), fmts[s], splits[s], words[s:e], lemmas[s:e], morph[s:e],
                          pos[s:e], dep[s:e], clients[s], sessions[s]))
        outcomes.append(wrong[s:e])
    return Dataset(track, tasks, outcomes)

