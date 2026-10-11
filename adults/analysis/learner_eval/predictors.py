"""Prediction rules: given a learner profile and a new task, P(wrong) for each token.

Each predictor may use `Population` (facts about the average learner, from the train split only)
and may fit parameters on the train split in `fit`.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset, Task
from .profile import CountUpdater, LearnerProfile


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Population:
    """Average-learner statistics from the train split. Lookups subtract the asking learner's own tokens:
    population = other learners; the learner's own history reaches predictions only through their profile."""

    def __init__(self, train: Dataset, k: float = 5.0):
        self.k = k
        n = w = 0
        self.word = defaultdict(lambda: [0, 0])          # word -> [seen, wrong]
        self.user_word = defaultdict(lambda: [0, 0])     # (user, word) -> [seen, wrong]
        for task, y in zip(train.tasks, train.outcomes):
            for word, yi in zip(task.words, y.tolist()):
                for c in (self.word[word], self.user_word[task.user, word]):
                    c[0] += 1
                    c[1] += yi
            n += len(y)
            w += int(y.sum())
        self.base = w / n

    def word_rate(self, word: str, user: str) -> float:
        s, w = self.word.get(word, (0, 0))
        us, uw = self.user_word.get((user, word), (0, 0))
        return (w - uw + self.k * self.base) / (s - us + self.k)


class Predictor:
    name = ""
    description = ""

    def fit(self, train: Dataset, pop: Population, updater: CountUpdater):
        self.pop = pop

    def predict(self, profile: LearnerProfile, task: Task) -> np.ndarray:
        raise NotImplementedError


class Random(Predictor):
    name = "random"
    description = "A uniformly random probability for every token. The floor: knows nothing, and is confidently wrong half the time."

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, profile, task):
        return self.rng.uniform(size=len(task.words))


class BaseRate(Predictor):
    name = "base_rate"
    description = "The overall error rate of this task type in the train split, for every token. The reference (skill = 0)."

    def predict(self, profile, task):
        return np.full(len(task.words), self.pop.base)


class LearnerAccuracy(Predictor):
    name = "learner_past_accuracy"
    description = "This learner's own error rate so far (all words), smoothed toward the base rate. Knows who, not what."

    def predict(self, profile, task):
        return np.full(len(task.words), profile.overall.rate(self.pop.base))


class WordDifficulty(Predictor):
    name = "word_difficulty"
    description = "How often other learners got this word wrong (train split). Same for every learner: knows what, not who."

    def predict(self, profile, task):
        return np.array([self.pop.word_rate(w, profile.user_id) for w in task.words])


class AbilityPlusDifficulty(Predictor):
    name = "ability_x_difficulty"
    description = ("Combine the two above on the log-odds scale: a weak learner on a hard word gets both penalties. "
                   "Nothing is fitted.")

    def predict(self, profile, task):
        lb = logit(self.pop.base)
        la = logit(profile.overall.rate(self.pop.base))
        return np.array([sigmoid(la + logit(self.pop.word_rate(w, profile.user_id)) - lb) for w in task.words])


class ProfileLR(Predictor):
    name = "profile_lr"
    description = ("Logistic regression on features read from the full profile: word difficulty, learner ability, how the "
                   "learner did on this word / lemma / morphology before, and how long since they last saw the word.")

    features = ["pop_word", "ability", "format_dev", "word_dev", "lemma_dev", "morph_dev",
                "word_unseen", "log_gap", "log_n_word", "gap_x_n"]

    def fit(self, train, pop, updater):
        self.pop = pop
        X, y = [], []

        def collect(profile, task, wrong):
            X.append(self.featurize(profile, task))
            y.append(wrong)

        replay(train, updater, collect)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.model.fit(np.vstack(X), np.concatenate(y))
        self.coef = dict(zip(self.features, self.model[-1].coef_[0].round(3)))

    def featurize(self, p: LearnerProfile, task: Task) -> np.ndarray:
        base = self.pop.base
        r_all = p.overall.rate(base)
        l_all = logit(r_all)
        f = p.format.get(task.format)
        format_dev = logit(f.rate(r_all)) - l_all if f else 0.0
        rows = []
        for word, lemma, morph in zip(task.words, task.lemmas, task.morph):
            l_pop = logit(self.pop.word_rate(word, p.user_id))
            expected = sigmoid(l_all + l_pop - logit(base))   # ability x difficulty
            ws = p.word.get(word)
            if ws:
                word_dev = logit(ws.rate(expected)) - logit(expected)
                gap, n = np.log1p(task.days - ws.last_seen), np.log1p(ws.n_seen)
            else:
                word_dev, gap, n = 0.0, 0.0, 0.0
            ls = p.lemma.get(lemma)
            lemma_dev = logit(ls.rate(r_all)) - l_all if ls else 0.0
            md = [logit(p.morph[m].rate(r_all)) - l_all for m in morph if m in p.morph]
            rows.append([l_pop, l_all, format_dev, word_dev, lemma_dev, np.mean(md) if md else 0.0,
                         float(ws is None), gap, n, gap * n])
        return np.array(rows)

    def predict(self, profile, task):
        return self.model.predict_proba(self.featurize(profile, task))[:, 1]


BASELINES = [Random, BaseRate, LearnerAccuracy, WordDifficulty, AbilityPlusDifficulty, ProfileLR]


def replay(ds: Dataset, updater: CountUpdater, on_task):
    """Walk each learner's tasks in time order: call on_task(profile, task, wrong) BEFORE the profile sees
    the task, then update. Profiles start empty for every learner."""
    profile = None
    for task, wrong in zip(ds.tasks, ds.outcomes):
        if profile is None or profile.user_id != task.user:
            profile = LearnerProfile(task.user)
        on_task(profile, task, wrong)
        updater.update(profile, task, wrong)
