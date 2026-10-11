"""Gradient-boosted trees over the same learner profile (the approach of the top SLAM 2018 systems).

Learner-side information still comes only from the profile; the trees just read many more numbers from it
and combine them non-linearly, together with what is known about the task (POS, dependency label, position,
neighbouring words, client, session) before the learner answers.
"""

from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Task
from .predictors import Predictor, logit, replay, sigmoid
from .profile import LearnerProfile

NAN = float("nan")


class ContextStats:
    """How often other learners (train split) got a token wrong in a given context: this word in this format,
    this exact sentence + position + format, or next to this previous / next word. Task-side population
    knowledge, like word difficulty."""

    def __init__(self, train):
        self.all, self.own = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for task, y in zip(train.tasks, train.outcomes):
            for key, yi in zip(self.keys(task), y.tolist()):
                for k in key:
                    for c in (self.all[k], self.own[task.user, k]):
                        c[0] += 1
                        c[1] += yi

    @staticmethod
    def keys(task):
        sent = " ".join(task.words)
        prev, nxt = ["<s>"] + task.words[:-1], task.words[1:] + ["</s>"]
        f = task.format
        return [(("F", f, b), ("S", f, sent, i), ("B", a, b), ("N", b, c))
                for i, (a, b, c) in enumerate(zip(prev, task.words, nxt))]

    def lookup(self, key, user) -> tuple[float, float]:
        n, w = self.all.get(key, (0, 0))
        un, uw = self.own.get((user, key), (0, 0))
        return n - un, w - uw


class ProfileGBDT(Predictor):
    name = "profile_gbdt"
    description = ("Gradient-boosted trees on ~40 numbers: read from the same profile (rates, counts, recency for word / "
                   "lemma / morphology, vocabulary size, time since last session; updated with 3-day decay), plus how hard "
                   "this token is for other learners in this format / sentence / word pair, plus task context (POS, "
                   "dependency, position, client, session).")

    numeric = ["pop_word", "log_word_freq", "ability", "log_n_all", "days", "gap_all", "log_vocab", "expected",
               "word_n", "word_wrong", "word_dev", "word_gap", "lemma_n", "lemma_dev", "lemma_gap",
               "morph_mean", "morph_min", "morph_max", "morph_min_n", "format_dev",
               "position", "sent_len", "word_len", "prev_pop", "next_pop", "prev_dev", "next_dev",
               "word_format_dev", "log_word_format_n", "sent_tok_dev", "log_sent_tok_n", "bigram_dev", "log_bigram_n", "next_bigram_dev", "log_next_bigram_n"]
    categorical = ["pos", "dep", "client", "session", "format"]

    def __init__(self, updater=None, **params):
        self.updater = updater          # None: use the runner's default (CountUpdater)
        self.params = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                           l2_regularization=1.0, early_stopping=False, random_state=0)
        self.params.update(params)
        self.codes = {c: {} for c in self.categorical}
        self.fitting = False

    def code(self, kind: str, value: str) -> int:
        d = self.codes[kind]
        if value not in d:
            d[value] = len(d) if self.fitting else -1   # unseen category at test time -> missing
        return d[value]

    def fit(self, train, pop, updater):
        self.pop = pop
        self.ctx = ContextStats(train)
        self.fitting = True
        X, y = [], []

        def collect(profile, task, wrong):
            X.append(self.featurize(profile, task))
            y.append(wrong)

        replay(train, self.updater or updater, collect)
        X = np.vstack(X)
        n_num = len(self.numeric)
        self.model = HistGradientBoostingClassifier(
            categorical_features=list(range(n_num, n_num + len(self.categorical))), **self.params)
        self.model.fit(X, np.concatenate(y))
        self.fitting = False

    def word_freq(self, word: str, user: str) -> float:
        return self.pop.word.get(word, (0, 0))[0] - self.pop.user_word.get((user, word), (0, 0))[0]

    def context(self, key, user, prior_logit, k: float = 5.0) -> tuple[float, float]:
        """(log-odds deviation of the context's error rate from the word's, log count) for a context key."""
        n, w = self.ctx.lookup(key, user)
        prior = sigmoid(prior_logit)
        return logit((w + k * prior) / (n + k)) - prior_logit, np.log1p(n)

    def featurize(self, p: LearnerProfile, task: Task) -> np.ndarray:
        base, u = self.pop.base, p.user_id
        r_all = p.overall.rate(base)
        l_all, l_base = logit(r_all), logit(base)
        gap_all = task.days - p.overall.last_seen if p.overall.last_seen is not None else NAN
        f = p.format.get(task.format)
        format_dev = logit(f.rate(r_all)) - l_all if f else NAN
        n = len(task.words)
        cats = [self.code("client", task.client), self.code("session", task.session), self.code("format", task.format)]

        l_pop = [logit(self.pop.word_rate(w, u)) for w in task.words]
        word_dev = []
        for w, lp in zip(task.words, l_pop):
            ws = p.word.get(w)
            exp = sigmoid(l_all + lp - l_base)
            word_dev.append(logit(ws.rate(exp)) - logit(exp) if ws else NAN)

        rows = []
        ctx_keys = ContextStats.keys(task)
        for i, (w, lemma, morph) in enumerate(zip(task.words, task.lemmas, task.morph)):
            ws, ls = p.word.get(w), p.lemma.get(lemma)
            md = [(logit(p.morph[m].rate(r_all)) - l_all, p.morph[m].n_seen) for m in morph if m in p.morph]
            rows.append([
                l_pop[i], np.log1p(self.word_freq(w, u)), l_all, np.log1p(p.overall.n_seen), task.days, gap_all,
                np.log1p(len(p.word)), l_all + l_pop[i] - l_base,
                ws.n_seen if ws else 0.0, ws.n_wrong if ws else 0.0, word_dev[i],
                task.days - ws.last_seen if ws else NAN,
                ls.n_seen if ls else 0.0, logit(ls.rate(r_all)) - l_all if ls else NAN,
                task.days - ls.last_seen if ls else NAN,
                np.mean([d for d, _ in md]) if md else NAN, min((d for d, _ in md), default=NAN),
                max((d for d, _ in md), default=NAN), min((c for _, c in md), default=0.0), format_dev,
                i, n, len(w),
                l_pop[i - 1] if i > 0 else NAN, l_pop[i + 1] if i < n - 1 else NAN,
                word_dev[i - 1] if i > 0 else NAN, word_dev[i + 1] if i < n - 1 else NAN,
                *(x for key in ctx_keys[i] for x in self.context(key, u, l_pop[i])),
                self.code("pos", task.pos[i]), self.code("dep", task.dep[i]), *cats,
            ])
        X = np.array(rows, dtype=float)
        cat = X[:, len(self.numeric):]
        cat[cat == -1] = NAN   # category never seen in training -> missing
        return X

    def predict(self, profile, task):
        return self.model.predict_proba(self.featurize(profile, task))[:, 1]
