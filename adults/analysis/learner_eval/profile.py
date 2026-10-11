"""The learner profile, and the rule that updates it from a task + outcome."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .data import Task


@dataclass
class KCStats:
    n_seen: float = 0.0
    n_wrong: float = 0.0
    last_seen: float | None = None   # SLAM `days`

    def rate(self, prior: float, k: float = 5.0) -> float:
        """Error rate smoothed toward `prior` as if k extra tokens had been seen at that rate."""
        return (self.n_wrong + k * prior) / (self.n_seen + k)

    def add(self, wrong: float, days: float, w: float = 1.0):
        self.n_seen += w
        self.n_wrong += w * wrong
        self.last_seen = days


@dataclass
class LearnerProfile:
    user_id: str
    overall: KCStats = field(default_factory=KCStats)
    format: dict[str, KCStats] = field(default_factory=lambda: defaultdict(KCStats))
    word: dict[str, KCStats] = field(default_factory=lambda: defaultdict(KCStats))
    lemma: dict[str, KCStats] = field(default_factory=lambda: defaultdict(KCStats))
    morph: dict[str, KCStats] = field(default_factory=lambda: defaultdict(KCStats))


class CountUpdater:
    """Observation adapter + update rule: every token adds w (seen) and w*wrong to each KC it touches.
    `format_weights` sets how much evidence each task type carries (default 1 for all)."""

    def __init__(self, format_weights: dict[str, float] | None = None):
        self.format_weights = format_weights or {}

    def update(self, p: LearnerProfile, task: Task, wrong: np.ndarray):
        w = self.format_weights.get(task.format, 1.0)
        d = task.days
        fmt = p.format[task.format]
        for word, lemma, morph, y in zip(task.words, task.lemmas, task.morph, wrong.tolist()):
            p.overall.add(y, d, w)
            fmt.add(y, d, w)
            p.word[word].add(y, d, w)
            p.lemma[lemma].add(y, d, w)
            for m in morph:
                p.morph[m].add(y, d, w)


class DecayUpdater(CountUpdater):
    """Same profile, but old evidence fades: before each new observation a KC's counts are multiplied by
    0.5 ** (days since last seen / half_life), so its rate tracks recent performance."""

    def __init__(self, half_life: float, format_weights: dict[str, float] | None = None):
        super().__init__(format_weights)
        self.half_life = half_life

    def update(self, p: LearnerProfile, task: Task, wrong: np.ndarray):
        d = task.days
        decayed = set()
        kcs = [p.overall, p.format[task.format]]
        for word, lemma, morph in zip(task.words, task.lemmas, task.morph):
            kcs += [p.word[word], p.lemma[lemma], *(p.morph[m] for m in morph)]
        for kc in kcs:
            if id(kc) not in decayed and kc.last_seen is not None:
                f = 0.5 ** ((d - kc.last_seen) / self.half_life)
                kc.n_seen *= f
                kc.n_wrong *= f
            decayed.add(id(kc))
        super().update(p, task, wrong)
