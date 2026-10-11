# Learner-profile eval (SLAM)

Plug-and-play evaluation of knowledge tracing on Duolingo SLAM: a **learner profile** is built from a learner's
history by an **update rule**, and a **predictor** reads it to give P(wrong) for each token of the next exercise.

```bash
# from adults/analysis, after slam/prepare.py
uv run python -m learner_eval.run                     # reverse_translate, all 3 tracks
uv run python -m learner_eval.run --formats listen reverse_tap reverse_translate   # all formats (SLAM leaderboard setting)
```

Writes `reports/<format>.md` (the readable report) and `reports/<format>.json`.

| File | Role |
|---|---|
| `data.py` | Loads a track into `Task`s (one per exercise: words, lemmas, morph features, SLAM split) + 0/1 outcomes |
| `profile.py` | `LearnerProfile`: seen / wrong / last-seen counts for overall, format, word, lemma, morph feature. `CountUpdater`: the update rule (with optional per-format evidence weights). `DecayUpdater`: same, but old counts fade with a half-life |
| `predictors.py` | `Population` (average-learner stats from the train split) and the baselines. `replay()`: predict-then-update over each learner's history |
| `boosted.py` | `ProfileGBDT`: gradient-boosted trees over ~40 numbers read from the same profile plus task context; `ContextStats`: how hard a token is in this format / sentence / next to these words, for other learners |
| `metrics.py` | skill (log-loss vs base rate), AUROC, within-exercise AUROC, F1, calibration |
| `run.py` | Runs every model in `MODELS` and writes the report |

To try something new, write a `Predictor` subclass (`fit` on the train split, `predict(profile, task)`) and add it
to `MODELS` in `run.py`; a model can bring its own update rule via an `updater` attribute.

## Setup choices
- SLAM's own split: each learner's exercises are train, then dev, then test (last ~10%) in time order. Every
  learner is replayed from their first exercise (predict, then update the profile); only test exercises are
  scored. Population stats and model weights are fitted on train + dev (everything before test, as in the
  shared task's final phase). `profile_gbdt` was designed and tuned on dev with fitting on train only.
  Only reverse_translate exercises are loaded, so the profile only sees that task type.
- Response time (SLAM's `time` field) is not used: it is measured while the learner answers, so it is part of the
  response, not something known before the task.
- Lemmas come from `simplemma` (lookup table, no context). Morph features are SLAM's tags prefixed by POS
  (`NOUN:Number=Plur`); they are a little noisy (`Somos` is tagged `Tense=Past`).
- `Population` word difficulty leaves out the asking learner's own answers; a learner's own history reaches
  predictions only through their profile.

## `profile_gbdt`: how it works
Same profile fields; two changes:
1. **Update rule** (`DecayUpdater`, half-life 3 days): before adding new evidence, a KC's counts are multiplied
   by 0.5^(days since last seen / 3), so rates track recent performance. Small gain (~+0.002 AUROC on dev).
2. **Prediction**: scikit-learn `HistGradientBoostingClassifier` (400 trees, learning rate 0.05, 31 leaves,
   min 500 samples per leaf) on:
   - learner (from the profile): ability, counts, time since last session, vocabulary size; for this word,
     lemma and morph features: seen / wrong counts, deviation from expected, days since last seen;
     the learner's deviation on the neighbouring words;
   - population (other learners, train data): word difficulty and frequency, and how hard this token is
     in this format, in this exact sentence at this position, and after / before these neighbouring words;
   - task: POS, dependency label, position, sentence length, word length, client, session, format.

Dev-set progression (fr_en, fit on train, score dev): LR 0.772 AUROC; GBDT with the same 10 features 0.770;
GBDT with all profile + task features 0.792; + sentence/bigram context 0.810; + decay 0.812; all formats
with format-aware context 0.845. The trees overfit badly with the library defaults (lr 0.1, 1000 trees).

## Findings so far
Test set, fit on train + dev. AUROC en_es / es_en / fr_en:

| model | reverse_translate only | all formats (leaderboard setting) |
|---|---|---|
| SLAM official baseline (LR) | – | 0.774 / 0.746 / 0.771 |
| `profile_lr` | 0.770 / 0.751 / 0.771 | 0.800 / 0.772 / 0.792 |
| **`profile_gbdt`** | **0.813 / 0.789 / 0.811** | **0.849 / 0.820 / 0.845** |
| SanaLabs, 2018 winner (RNN + GBDT) | – | 0.861 / 0.838 / 0.857 |
| Best published (Hu et al. 2020) | – | 0.864 / 0.839 / 0.860 |

- `profile_gbdt` is ~0.015-0.02 AUROC behind the best published systems, with the same learner profile as
  `profile_lr`. Skill (log-loss vs base rate) rises from +0.15 to +0.21 (reverse_translate) and +0.18 to +0.27
  (all formats); within-exercise AUROC from 0.75 to 0.79-0.80, so it is better at *which* words, not only *who*.
- Biggest single gain: task-side population context (how hard this token is in this sentence / format / next
  to these words, for other learners): +0.02 to +0.03 AUROC on dev. Profile-side extras gave +0.02, decay +0.002.
- Remaining gap is information, not capacity: more trees / leaves do not help on dev. Things the leaders used
  that this setup deliberately leaves out: response time, per-learner learned parameters (user embeddings),
  and per-learner memory of specific sentences (the profile only stores counts per word / lemma / morph).
- All-formats numbers are higher partly because pooled AUROC rewards knowing that reverse_tap is easy.
- Per-learner AUROC is a misleading metric: within a fixed history, a learner's past error rate is mechanically
  anti-correlated with their remaining answers. Within-exercise AUROC is used instead.
