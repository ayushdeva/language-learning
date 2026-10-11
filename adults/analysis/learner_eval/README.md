# Learner-profile eval (SLAM)

Plug-and-play evaluation of knowledge tracing on Duolingo SLAM: a **learner profile** is built from a learner's
history by an **update rule**, and a **predictor** reads it to give P(wrong) for each token of the next exercise.

```bash
# from adults/analysis, after slam/prepare.py
uv run python -m learner_eval.run                     # reverse_translate, all 3 tracks (~1 min)
uv run python -m learner_eval.run --formats listen    # another task type
```

Writes `reports/<format>.md` (the readable report) and `reports/<format>.json`.

| File | Role |
|---|---|
| `data.py` | Loads a track into `Task`s (one per exercise: words, lemmas, morph features, SLAM split) + 0/1 outcomes |
| `profile.py` | `LearnerProfile`: seen / wrong / last-seen counts for overall, format, word, lemma, morph feature. `CountUpdater`: the update rule (with optional per-format evidence weights) |
| `predictors.py` | `Population` (average-learner stats from the train split) and the baselines. `replay()`: predict-then-update over each learner's history |
| `metrics.py` | skill (log-loss vs base rate), AUROC, within-exercise AUROC, F1, calibration |
| `run.py` | Runs every predictor in `BASELINES` and writes the report |

To try something new, write a `Predictor` subclass (`fit` on the train split, `predict(profile, task)`) and add it
to `BASELINES`, or swap `CountUpdater` for another update rule.

## Setup choices
- SLAM's own split: each learner's exercises are train, then dev, then test (last ~10%) in time order. Every
  learner is replayed from their first exercise (predict, then update the profile); only test exercises are
  scored. Population stats and `profile_lr` weights use the train split only; dev is unused for now.
  Only reverse_translate exercises are loaded, so the profile only sees that task type.
- Lemmas come from `simplemma` (lookup table, no context). Morph features are SLAM's tags prefixed by POS
  (`NOUN:Number=Plur`); they are a little noisy (`Somos` is tagged `Tense=Past`).
- `Population` word difficulty leaves out the asking learner's own answers; a learner's own history reaches
  predictions only through their profile.

## Findings so far (reverse_translate, SLAM split)
- **Skill** (mean over tracks): random -1.04, past accuracy +0.03, word difficulty +0.10,
  ability x difficulty +0.13, profile_lr +0.15.
- profile_lr AUROC 0.768 / 0.750 / 0.770 (en_es / es_en / fr_en): level with SLAM's official logistic-regression
  baseline (0.774 / 0.746 / 0.771, all formats); best published ~0.864 / 0.839 / 0.860.
- **Most of the signal is "who" plus "what"**: overall learner strength and how hard the word is for everyone.
  The two together, with nothing fitted, get ~85% of the way to profile_lr.
- **Personal history adds a little**: profile_lr ranks words within a sentence better than word difficulty
  alone (within-exercise AUROC 0.75 vs 0.73).
- Per-learner AUROC is a misleading metric: within a fixed history, a learner's past error rate is mechanically
  anti-correlated with their remaining answers. Within-exercise AUROC is used instead.
