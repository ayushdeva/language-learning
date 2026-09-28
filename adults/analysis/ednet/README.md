# EdNet (KT1) explorer

Data: Choi et al. (2019), "EdNet: A Large-Scale Hierarchical Dataset in Education", arXiv:1912.03072.
[github.com/riiid/ednet](https://github.com/riiid/ednet), CC BY-NC 4.0. Every multiple-choice answer by 784k
students on Santa, a Korean TOEIC-prep app. We use KT1 (question answers) + Contents (question metadata);
KT2-KT4 (full action logs, lectures, explanations, media, payments) are described in the app's Overview tab.

Download into `adults/data/ednet/`: `EdNet-KT1.zip` (bit.ly/ednet_kt1, 1.1 GB, keep zipped) and the
Contents zip (bit.ly/ednet-content), unzipped to `contents/`.

```bash
# from adults/analysis
uv run python ednet/prepare.py                        # zip -> parquet + summaries (~2 min, ~3 GB on disk)
uv run streamlit run explorer.py   # the combined explorer
```

## Findings so far
- 95.3M answers, 65% correct overall. Part 5 (incomplete sentences) is hardest (60%), part 1 easiest (74%).
- Median student answers only 11 questions; ~27% of students start with the identical question sequence,
  apparently a placement test, with ~49% accuracy on the first 10 answers.
- Accuracy rises with practice even among students who stayed for 1k+ answers (53% -> 72%), and still rises
  after subtracting question difficulty (-3 -> +4 points), so it's not only Santa serving easier questions.
- Skill tags link up after removing ability and difficulty (e.g. 182/185, 24/34, 52/53/54) even when they
  never share a question; tags have no names, so interpretation needs the questions themselves.
