# speechocean762 pronunciation explorer

Data: Zhang et al. (2021), speechocean762, OpenSLR 101, CC BY 4.0. 5,000 English sentences read by
250 Mandarin L1 speakers (104 aged 6-12, 18 aged 13-15, 128 aged 19-43), each scored by 5 experts at
sentence, word and phone level. Raw files in `adults/data/speechocean762/`.

```bash
# from adults/analysis
uv run python speechocean762/prepare.py                        # json + kaldi files -> parquet (~5s)
uv run streamlit run explorer.py   # the combined explorer
```

Note: `completeness` is on a 0-10 scale in the data (README says 0-1).

## Findings so far
- Children and adults read almost entirely different sentences (1 shared of ~4,500), and children's
  are shorter (5.3 vs 7.1 words), so raw group comparisons are unfair. The app compares groups only on
  (word, sound) slots both groups read.
- Even matched, children score higher on almost every sound (1.93 vs 1.84 out of 2). Could be a real
  age effect or experts grading children more leniently; the data can't separate the two.
- Hardest sounds: TH (thin), Z, EY, EH, ER - several absent from Mandarin.
- Sounds at the start of a word score best (1.92 of 2) vs middle (1.82) and end (1.85). The most common inserted sound is a
  schwa ("uh") at the end of a word: the expected pattern for Mandarin speakers.
- Experts disagree by a median 2.3 points (of 10) on sentence totals.
