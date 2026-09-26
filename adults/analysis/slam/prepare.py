"""Parse the raw Duolingo SLAM 2018 files into Parquet tables.

For each track (en_es, es_en, fr_en) the train/dev/test files are merged back
into one continuous 30-day history per learner, with dev/test labels joined in
from the .key files.

Outputs (in adults/data/slam/parquet/):
  {track}_exercises.parquet  one row per exercise
  {track}_tokens.parquet     one row per token (word) in an exercise
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "slam" / "dataverse_files"
OUT = ROOT / "data" / "slam" / "parquet"
TRACKS = ["en_es", "es_en", "fr_en"]
SPLITS = ["train", "dev", "test"]
VERSION = "slam.20190204"


def read_keys(path: Path) -> dict[str, int]:
    keys = {}
    with open(path) as f:
        for line in f:
            tid, label = line.split()
            keys[tid] = int(label)
    return keys


def parse(path: Path, split: str, keys: dict[str, int] | None):
    exercises, tokens = [], []
    prompt, meta, ex_id, idx = None, None, None, 0
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                prompt, meta, ex_id = None, None, None
                continue
            if line.startswith("# prompt:"):
                prompt = line[len("# prompt:"):]
                continue
            if line.startswith("# user:"):
                parts = line[2:].split()
                meta = dict(p.split(":", 1) for p in parts)
                continue
            f_ = line.split()
            tid = f_[0]
            if ex_id is None:
                # exercise id = token id minus its last 2 digits (token position)
                ex_id = tid[:-2]
                idx = 0
                exercises.append({
                    "ex_id": ex_id,
                    "user": meta["user"],
                    "countries": meta["countries"],
                    "days": float(meta["days"]),
                    "client": meta["client"],
                    "session": meta["session"],
                    "format": meta["format"],
                    "time": None if meta["time"] == "null" else int(meta["time"]),
                    "prompt": prompt,
                    "split": split,
                })
            label = int(f_[6]) if len(f_) > 6 else keys[tid]
            tokens.append({
                "token_id": tid,
                "ex_id": ex_id,
                "pos_in_ex": idx,
                "token": f_[1],
                "word": f_[1].lower(),
                "pos": f_[2],
                "morph": f_[3],
                "dep": f_[4],
                "head": int(f_[5]),
                "wrong": label,  # 1 = the learner got this word wrong
            })
            idx += 1
    return exercises, tokens


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        all_ex, all_tok = [], []
        for split in SPLITS:
            path = RAW / f"{track}.{VERSION}.{split}"
            keys = read_keys(RAW / f"{track}.{VERSION}.{split}.key") if split != "train" else None
            ex, tok = parse(path, split, keys)
            all_ex += ex
            all_tok += tok
            print(f"{track} {split}: {len(ex):,} exercises, {len(tok):,} tokens")
        ex = pd.DataFrame(all_ex)
        tok = pd.DataFrame(all_tok)
        # chronological order per learner, then a per-learner exercise counter
        ex = ex.sort_values(["user", "days"], kind="stable").reset_index(drop=True)
        ex["ex_num"] = ex.groupby("user").cumcount()
        for c in ["user", "countries", "client", "session", "format", "split"]:
            ex[c] = ex[c].astype("category")
        for c in ["pos", "dep"]:
            tok[c] = tok[c].astype("category")
        tok["wrong"] = tok["wrong"].astype("int8")
        ex.to_parquet(OUT / f"{track}_exercises.parquet", index=False)
        tok.to_parquet(OUT / f"{track}_tokens.parquet", index=False)


if __name__ == "__main__":
    main()
