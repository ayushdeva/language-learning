"""Flatten speechocean762 (Zhang et al. 2021) into Parquet tables.

5,000 English sentences read aloud by 250 Mandarin L1 speakers (children and
adults), scored by 5 experts at sentence, word and phone level.

Phone scores: 2 = correct, 1 = right but heavily accented, 0 = wrong or missing.
scores-detail.json keeps each expert's markings; we count how many of the 5
experts gave each score and collect phones they heard *inserted*.

Outputs (adults/data/speechocean762/parquet/):
  speakers.parquet    one row per speaker (age, gender, group, averages)
  utterances.parquet  one row per recording with sentence-level scores
  words.parquet       one row per word
  phones.parquet      one row per expected phone
  inserted.parquet    one row per phone an expert heard inserted
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "speechocean762"
RAW = DATA / "speechocean762"
OUT = DATA / "parquet"
TOKEN = re.compile(r"\[([^\]]+)\]|\(([^)]+)\)|\{([^}]+)\}|(\S+)")


def kaldi_map(path: Path) -> dict[str, str]:
    return dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())


def base_phone(p: str) -> str:
    return re.sub(r"\d", "", p)


def parse_expert(s: str):
    """Return (scores for expected phones, [(position, inserted phone)])."""
    scores, inserted = [], []
    for ins, zero, one, two in TOKEN.findall(s):
        if ins:
            inserted.append((len(scores), ins))
        elif zero:
            scores.append(0)
        elif one:
            scores.append(1)
        else:
            scores.append(2)
    return scores, inserted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spk_rows, utt_split, utt_spk, utt_wav = [], {}, {}, {}
    for split in ["train", "test"]:
        d = RAW / split
        ages, genders = kaldi_map(d / "spk2age"), kaldi_map(d / "spk2gender")
        for spk, age in ages.items():
            spk_rows.append({"speaker": spk, "age": int(age), "gender": genders[spk], "split": split})
        for utt, spk in kaldi_map(d / "utt2spk").items():
            utt_split[utt], utt_spk[utt] = split, spk
        utt_wav.update(kaldi_map(d / "wav.scp"))

    scores = json.loads((RAW / "resource" / "scores.json").read_text())
    detail = json.loads((RAW / "resource" / "scores-detail.json").read_text())

    utts, words, phones, inserted = [], [], [], []
    for utt, s in scores.items():
        det = detail[utt]
        utts.append({
            "utt": utt, "speaker": utt_spk[utt], "split": utt_split[utt], "text": s["text"],
            "wav": str(RAW / utt_wav[utt]),
            **{k: s[k] for k in ["accuracy", "completeness", "fluency", "prosodic", "total"]},
            "expert_total_min": min(det["total"]), "expert_total_max": max(det["total"]),
        })
        n_words = len(s["words"])
        for wi, (w, wd) in enumerate(zip(s["words"], det["words"])):
            ref = w["phones"]
            words.append({
                "utt": utt, "word_idx": wi, "word": w["text"], "n_words": n_words,
                "accuracy": w["accuracy"], "stress": w["stress"], "total": w["total"],
                "ref_phones": " ".join(ref), "n_phones": len(ref),
            })
            # per-expert markings; skip experts whose phone count doesn't line up
            expert = [parse_expert(p) for p in wd["phones"] if isinstance(p, str)]
            aligned = [e for e in expert if len(e[0]) == len(ref)]
            for pi, p in enumerate(ref):
                votes = [e[0][pi] for e in aligned]
                phones.append({
                    "utt": utt, "word_idx": wi, "phone_idx": pi, "word": w["text"],
                    "phone": p, "base": base_phone(p),
                    "position": "only" if len(ref) == 1 else
                                "first" if pi == 0 else "last" if pi == len(ref) - 1 else "middle",
                    "score": w["phones-accuracy"][pi],
                    "n_experts": len(votes),
                    "n_wrong": votes.count(0), "n_accent": votes.count(1), "n_ok": votes.count(2),
                })
            for ei, (_, ins) in enumerate(expert):
                for pos, ph in ins:
                    prev = ref[pos - 1] if 0 < pos <= len(ref) else None
                    inserted.append({
                        "utt": utt, "word_idx": wi, "word": w["text"], "expert": ei,
                        "inserted": ph, "base": base_phone(ph), "after_idx": pos,
                        "after_phone": prev,
                        "where": "start" if pos == 0 else "end" if pos == len(ref) else "middle",
                    })

    utt_df = pd.DataFrame(utts)
    spk = pd.DataFrame(spk_rows)
    spk["group"] = pd.cut(spk.age, [0, 12, 18, 100], labels=["child (6-12)", "teen (13-15)", "adult (19+)"])
    spk = spk.merge(
        utt_df.groupby("speaker")[["accuracy", "fluency", "prosodic", "completeness", "total"]].mean()
        .add_prefix("mean_").reset_index().merge(
            utt_df.groupby("speaker").size().rename("n_utts").reset_index()), on="speaker")
    spk.to_parquet(OUT / "speakers.parquet", index=False)
    utt_df.to_parquet(OUT / "utterances.parquet", index=False)
    pd.DataFrame(words).to_parquet(OUT / "words.parquet", index=False)
    pd.DataFrame(phones).to_parquet(OUT / "phones.parquet", index=False)
    pd.DataFrame(inserted).to_parquet(OUT / "inserted.parquet", index=False)
    print(f"{len(spk)} speakers, {len(utt_df)} utterances, {len(words)} words, "
          f"{len(phones)} phones, {len(inserted)} inserted-phone marks")


if __name__ == "__main__":
    main()
