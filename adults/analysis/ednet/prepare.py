"""Turn EdNet-KT1 (784k per-student CSVs inside one zip) plus the Contents tables
into Parquet and summary tables for the explorer.

KT1 row = one question answered by one student:
  timestamp     ms (shifted by a fixed amount for privacy, so real dates are unknown)
  solving_id    bundle attempt counter for this student (1, 2, 3, ...)
  question_id   q{n}
  user_answer   a-d
  elapsed_time  ms spent

Outputs (adults/data/ednet/parquet/):
  kt1.parquet          all answers, sorted by student and time, with correctness and question info
  questions.parquet    per question: content info, accuracy, answer-choice counts, timing
  users.parquet        per student summary + ability (share correct beyond question difficulty)
  tags.parquet         per skill tag
  tag_pairs.parquet    links between tags (residual correlation across students)
  curve.parquet        accuracy by how many questions the student had answered so far
  timing.parquet       accuracy by time spent, per part
"""

import io
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[2] / "data" / "ednet"
ZIP = DATA / "EdNet-KT1.zip"
CONTENTS = DATA / "contents"
OUT = DATA / "parquet"
RAW = OUT / "kt1_raw.parquet"

MIN_ANSWERS_FOR_LINKS = 200  # students need this many answers to enter the tag-link matrix
MIN_TAG_ANSWERS = 5          # ...and this many answers on a tag for it to count for them
MIN_SHARED = 300


def zip_to_parquet():
    schema = pa.schema([("user", pa.int32()), ("timestamp", pa.int64()), ("solving_id", pa.int32()),
                        ("question_id", pa.string()), ("user_answer", pa.string()),
                        ("elapsed_time", pa.int64())])
    opts = pacsv.ReadOptions(column_names=schema.names)
    conv = pacsv.ConvertOptions(column_types=schema)
    writer = pq.ParquetWriter(RAW, schema)
    buf, n_files = [], 0
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and "/u" in "/" + n]
        print(f"{len(names):,} student files")
        for i, name in enumerate(names):
            uid = name.rsplit("/", 1)[-1][1:-4].encode()
            lines = z.read(name).rstrip(b"\n").split(b"\n")[1:]  # drop header
            if lines:
                buf.append(uid + b"," + (b"\n" + uid + b",").join(lines))
            if len(buf) >= 20000 or i == len(names) - 1:
                tbl = pacsv.read_csv(io.BytesIO(b"\n".join(buf) + b"\n"), read_options=opts,
                                     convert_options=conv)
                writer.write_table(tbl)
                n_files += len(buf)
                buf = []
                print(f"  {i + 1:,} files", end="\r")
    writer.close()
    print()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        zip_to_parquet()
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    con.execute(f"""
        CREATE TABLE q AS
        SELECT CAST(substr(question_id, 2) AS INTEGER) AS qid, question_id, bundle_id, explanation_id,
               correct_answer, part, tags, to_timestamp(deployed_at / 1000) AS deployed_at
        FROM read_csv('{CONTENTS / "questions.csv"}', header=true,
                      columns={{'question_id':'VARCHAR','bundle_id':'VARCHAR','explanation_id':'VARCHAR',
                                'correct_answer':'VARCHAR','part':'INTEGER','tags':'VARCHAR',
                                'deployed_at':'BIGINT'}})""")

    print("joining answers with questions ...")
    con.execute(f"""
        COPY (
            SELECT r.user, r.timestamp, to_timestamp(r.timestamp / 1000) AS time, r.solving_id,
                   CAST(substr(r.question_id, 2) AS INTEGER) AS qid, q.bundle_id, q.part,
                   r.user_answer, q.correct_answer,
                   (r.user_answer = q.correct_answer)::TINYINT AS correct,
                   r.elapsed_time / 1000.0 AS elapsed_s,
                   row_number() OVER (PARTITION BY r.user ORDER BY r.timestamp, r.question_id) AS n_th
            FROM '{RAW}' r LEFT JOIN q ON q.question_id = r.question_id
            ORDER BY r.user, r.timestamp
        ) TO '{OUT / "kt1.parquet"}' (FORMAT parquet, ROW_GROUP_SIZE 200000)""")
    con.execute(f"CREATE VIEW a AS SELECT * FROM '{OUT / 'kt1.parquet'}'")
    n, users = con.execute("SELECT count(*), count(DISTINCT user) FROM a").fetchone()
    print(f"{n:,} answers from {users:,} students")

    print("question summaries ...")
    con.execute("""
        CREATE TABLE qs AS
        SELECT qid, count(*) AS answers, count(DISTINCT user) AS students, avg(correct) AS accuracy,
               count_if(user_answer = 'a') AS n_a, count_if(user_answer = 'b') AS n_b,
               count_if(user_answer = 'c') AS n_c, count_if(user_answer = 'd') AS n_d,
               count_if(user_answer IS NULL OR user_answer NOT IN ('a','b','c','d')) AS n_other,
               median(elapsed_s) FILTER (WHERE correct = 1) AS median_s_right,
               median(elapsed_s) FILTER (WHERE correct = 0) AS median_s_wrong
        FROM a GROUP BY qid""")
    con.execute(f"""COPY (SELECT q.*, qs.* EXCLUDE (qid) FROM q LEFT JOIN qs USING (qid))
                    TO '{OUT / "questions.parquet"}' (FORMAT parquet)""")

    print("student summaries ...")
    # residual beyond question difficulty; a student's mean residual = ability
    con.execute("""
        CREATE TABLE ua AS
        SELECT a.user, avg(a.correct - qs.accuracy) AS ability
        FROM a JOIN qs USING (qid) GROUP BY a.user""")
    con.execute(f"""COPY (
        SELECT user, count(*) AS answers, count(DISTINCT solving_id) AS bundles, avg(correct) AS accuracy,
               min(time) AS first_time, max(time) AS last_time,
               count(DISTINCT CAST(time AS DATE)) AS active_days,
               median(elapsed_s) AS median_s, any_value(ua.ability) AS ability
        FROM a JOIN ua USING (user) GROUP BY user) TO '{OUT / "users.parquet"}' (FORMAT parquet)""")

    print("learning curve and timing ...")
    con.execute(f"""COPY (
        WITH b AS (
            SELECT a.part, a.correct, a.correct - qs.accuracy AS resid,
                   CASE WHEN n_th <= 10 THEN 1 WHEN n_th <= 30 THEN 2 WHEN n_th <= 100 THEN 3
                        WHEN n_th <= 300 THEN 4 WHEN n_th <= 1000 THEN 5 WHEN n_th <= 3000 THEN 6
                        ELSE 7 END AS stage,
                   u.answers >= 1000 AS stayer
            FROM a JOIN qs USING (qid) JOIN '{OUT / "users.parquet"}' u USING (user))
        SELECT stage, part, stayer, count(*) AS n, avg(correct) AS accuracy, avg(resid) AS resid
        FROM b GROUP BY ALL) TO '{OUT / "curve.parquet"}' (FORMAT parquet)""")
    con.execute(f"""COPY (
        SELECT part, least(floor(elapsed_s / 5) * 5, 120) AS secs, count(*) AS n, avg(correct) AS accuracy
        FROM a WHERE elapsed_s >= 0 GROUP BY ALL) TO '{OUT / "timing.parquet"}' (FORMAT parquet)""")

    print("tags ...")
    con.execute("""
        CREATE TABLE qt AS
        SELECT qid, part, CAST(unnest(string_split(tags, ';')) AS INTEGER) AS tag FROM q
        WHERE tags IS NOT NULL AND tags <> '-1'""")
    con.execute(f"""COPY (
        SELECT qt.tag, count(DISTINCT qt.qid) AS questions,
               string_agg(DISTINCT CAST(qt.part AS VARCHAR), ',' ORDER BY CAST(qt.part AS VARCHAR)) AS parts,
               sum(qs.answers) AS answers, sum(qs.accuracy * qs.answers) / sum(qs.answers) AS accuracy
        FROM qt JOIN qs USING (qid) GROUP BY qt.tag) TO '{OUT / "tags.parquet"}' (FORMAT parquet)""")

    print("tag links ...")
    # residual = right/wrong minus question accuracy minus the student's ability,
    # averaged per student x tag, for students with enough answers
    cells = con.execute(f"""
        WITH r AS (
            SELECT a.user, a.qid, a.correct - qs.accuracy - ua.ability AS resid
            FROM a JOIN qs USING (qid) JOIN ua USING (user)
            WHERE a.user IN (SELECT user FROM '{OUT / "users.parquet"}'
                             WHERE answers >= {MIN_ANSWERS_FOR_LINKS}))
        SELECT r.user, qt.tag, avg(r.resid) AS resid, count(*) AS n
        FROM r JOIN qt USING (qid) GROUP BY ALL HAVING count(*) >= {MIN_TAG_ANSWERS}""").df()
    tags = np.sort(cells.tag.unique())
    tcol = pd.Series(np.arange(len(tags)), index=tags)
    ucode = cells.user.astype("category").cat.codes.to_numpy()
    X = np.zeros((ucode.max() + 1, len(tags)), dtype=np.float32)
    M = np.zeros_like(X)
    X[ucode, tcol[cells.tag].to_numpy()] = cells.resid
    M[ucode, tcol[cells.tag].to_numpy()] = 1
    print(f"  {X.shape[0]:,} students x {X.shape[1]} tags")
    X64, M64 = X.astype(np.float64), M.astype(np.float64)
    nn = M64.T @ M64
    sx = X64.T @ M64
    sxx = (X64 * X64).T @ M64
    sxy = X64.T @ X64
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / nn, sx.T / nn
        r = (sxy / nn - mx * my) / np.sqrt((sxx / nn - mx ** 2) * (sxx.T / nn - my ** 2))
    # confound: how often the two tags sit on the same question
    qt = con.execute("SELECT qid, tag FROM qt").df()
    qt = qt[qt.tag.isin(tags)]
    B = np.zeros((qt.qid.max() + 1, len(tags)), dtype=np.float32)
    B[qt.qid.to_numpy(), tcol[qt.tag].to_numpy()] = 1
    co = B.T @ B
    same_q = co / np.minimum.outer(np.diag(co), np.diag(co))
    ia, ib = np.triu_indices(len(tags), k=1)
    pairs = pd.DataFrame({"tag_a": tags[ia], "tag_b": tags[ib], "n_shared": nn[ia, ib].astype(int),
                          "link": r[ia, ib], "same_question_share": same_q[ia, ib]})
    pairs = pairs[pairs.n_shared >= MIN_SHARED].dropna(subset=["link"])
    pairs["z"] = pairs.link * np.sqrt(pairs.n_shared - 3)
    pairs.to_parquet(OUT / "tag_pairs.parquet", index=False)
    print(f"  {len(pairs):,} tag pairs")
    print("done")


if __name__ == "__main__":
    main()
