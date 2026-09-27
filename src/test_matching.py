import sqlite3
import pandas as pd
import re

from matching import calculate_match_score, is_match


SOURCE1_FILE = "student_resource/dataset/train/train_source1.tsv"
GROUND_TRUTH_FILE = "student_resource/dataset/train/train_ground_truth.tsv"

SOURCE2_INDEX = "student_resource/source2_index.db"
SOURCE3_INDEX = "student_resource/source3_index.db"

SAMPLE_SIZE = 100
LIMIT = 500


def tokens(text):

    if pd.isna(text):
        return []

    return [
        x for x in re.findall(
            r"[A-Za-z0-9\u0900-\u097F]+",
            str(text).lower()
        )
        if len(x) >= 3
    ]


def query(text):

    ts = list(dict.fromkeys(tokens(text)))

    if not ts:
        return ""

    return " OR ".join(
        f'"{x}"'
        for x in ts
    )


def search(conn, q):

    if not q:
        return []

    try:

        return conn.execute(
            """
            SELECT
                entity_id,
                business_name_normalized,
                business_address_normalized,
                country
            FROM businesses
            WHERE businesses MATCH ?
            ORDER BY bm25(businesses)
            LIMIT ?
            """,
            (q, LIMIT)
        ).fetchall()

    except Exception:

        return []


print("Loading training sample...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_SIZE
)

ids = gt["source1_entity_id"].tolist()

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
)

s1 = s1[
    s1["entity_id"].isin(ids)
].copy()


conn2 = sqlite3.connect(SOURCE2_INDEX)
conn3 = sqlite3.connect(SOURCE3_INDEX)


TP = 0
FP = 0
FN = 0

total_true = 0
total_predicted = 0


print()
print("=" * 70)
print("MATCHING EVALUATION")
print("=" * 70)


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    source1 = {
        "entity_id": s1_id,
        "business_name": row["business_name"],
        "business_address": row["business_address"],
        "country": row["country"]
    }


    # -----------------------------------------
    # Retrieve candidates
    # -----------------------------------------

    name_q = query(
        row["business_name"]
    )

    address_q = query(
        row["business_address"]
    )


    rows2 = (
        search(conn2, name_q)
        +
        search(conn2, address_q)
    )

    rows3 = (
        search(conn3, name_q)
        +
        search(conn3, address_q)
    )


    candidates = {}

    for r in rows2 + rows3:

        candidates[r[0]] = {
            "entity_id": r[0],
            "business_name": r[1],
            "business_address": r[2],
            "country": r[3]
        }


    # -----------------------------------------
    # Score candidates
    # -----------------------------------------

    predicted = set()

    for candidate_id, candidate in candidates.items():

        scores = calculate_match_score(
            source1,
            candidate
        )

        if is_match(scores):

            predicted.add(
                candidate_id
            )


    # -----------------------------------------
    # Ground truth
    # -----------------------------------------

    gt_row = gt[
        gt["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue


    value = gt_row.iloc[0][
        "matched_entity_ids"
    ]


    if (
        pd.isna(value)
        or not str(value).strip()
    ):

        true_matches = set()

    else:

        true_matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }


    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    tp = len(
        predicted & true_matches
    )

    fp = len(
        predicted - true_matches
    )

    fn = len(
        true_matches - predicted
    )


    TP += tp
    FP += fp
    FN += fn

    total_true += len(
        true_matches
    )

    total_predicted += len(
        predicted
    )


    print(
        f"{s1_id} | "
        f"TRUE={len(true_matches)} | "
        f"PRED={len(predicted)} | "
        f"TP={tp} | "
        f"FP={fp} | "
        f"FN={fn}"
    )


conn2.close()
conn3.close()


# =========================================================
# FINAL METRICS
# =========================================================

precision = (
    TP / (TP + FP)
    if TP + FP > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN > 0
    else 0
)


beta = 0.5

f05 = (
    (1 + beta ** 2)
    * precision
    * recall
    /
    (
        beta ** 2 * precision
        + recall
    )
    if precision + recall > 0
    else 0
)


print()
print("=" * 70)
print("MATCHING RESULT")
print("=" * 70)

print("True matches      :", total_true)
print("Predicted matches :", total_predicted)
print("TP                :", TP)
print("FP                :", FP)
print("FN                :", FN)

print()

print(
    "Precision         :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall            :",
    round(recall * 100, 2),
    "%"
)

print(
    "F0.5              :",
    round(f05 * 100, 2),
    "%"
)

print("=" * 70)