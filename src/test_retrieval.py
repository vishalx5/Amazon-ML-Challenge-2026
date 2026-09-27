import sqlite3
import pandas as pd
import re


SOURCE1_FILE = "student_resource/dataset/train/train_source1.tsv"
GROUND_TRUTH_FILE = "student_resource/dataset/train/train_ground_truth.tsv"

SOURCE2_INDEX = "student_resource/source2_index.db"
SOURCE3_INDEX = "student_resource/source3_index.db"


def get_tokens(text):

    if pd.isna(text):
        return []

    tokens = re.findall(
        r"[A-Za-z0-9\u0900-\u097F]+",
        str(text).lower()
    )

    return [
        token
        for token in tokens
        if len(token) >= 3
    ]


def make_or_query(text):

    tokens = get_tokens(text)

    if not tokens:
        return ""

    return " OR ".join(
        f'"{token}"'
        for token in tokens
    )


def make_and_query(text):

    tokens = get_tokens(text)

    if not tokens:
        return ""

    # Remove duplicate tokens
    tokens = list(dict.fromkeys(tokens))

    return " AND ".join(
        f'"{token}"'
        for token in tokens
    )


def search_index(conn, query, limit=500):

    if not query:
        return set()

    try:

        rows = conn.execute(
            """
            SELECT entity_id
            FROM businesses
            WHERE businesses MATCH ?
            ORDER BY bm25(businesses)
            LIMIT ?
            """,
            (query, limit)
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    except sqlite3.OperationalError:

        return set()


def get_record(conn, entity_id):

    return conn.execute(
        """
        SELECT
            entity_id,
            business_name_normalized,
            business_address_normalized,
            country
        FROM businesses
        WHERE entity_id = ?
        """,
        (entity_id,)
    ).fetchone()


print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    nrows=100,
    dtype=str
)


source1_ids = ground_truth[
    "source1_entity_id"
].tolist()


source1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
)


source1 = source1[
    source1["entity_id"].isin(source1_ids)
].copy()


conn2 = sqlite3.connect(SOURCE2_INDEX)
conn3 = sqlite3.connect(SOURCE3_INDEX)


total_matches = 0
found_old = 0
found_strong = 0
missed_old = 0
missed_strong = 0


print()
print("=" * 80)
print("TESTING STRONG ADDRESS RETRIEVAL")
print("=" * 80)


for _, s1 in source1.iterrows():

    source1_id = s1["entity_id"]

    name = s1["business_name"]
    address = s1["business_address"]


    # -------------------------------------------------
    # OLD METHOD
    # -------------------------------------------------

    name_query = make_or_query(name)
    address_query = make_or_query(address)


    old_candidates = (
        search_index(conn2, name_query, 500)
        |
        search_index(conn3, name_query, 500)
        |
        search_index(conn2, address_query, 500)
        |
        search_index(conn3, address_query, 500)
    )


    # -------------------------------------------------
    # NEW STRONG ADDRESS METHOD
    # -------------------------------------------------

    address_tokens = get_tokens(address)

    strong_address_candidates = set()


    # Use combinations of the first useful address tokens.
    # We test several small AND queries rather than
    # requiring the entire address to match.

    if len(address_tokens) >= 2:

        unique_tokens = list(
            dict.fromkeys(address_tokens)
        )

        # Test pairs of address tokens.
        pairs = []

        for i in range(len(unique_tokens)):

            for j in range(i + 1, len(unique_tokens)):

                pairs.append(
                    (
                        unique_tokens[i],
                        unique_tokens[j]
                    )
                )


        # Limit the number of queries so this test stays small.
        pairs = pairs[:20]


        for token1, token2 in pairs:

            query = (
                f'"{token1}" AND "{token2}"'
            )


            strong_address_candidates |= search_index(
                conn2,
                query,
                500
            )

            strong_address_candidates |= search_index(
                conn3,
                query,
                500
            )


    new_candidates = (
        old_candidates
        |
        strong_address_candidates
    )


    # -------------------------------------------------
    # Ground truth
    # -------------------------------------------------

    gt_row = ground_truth[
        ground_truth["source1_entity_id"]
        == source1_id
    ]


    if gt_row.empty:
        continue


    matched_value = gt_row.iloc[0][
        "matched_entity_ids"
    ]


    if (
        pd.isna(matched_value)
        or not str(matched_value).strip()
    ):
        continue


    true_matches = {
        x.strip()
        for x in str(matched_value).split(",")
        if x.strip()
    }


    total_matches += len(true_matches)


    old_found = true_matches & old_candidates
    new_found = true_matches & new_candidates


    found_old += len(old_found)
    found_strong += len(new_found)


    old_missed = true_matches - old_candidates
    new_missed = true_matches - new_candidates


    if old_missed:

        missed_old += len(old_missed)


        if new_missed:

            missed_strong += len(new_missed)


            print()
            print("-" * 80)

            print("SOURCE 1:", source1_id)
            print("NAME    :", name)
            print("ADDRESS :", address)

            print()
            print("OLD MISSED:")
            print(old_missed)

            print()
            print("STILL MISSED AFTER STRONG ADDRESS:")
            print(new_missed)


        else:

            print()
            print("-" * 80)

            print("RECOVERED!")

            print("SOURCE 1:", source1_id)
            print("NAME    :", name)
            print("ADDRESS :", address)

            print()
            print("RECOVERED MATCHES:")
            print(old_missed)


# -----------------------------------------------------
# FINAL RESULT
# -----------------------------------------------------

conn2.close()
conn3.close()


print()
print("=" * 80)
print("STRONG RETRIEVAL RESULT")
print("=" * 80)

print(
    "Total true matches :",
    total_matches
)

print(
    "Old method found   :",
    found_old
)

print(
    "Strong method found:",
    found_strong
)


if total_matches > 0:

    old_recall = (
        found_old / total_matches
    ) * 100

    new_recall = (
        found_strong / total_matches
    ) * 100

    print(
        "Old recall         :",
        round(old_recall, 2),
        "%"
    )

    print(
        "New recall         :",
        round(new_recall, 2),
        "%"
    )


print(
    "Old missed matches :",
    missed_old
)

print(
    "Still missed       :",
    missed_strong
)

print("=" * 80)