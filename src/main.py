import os
import sys
import sqlite3
import re
import pandas as pd

from matching import calculate_match_score, is_match


# =========================================================
# FILE PATHS
# =========================================================

SOURCE1_FILE = "student_resource/dataset/test/test_source1.tsv"

SOURCE2_INDEX = "student_resource/source2_index.db"
SOURCE3_INDEX = "student_resource/source3_index.db"

OUTPUT_DIR = "output"

MATCHING_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

CANDIDATE_PAIRS_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


# =========================================================
# SETTINGS
# =========================================================

# IMPORTANT:
# Keep this reasonably small because your laptop has limited RAM.
CHUNK_SIZE = 1000

NAME_LIMIT = 500
ADDRESS_LIMIT = 500

# Run:
# python src/main.py --smoke
#
# to test only 100 records.
SMOKE_TEST = "--smoke" in sys.argv

SMOKE_ROWS = 100


# =========================================================
# TEXT TOKENIZATION
# =========================================================

def get_tokens(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    tokens = re.findall(
        r"[A-Za-z0-9\u0900-\u097F]+",
        text
    )

    return [
        token
        for token in tokens
        if len(token) >= 3
    ]


# =========================================================
# CREATE SQLITE FTS QUERY
# =========================================================

def make_fts_query(text):

    tokens = get_tokens(text)

    if not tokens:
        return ""

    # Remove duplicate tokens
    tokens = list(dict.fromkeys(tokens))

    return " OR ".join(
        f'"{token}"'
        for token in tokens
    )


# =========================================================
# SEARCH SQLITE INDEX
# =========================================================

def search_index(
    conn,
    query,
    limit
):

    if not query:
        return []

    try:

        rows = conn.execute(
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
            (
                query,
                limit
            )
        ).fetchall()

        return rows

    except sqlite3.OperationalError:

        return []


# =========================================================
# GET CANDIDATES FROM ONE INDEX
# =========================================================

def get_candidates_from_index(
    conn,
    name_query,
    address_query
):

    candidates = {}

    # -----------------------------------------------------
    # Name candidates
    # -----------------------------------------------------

    name_rows = search_index(
        conn,
        name_query,
        NAME_LIMIT
    )

    for row in name_rows:

        entity_id = row[0]

        candidates[entity_id] = {
            "entity_id": row[0],
            "business_name": row[1],
            "business_address": row[2],
            "country": row[3]
        }


    # -----------------------------------------------------
    # Address candidates
    # -----------------------------------------------------

    address_rows = search_index(
        conn,
        address_query,
        ADDRESS_LIMIT
    )

    for row in address_rows:

        entity_id = row[0]

        if entity_id not in candidates:

            candidates[entity_id] = {
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3]
            }


    return candidates


# =========================================================
# PROCESS ONE SOURCE 1 RECORD
# =========================================================

def process_source1_record(
    source1_row,
    conn2,
    conn3
):

    source1_id = str(
        source1_row["entity_id"]
    )

    source1_record = {
        "entity_id": source1_id,
        "business_name": source1_row["business_name"],
        "business_address": source1_row["business_address"],
        "country": source1_row["country"]
    }


    # -----------------------------------------------------
    # Build search queries
    # -----------------------------------------------------

    name_query = make_fts_query(
        source1_row["business_name"]
    )

    address_query = make_fts_query(
        source1_row["business_address"]
    )


    # -----------------------------------------------------
    # Source 2 candidates
    # -----------------------------------------------------

    candidates2 = get_candidates_from_index(
        conn2,
        name_query,
        address_query
    )


    # -----------------------------------------------------
    # Source 3 candidates
    # -----------------------------------------------------

    candidates3 = get_candidates_from_index(
        conn3,
        name_query,
        address_query
    )


    # -----------------------------------------------------
    # Combine candidates
    # -----------------------------------------------------

    candidates = {}

    candidates.update(candidates2)
    candidates.update(candidates3)


    # -----------------------------------------------------
    # Safety: remove Source 1 ID if somehow present
    # -----------------------------------------------------

    candidates.pop(
        source1_id,
        None
    )


    # -----------------------------------------------------
    # Score every candidate
    # -----------------------------------------------------

    scored_candidates = []

    for candidate_id, candidate in candidates.items():

        scores = calculate_match_score(
            source1_record,
            candidate
        )

        scored_candidates.append(
            (
                candidate_id,
                candidate,
                scores
            )
        )


    # -----------------------------------------------------
    # Highest score first
    # -----------------------------------------------------

    scored_candidates.sort(
        key=lambda x: x[2]["final_score"],
        reverse=True
    )


    # -----------------------------------------------------
    # Decide matches
    # -----------------------------------------------------

    matched_ids = []

    for candidate_id, candidate, scores in scored_candidates:

        if is_match(scores):

            matched_ids.append(
                candidate_id
            )


    return (
        candidates,
        scored_candidates,
        matched_ids
    )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Delete previous output files
    # -----------------------------------------------------

    if os.path.exists(MATCHING_RESULTS_FILE):

        os.remove(
            MATCHING_RESULTS_FILE
        )


    if os.path.exists(CANDIDATE_PAIRS_FILE):

        os.remove(
            CANDIDATE_PAIRS_FILE
        )


    print()
    print("=" * 70)
    print("AMAZON ML BUSINESS ENTITY RESOLUTION")
    print("=" * 70)


    if SMOKE_TEST:

        print("MODE: SMOKE TEST")
        print("Records:", SMOKE_ROWS)

    else:

        print("MODE: FULL TEST DATA")


    print()
    print(
        "Source 1:",
        SOURCE1_FILE
    )

    print(
        "Source 2 index:",
        SOURCE2_INDEX
    )

    print(
        "Source 3 index:",
        SOURCE3_INDEX
    )

    print()


    # -----------------------------------------------------
    # Open SQLite databases
    # -----------------------------------------------------

    print(
        "Opening Source 2 SQLite index..."
    )

    conn2 = sqlite3.connect(
        SOURCE2_INDEX
    )


    print(
        "Opening Source 3 SQLite index..."
    )

    conn3 = sqlite3.connect(
        SOURCE3_INDEX
    )


    print(
        "Indexes opened successfully."
    )

    print()


    # -----------------------------------------------------
    # Create output files
    # -----------------------------------------------------

    with open(
        MATCHING_RESULTS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        f.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )


    with open(
        CANDIDATE_PAIRS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        f.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )


    # -----------------------------------------------------
    # Read Source 1 in chunks
    # -----------------------------------------------------

    reader_kwargs = {
        "sep": "\t",
        "dtype": str,
        "chunksize": CHUNK_SIZE
    }


    if SMOKE_TEST:

        reader_kwargs["nrows"] = SMOKE_ROWS


    source1_reader = pd.read_csv(
        SOURCE1_FILE,
        **reader_kwargs
    )


    total_processed = 0
    total_matches = 0
    total_candidates = 0


    try:

        for chunk_number, chunk in enumerate(
            source1_reader,
            start=1
        ):

            print(
                f"Processing chunk {chunk_number}..."
            )


            matching_lines = []
            candidate_lines = []


            # -------------------------------------------------
            # Process each Source 1 record
            # -------------------------------------------------

            for _, row in chunk.iterrows():

                source1_id = str(
                    row["entity_id"]
                )


                (
                    candidates,
                    scored_candidates,
                    matched_ids
                ) = process_source1_record(
                    row,
                    conn2,
                    conn3
                )


                # -------------------------------------------------
                # Candidate output
                # -------------------------------------------------

                candidate_ids = sorted(
                    candidates.keys()
                )


                candidate_string = ",".join(
                    candidate_ids
                )


                candidate_lines.append(
                    f"{source1_id}\t{candidate_string}\n"
                )


                # -------------------------------------------------
                # Matching output
                # -------------------------------------------------

                matched_string = ",".join(
                    matched_ids
                )


                matching_lines.append(
                    f"{source1_id}\t{matched_string}\n"
                )


                total_processed += 1

                total_candidates += len(
                    candidate_ids
                )

                total_matches += len(
                    matched_ids
                )


            # -------------------------------------------------
            # Write immediately
            # -------------------------------------------------

            with open(
                MATCHING_RESULTS_FILE,
                "a",
                encoding="utf-8",
                newline=""
            ) as f:

                f.writelines(
                    matching_lines
                )


            with open(
                CANDIDATE_PAIRS_FILE,
                "a",
                encoding="utf-8",
                newline=""
            ) as f:

                f.writelines(
                    candidate_lines
                )


            print(
                f"Processed: {total_processed:,}"
            )

            print(
                f"Matches so far: {total_matches:,}"
            )

            print(
                f"Candidates so far: {total_candidates:,}"
            )

            print()


    finally:

        conn2.close()
        conn3.close()


    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "Source 1 processed:",
        f"{total_processed:,}"
    )

    print(
        "Total predicted matches:",
        f"{total_matches:,}"
    )

    print(
        "Total candidate IDs:",
        f"{total_candidates:,}"
    )

    print()

    print(
        "Matching results:",
        MATCHING_RESULTS_FILE
    )

    print(
        "Candidate pairs:",
        CANDIDATE_PAIRS_FILE
    )

    print("=" * 70)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()