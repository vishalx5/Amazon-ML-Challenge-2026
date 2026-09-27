import sqlite3
import os
import pandas as pd

from preprocessing import preprocess_dataframe


INPUT_FILE = "student_resource/dataset/train/train_source2.tsv"
INDEX_FILE = "student_resource/source2_index.db"

CHUNK_SIZE = 20000


def build_source2_index():

    # Remove old test index if it exists
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)

    conn = sqlite3.connect(INDEX_FILE)

    # FTS5 table:
    # entity_id is stored but not searched.
    conn.execute("""
        CREATE VIRTUAL TABLE businesses USING fts5(
            entity_id UNINDEXED,
            business_name_normalized,
            business_address_normalized,
            country UNINDEXED
        )
    """)

    conn.commit()

    total_rows = 0

    print("Building Source 2 disk index...")
    print("This will process the file in chunks.")
    print()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            sep="\t",
            chunksize=CHUNK_SIZE,
            dtype=str
        ),
        start=1
    ):

        chunk = preprocess_dataframe(chunk)

        rows = []

        for _, row in chunk.iterrows():

            rows.append((
                str(row["entity_id"]),
                str(row["business_name_normalized"])
                if pd.notna(row["business_name_normalized"])
                else "",
                str(row["business_address_normalized"])
                if pd.notna(row["business_address_normalized"])
                else "",
                str(row["country"])
                if pd.notna(row["country"])
                else ""
            ))

        conn.executemany(
            """
            INSERT INTO businesses (
                entity_id,
                business_name_normalized,
                business_address_normalized,
                country
            )
            VALUES (?, ?, ?, ?)
            """,
            rows
        )

        conn.commit()

        total_rows += len(chunk)

        print(
            f"Processed chunk {chunk_number} | "
            f"Rows: {total_rows:,}"
        )

    conn.close()

    print()
    print("========================================")
    print("Source 2 index completed successfully!")
    print("Rows indexed:", f"{total_rows:,}")
    print("Index:", INDEX_FILE)
    print("========================================")


if __name__ == "__main__":
    build_source2_index()