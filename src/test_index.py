import sqlite3


INDEX_FILE = "student_resource/source2_index.db"

TRUE_MATCH_ID = "S2-681193310"


conn = sqlite3.connect(INDEX_FILE)

print("Testing improved SQLite FTS search...")
print()


# Use AND so the candidate should contain BOTH useful words.
query = "maure AND colombier"


rows = conn.execute(
    """
    SELECT
        entity_id,
        business_name_normalized,
        business_address_normalized,
        country,
        bm25(businesses) AS rank
    FROM businesses
    WHERE businesses MATCH ?
    ORDER BY rank
    LIMIT 20
    """,
    (query,)
).fetchall()


print("Search query:", query)
print("Candidates found:", len(rows))
print()


for row in rows:

    print("ID      :", row[0])
    print("Name    :", row[1])
    print("Address :", row[2])
    print("Country :", row[3])
    print("Rank    :", row[4])
    print("-" * 50)


candidate_ids = {row[0] for row in rows}


print()


if TRUE_MATCH_ID in candidate_ids:
    print("SUCCESS: True match found in AND search!")
else:
    print("WARNING: True match not found in first 20 results.")


conn.close()