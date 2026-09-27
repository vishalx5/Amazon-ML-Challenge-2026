import pandas as pd
from preprocessing import preprocess_dataframe


INPUT_FILE = "dataset/test/test_source1.tsv"
OUTPUT_FILE = "processed/test_source1_clean.tsv"

CHUNK_SIZE = 100000

first_chunk = True
total_rows = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    sep="\t",
    chunksize=CHUNK_SIZE
):
    processed_chunk = preprocess_dataframe(chunk)

    processed_chunk.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_chunk else "a",
        header=first_chunk
    )

    total_rows += len(processed_chunk)
    first_chunk = False

    print(f"Processed rows: {total_rows:,}")

print("\nTest Source 1 processing completed!")
print(f"Output file: {OUTPUT_FILE}")
print(f"Total rows: {total_rows:,}")