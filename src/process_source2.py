import os
import pandas as pd

from preprocessing import preprocess_dataframe


INPUT_FILE = "dataset/train/train_source2.tsv"
OUTPUT_DIR = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_source2_clean.tsv")

CHUNK_SIZE = 100000


def process_source2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):
        clean_chunk = preprocess_dataframe(chunk)

        clean_chunk.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        total_rows += len(clean_chunk)

        print(f"Processed rows: {total_rows:,}")

        first_chunk = False

    print("\nSource 2 processing completed!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total rows: {total_rows:,}")


if __name__ == "__main__":
    process_source2()
