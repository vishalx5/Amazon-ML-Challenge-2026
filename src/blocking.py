import pandas as pd


def create_blocking_keys(df):
    """
    Create blocking keys.
    """

    df = df.copy()

    df["name_key"] = (
        df["business_name_normalized"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["address_key"] = (
        df["business_address_normalized"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["country_key"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    df["name_country_key"] = (
        df["name_key"] + "||" + df["country_key"]
    )

    df["address_country_key"] = (
        df["address_key"] + "||" + df["country_key"]
    )

    return df


def get_candidate_ids(source1_row, source_df):
    """
    Exact blocking using normalized name/address.
    """

    name = str(
        source1_row.get("business_name_normalized", "") or ""
    ).strip()

    address = str(
        source1_row.get("business_address_normalized", "") or ""
    ).strip()

    country = str(
        source1_row.get("country", "") or ""
    ).strip().casefold()

    candidates = set()

    if name:
        candidates.update(
            source_df.loc[
                source_df["name_key"] == name,
                "entity_id"
            ].tolist()
        )

    if address:
        candidates.update(
            source_df.loc[
                source_df["address_key"] == address,
                "entity_id"
            ].tolist()
        )

    if name and country:
        candidates.update(
            source_df.loc[
                source_df["name_country_key"]
                == f"{name}||{country}",
                "entity_id"
            ].tolist()
        )

    if address and country:
        candidates.update(
            source_df.loc[
                source_df["address_country_key"]
                == f"{address}||{country}",
                "entity_id"
            ].tolist()
        )

    return candidates


def get_name_tokens(name):
    """
    Extract useful business-name tokens.
    """

    if not name:
        return set()

    stop_words = {
        "inc",
        "llc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "company",
        "co",
        "and",
        "the"
    }

    return {
        token
        for token in str(name).split()
        if len(token) >= 3 and token not in stop_words
    }


def build_token_index(source_df):
    """
    Build an inverted index once.

    Instead of scanning millions of rows for every Source 1
    record, we create:

        token -> set(entity_ids)
    """

    token_index = {}

    for entity_id, name in zip(
        source_df["entity_id"],
        source_df["business_name_normalized"]
    ):

        tokens = get_name_tokens(name)

        for token in tokens:
            if token not in token_index:
                token_index[token] = set()

            token_index[token].add(entity_id)

    return token_index


def get_token_block_candidates(source1_row, token_index):
    """
    Get candidates from the pre-built token index.
    """

    source1_tokens = get_name_tokens(
        source1_row.get(
            "business_name_normalized",
            ""
        )
    )

    candidates = set()

    for token in source1_tokens:
        candidates.update(
            token_index.get(token, set())
        )

    return candidates


def get_all_candidates(source1_row, source_df, token_index=None):
    """
    Combine exact blocking and token blocking.
    """

    candidates = get_candidate_ids(
        source1_row,
        source_df
    )

    if token_index is not None:
        candidates.update(
            get_token_block_candidates(
                source1_row,
                token_index
            )
        )

    return candidates