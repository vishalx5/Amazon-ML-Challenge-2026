import re
import unicodedata
import pandas as pd


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = text.replace("&", " and ")
    text = text.replace("+", " and ")

    if text.strip() in {"null", "none", "nan"}:
        return ""

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith(("L", "N", "M")):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


def preprocess_dataframe(df):
    df = df.copy()

    df["business_name_normalized"] = (
        df["business_name"].apply(normalize_name)
    )

    df["business_address_normalized"] = (
        df["business_address"].apply(normalize_address)
    )

    return df
