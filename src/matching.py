import re

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio
)


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(value):

    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    return text.lower().strip()


# =========================================================
# TEXT SIMILARITY
# =========================================================

def similarity_score(text1, text2):

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    r = ratio(
        text1,
        text2
    )

    token_sort = token_sort_ratio(
        text1,
        text2
    )

    token_set = token_set_ratio(
        text1,
        text2
    )

    return max(
        r,
        token_sort,
        token_set
    )


# =========================================================
# TOKEN OVERLAP
# =========================================================

def token_overlap(text1, text2):

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    tokens1 = set(
        re.findall(
            r"[A-Za-z0-9\u0900-\u097F]+",
            text1
        )
    )

    tokens2 = set(
        re.findall(
            r"[A-Za-z0-9\u0900-\u097F]+",
            text2
        )
    )

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if not union:
        return 0.0

    return (
        len(intersection)
        / len(union)
    ) * 100.0


# =========================================================
# COUNTRY SCORE
# =========================================================

def country_score(country1, country2):

    country1 = clean_text(country1)
    country2 = clean_text(country2)

    if not country1 or not country2:
        return 0.0

    if country1 == country2:
        return 100.0

    return 0.0


# =========================================================
# CALCULATE FINAL MATCH SCORE
# =========================================================

def calculate_match_score(
    source1,
    candidate
):

    name_score = similarity_score(
        source1.get("business_name"),
        candidate.get("business_name")
    )

    address_score = similarity_score(
        source1.get("business_address"),
        candidate.get("business_address")
    )

    country_score_value = country_score(
        source1.get("country"),
        candidate.get("country")
    )

    name_overlap = token_overlap(
        source1.get("business_name"),
        candidate.get("business_name")
    )

    address_overlap = token_overlap(
        source1.get("business_address"),
        candidate.get("business_address")
    )


    # -----------------------------------------------------
    # Weighted score
    # -----------------------------------------------------

    final_score = (
        0.50 * name_score
        +
        0.35 * address_score
        +
        0.15 * country_score_value
    )


    # -----------------------------------------------------
    # Token overlap bonuses
    # -----------------------------------------------------

    if name_overlap >= 50:

        final_score += 5


    if address_overlap >= 50:

        final_score += 5


    # Keep score between 0 and 100

    final_score = min(
        100.0,
        final_score
    )


    return {
        "name_score": round(
            name_score,
            2
        ),

        "address_score": round(
            address_score,
            2
        ),

        "country_score": round(
            country_score_value,
            2
        ),

        "name_overlap": round(
            name_overlap,
            2
        ),

        "address_overlap": round(
            address_overlap,
            2
        ),

        "final_score": round(
            final_score,
            2
        )
    }


# =========================================================
# MATCH DECISION
# =========================================================

def is_match(scores):

    name_score = scores["name_score"]
    address_score = scores["address_score"]
    final_score = scores["final_score"]

    name_overlap = scores["name_overlap"]
    address_overlap = scores["address_overlap"]


    # =====================================================
    # RULE 1: Extremely strong business-name match
    #
    # Useful when the address is missing or unreliable.
    # Example:
    # Maure Williams Colombier
    # vs
    # Maure Wilblims Colombier Inc
    # =====================================================

    if (
        name_score >= 92
        and final_score >= 60
    ):
        return True


    # =====================================================
    # RULE 2: Extremely strong address match
    #
    # Useful for cross-language / transliteration cases
    # where the business names can look completely different.
    # =====================================================

    if (
        address_score >= 88
        and final_score >= 65
    ):
        return True


    # =====================================================
    # RULE 3: Strong name + strong address
    # =====================================================

    if (
        name_score >= 78
        and address_score >= 70
        and final_score >= 75
    ):
        return True


    # =====================================================
    # RULE 4: Strong token overlap on BOTH fields
    # =====================================================

    if (
        name_score >= 70
        and address_score >= 65
        and name_overlap >= 40
        and address_overlap >= 35
        and final_score >= 72
    ):
        return True


    return False

    name_score = scores["name_score"]

    address_score = scores["address_score"]

    final_score = scores["final_score"]


    # -----------------------------------------------------
    # Very strong name match
    # -----------------------------------------------------

    if (
        name_score >= 90
        and final_score >= 65
    ):

        return True


    # -----------------------------------------------------
    # Strong address + reasonable name
    # -----------------------------------------------------

    if (
        address_score >= 80
        and name_score >= 55
        and final_score >= 65
    ):

        return True


    # -----------------------------------------------------
    # General matching rule
    # -----------------------------------------------------

    if (
        name_score >= name_threshold
        and address_score >= address_threshold
        and final_score >= final_threshold
    ):

        return True


    return False