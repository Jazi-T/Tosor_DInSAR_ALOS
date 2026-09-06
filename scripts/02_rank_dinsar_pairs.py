#!/usr/bin/env python3

"""
Rank preliminary ALOS/PALSAR and ALOS-2/PALSAR-2
interferometric pairs for the Tosor DInSAR project.

The script DOES NOT download SAR data.

Priority:
1. Same acquisition geometry
2. Cross-year pairs
3. Similar acquisition season
4. Temporal baseline close to mission repeat cycle multiples
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"

ALOS_FILE = (
    TABLE_DIR
    / "ALOS_PALSAR_2009_2010_pair_candidates.csv"
)

ALOS2_FILE = (
    TABLE_DIR
    / "ALOS2_PALSAR2_2023_2025_pair_candidates.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# For long-term analysis requested for this project,
# prioritize cross-year comparisons.
TARGET_YEAR_PAIRS = {
    "ALOS": [
        (2009, 2010),
    ],

    "ALOS-2": [
        (2023, 2024),
        (2024, 2025),
        (2023, 2025),
    ],
}


# Maximum difference in day-of-year.
# Example:
# 20 Aug vs 27 Aug = 7 days.
#
# Keeping similar seasons is important for reducing
# snow / freeze-thaw / glacier surface differences.
MAX_SEASON_DIFFERENCE_DAYS = 45


# ============================================================
# FUNCTIONS
# ============================================================

def circular_day_difference(date1, date2):
    """
    Difference between acquisition dates within the annual cycle.

    Ignores year and compares season/day-of-year.
    """

    doy1 = date1.dayofyear
    doy2 = date2.dayofyear

    diff = abs(doy1 - doy2)

    # Account for year boundary
    return min(diff, 365 - diff)


def target_year_pair(year1, year2, mission):
    """
    Check whether the pair corresponds to one of our
    intended long-term comparisons.
    """

    pair = tuple(sorted((int(year1), int(year2))))

    return pair in TARGET_YEAR_PAIRS.get(
        mission,
        []
    )


def calculate_score(row, mission):
    """
    Higher score = more promising preliminary DInSAR pair.

    This is NOT a substitute for perpendicular baseline
    or coherence evaluation.
    """

    score = 0

    # --------------------------------------------------------
    # 1. Geometry
    # --------------------------------------------------------

    if row.get("same_geometry_metadata", False):
        score += 50
    else:
        score -= 100

    # --------------------------------------------------------
    # 2. Desired year comparison
    # --------------------------------------------------------

    if target_year_pair(
        row["year_1"],
        row["year_2"],
        mission,
    ):
        score += 30

    # --------------------------------------------------------
    # 3. Similar season
    # --------------------------------------------------------

    seasonal_difference = row[
        "season_difference_days"
    ]

    if seasonal_difference <= 10:
        score += 25

    elif seasonal_difference <= 20:
        score += 20

    elif seasonal_difference <= 30:
        score += 15

    elif seasonal_difference <= 45:
        score += 5

    else:
        score -= 20

    # --------------------------------------------------------
    # 4. Repeat-cycle consistency
    # --------------------------------------------------------

    cycle_diff = row.get(
        "cycle_difference_days",
        np.nan,
    )

    if pd.notna(cycle_diff):

        if cycle_diff <= 1:
            score += 20

        elif cycle_diff <= 3:
            score += 10

        elif cycle_diff <= 7:
            score += 5

        else:
            score -= 10

    return score


def rank_pairs(filepath, mission):

    print("\n" + "=" * 80)
    print(f"{mission} PAIR RANKING")
    print("=" * 80)

    df = pd.read_csv(filepath)

    if df.empty:
        print("No pairs found.")
        return df

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    df["date_1"] = pd.to_datetime(
        df["date_1"],
        utc=True,
    )

    df["date_2"] = pd.to_datetime(
        df["date_2"],
        utc=True,
    )

    # --------------------------------------------------------
    # Seasonal difference
    # --------------------------------------------------------

    df["season_difference_days"] = df.apply(
        lambda row: circular_day_difference(
            row["date_1"],
            row["date_2"],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Intended year comparison?
    # --------------------------------------------------------

    df["target_year_pair"] = df.apply(
        lambda row: target_year_pair(
            row["year_1"],
            row["year_2"],
            mission,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Scientific priority score
    # --------------------------------------------------------

    df["priority_score"] = df.apply(
        lambda row: calculate_score(
            row,
            mission,
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Filter obvious unsuitable pairs
    # --------------------------------------------------------

    suitable = df[
        (df["same_geometry_metadata"] == True)
        &
        (df["target_year_pair"] == True)
        &
        (
            df["season_difference_days"]
            <= MAX_SEASON_DIFFERENCE_DAYS
        )
    ].copy()

    suitable = suitable.sort_values(
        by=[
            "priority_score",
            "season_difference_days",
            "cycle_difference_days",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = TABLE_DIR / (
        f"{mission.lower().replace('-', '')}"
        "_ranked_dinsar_pairs.csv"
    )

    suitable.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSuitable preliminary pairs: "
        f"{len(suitable)}"
    )

    print(
        f"Saved to:\n{output_path}"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    if not suitable.empty:

        columns = [
            "date_1",
            "date_2",
            "year_1",
            "year_2",
            "temporal_baseline_days",
            "season_difference_days",
            "repeat_cycle_multiple",
            "cycle_difference_days",
            "path_1",
            "frame_1",
            "beam_1",
            "polarization_1",
            "direction_1",
            "priority_score",
        ]

        existing = [
            col
            for col in columns
            if col in suitable.columns
        ]

        print("\nBEST CANDIDATES\n")

        print(
            suitable[
                existing
            ]
            .head(20)
            .to_string(index=False)
        )

    return suitable


# ============================================================
# MAIN
# ============================================================

def main():

    alos = rank_pairs(
        ALOS_FILE,
        "ALOS",
    )

    alos2 = rank_pairs(
        ALOS2_FILE,
        "ALOS-2",
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"ALOS 2009–2010 candidates: "
        f"{len(alos)}"
    )

    print(
        f"ALOS-2 2023–2025 candidates: "
        f"{len(alos2)}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese are still preliminary pairs."
        "\nBefore downloading/processing we must check:"
        "\n  1. actual acquisition geometry"
        "\n  2. perpendicular baseline"
        "\n  3. exact product type / processing level"
        "\n  4. scene footprint over Tosor"
        "\n  5. expected coherence"
    )


if __name__ == "__main__":
    main()