#!/usr/bin/env python3

"""
Build a complete ALOS-2 PALSAR-2 DInSAR candidate network
for the Tosor study area for 2022-2024.

Purpose
-------
1. Find ALL available ALOS-2 Level 1.1 acquisitions over Tosor.
2. Do NOT restrict the search to one acquisition per year.
3. Group scenes by identical acquisition geometry.
4. Generate compatible interferometric pairs for:
      2022 -> 2023
      2023 -> 2024
      2022 -> 2024
   plus useful within-year short pairs.
5. Rank pairs by:
      - identical path/frame
      - orbit direction
      - beam mode
      - look direction
      - common polarization
      - seasonal similarity
      - ALOS-2 14-day repeat-cycle consistency
      - temporal baseline
6. Select multiple high-quality pairs per geometry/year interval.
7. Save inventory, pair network, and download shortlist.

IMPORTANT
---------
This script does NOT claim uplift/subsidence.
It only identifies DInSAR candidate pairs.

Actual deformation must later be derived from:
interferogram -> coherence -> unwrap -> LOS displacement
-> stable-bedrock referencing -> interpretation.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from datetime import datetime
import math
import re
import sys

import numpy as np
import pandas as pd
import requests
import asf_search as asf


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path("/home/jazi/Desktop/Tosor_DInSAR_ALOS")

OUT = PROJECT / "output" / "tables"
META = PROJECT / "data" / "metadata"

OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)


# ============================================================
# STUDY AREA
# ============================================================

# Tosor Lake / moraine study point
LAT = 41.961825
LON = 77.373620

START = "2022-01-01T00:00:00Z"
END = "2024-12-31T23:59:59Z"

# Current NASA CMR collection used in this project.
CMR_SHORT_NAME = "ALOS2_L1_PSR2"

ALOS2_REPEAT_DAYS = 14

# We deliberately keep several pairs rather than one pair/year.
TOP_PAIRS_PER_GEOMETRY_INTERVAL = 5


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    return str(value).strip()


def normalize_pol(value):
    """Return sorted semicolon-separated polarization string."""

    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        vals = [str(x).upper().strip() for x in value]
    else:
        text = str(value).upper()

        vals = re.split(
            r"[;,/+\s]+",
            text,
        )

    vals = sorted(
        {
            x
            for x in vals
            if x in {"HH", "HV", "VH", "VV"}
        }
    )

    return ";".join(vals)


def pol_set(text):
    if not text:
        return set()

    return {
        x.strip().upper()
        for x in str(text).split(";")
        if x.strip()
    }


def common_polarization(a, b):
    common = sorted(pol_set(a) & pol_set(b))
    return ";".join(common)


def safe_int(value):
    if value is None:
        return None

    try:
        return int(float(value))
    except Exception:
        return None


def getprop(properties, *names):
    for name in names:
        if name in properties:
            value = properties.get(name)

            if value is not None:
                return value

    return None


def seasonal_difference_days(ts1, ts2):
    """
    Circular day-of-year difference based on month/day only.

    This compares acquisition season independently from acquisition year.
    """

    a = pd.Timestamp(ts1)
    b = pd.Timestamp(ts2)

    # Reference non-leap year.
    da = datetime(2001, a.month, a.day)
    db = datetime(2001, b.month, b.day)

    diff = abs((da - db).days)

    return min(diff, 365 - diff)


def summer_month(month):
    # High-mountain late melt / snow-minimum window.
    return month in {7, 8, 9}


def geometry_complete(row):
    required = [
        "path",
        "frame",
        "orbit_direction",
        "beam_mode",
    ]

    return all(
        row.get(key) not in (None, "", "nan")
        and not pd.isna(row.get(key))
        for key in required
    )


def same_geometry(a, b):
    """
    Conservative interferometric compatibility criterion.

    Path, frame, orbit direction, and beam mode must match.
    Polarization compatibility is checked separately.

    ``look_direction`` is intentionally not required because ASF metadata
    for these ALOS-2 records does not provide it.
    """

    if not geometry_complete(a) or not geometry_complete(b):
        return False

    keys = [
        "path",
        "frame",
        "orbit_direction",
        "beam_mode",
    ]

    return all(a.get(key) == b.get(key) for key in keys)


# ============================================================
# CMR DISCOVERY
# ============================================================

def cmr_granule_search():
    """
    Use NASA CMR to discover every ALOS-2 L1.1 granule intersecting
    the Tosor point during 2022-2024.

    CMR is used for broad collection discovery.
    ASF is subsequently used to normalize SAR metadata.
    """

    url = (
        "https://cmr.earthdata.nasa.gov/"
        "search/granules.umm_json"
    )

    page = 1
    all_items = []

    print("=" * 72)
    print("NASA CMR SEARCH")
    print("=" * 72)

    while True:

        params = {
            "short_name": CMR_SHORT_NAME,
            "point": f"{LON},{LAT}",
            "temporal": f"{START},{END}",
            "page_size": 2000,
            "page_num": page,
        }

        response = requests.get(
            url,
            params=params,
            timeout=120,
        )

        response.raise_for_status()

        payload = response.json()

        items = payload.get("items", [])

        if not items:
            break

        all_items.extend(items)

        print(
            f"CMR page {page}: "
            f"{len(items)} granules"
        )

        if len(items) < 2000:
            break

        page += 1

    if not all_items:
        raise RuntimeError(
            "CMR returned zero ALOS-2 granules. "
            "Check the collection short name or CMR availability."
        )

    records = []

    for item in all_items:

        meta = item.get("meta", {})
        umm = item.get("umm", {})

        granule_ur = (
            umm.get("GranuleUR")
            or meta.get("native-id")
            or meta.get("native_id")
        )

        if not granule_ur:
            continue

        records.append(
            {
                "granule_ur": granule_ur,
                "concept_id": meta.get("concept-id"),
                "revision_id": meta.get("revision-id"),
            }
        )

    cmr = pd.DataFrame(records)

    cmr = (
        cmr
        .drop_duplicates(subset=["granule_ur"])
        .sort_values("granule_ur")
        .reset_index(drop=True)
    )

    print()
    print("Unique CMR granules:", len(cmr))

    cmr.to_csv(
        OUT / "alos2_2022_2024_cmr_discovery.csv",
        index=False,
    )

    return cmr


# ============================================================
# ASF METADATA
# ============================================================

def asf_metadata(cmr):
    """
    Resolve CMR granules through ASF to obtain normalized SAR metadata.
    """

    names = cmr["granule_ur"].tolist()

    products = []

    print()
    print("=" * 72)
    print("ASF METADATA RESOLUTION")
    print("=" * 72)

    # Small batches are safer than one enormous granule_search request.
    batch_size = 50

    for start in range(0, len(names), batch_size):

        batch = names[start:start + batch_size]

        try:

            results = asf.granule_search(batch)

            products.extend(results)

            print(
                f"ASF batch "
                f"{start + 1}-{start + len(batch)}: "
                f"{len(results)} products"
            )

        except Exception as exc:

            print(
                "Batch search failed; retrying individually:",
                exc,
            )

            for name in batch:

                try:
                    results = asf.granule_search([name])
                    products.extend(results)

                except Exception as individual_exc:
                    print(
                        "WARNING:",
                        name,
                        individual_exc,
                    )

    records = []

    for product in products:

        p = product.properties

        scene = getprop(
            p,
            "sceneName",
            "productName",
            "fileID",
        )

        filename = getprop(
            p,
            "fileName",
            "fileID",
        )

        start_time = getprop(
            p,
            "startTime",
            "sceneDate",
        )

        if not start_time:
            continue

        date = pd.to_datetime(
            start_time,
            utc=True,
        )

        # Keep only requested years.
        if date.year not in {2022, 2023, 2024}:
            continue

        # Keep ALOS-2 Level 1.1 products.
        name_text = f"{scene or ''} {filename or ''}".upper()

        if "ALOS2" not in name_text:
            continue

        if "1.1" not in name_text:
            continue

        path = safe_int(
            getprop(
                p,
                "pathNumber",
                "path",
            )
        )

        frame = safe_int(
            getprop(
                p,
                "frameNumber",
                "frame",
            )
        )

        orbit = clean(
            getprop(
                p,
                "flightDirection",
                "orbitDirection",
            )
        )

        if orbit:
            orbit = orbit.upper()

        beam = clean(
            getprop(
                p,
                "beamModeType",
                "beamMode",
            )
        )

        if beam:
            beam = beam.upper()

        look = clean(
            getprop(
                p,
                "lookDirection",
            )
        )

        if look:
            look = look.upper()

        pol = normalize_pol(
            getprop(
                p,
                "polarization",
            )
        )

        records.append(
            {
                "scene": scene,
                "file_name": filename,
                "date": date.isoformat(),
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "path": path,
                "frame": frame,
                "orbit_direction": orbit,
                "beam_mode": beam,
                "polarization": pol,
                "look_direction": look,
                "processing_level": clean(
                    getprop(
                        p,
                        "processingLevel",
                    )
                ),
                "url": getprop(
                    p,
                    "url",
                ),
            }
        )

    inventory = pd.DataFrame(records)

    if inventory.empty:
        raise RuntimeError(
            "No usable ALOS-2 Level 1.1 ASF products "
            "were resolved from the CMR granules."
        )

    inventory = (
        inventory
        .drop_duplicates(
            subset=[
                "scene",
                "date",
                "path",
                "frame",
                "beam_mode",
            ]
        )
        .sort_values(
            [
                "date",
                "path",
                "frame",
            ]
        )
        .reset_index(drop=True)
    )

    inventory.to_csv(
        OUT / "alos2_2022_2024_inventory.csv",
        index=False,
    )

    return inventory


# ============================================================
# PAIR NETWORK
# ============================================================

def build_pair_network(inventory):

    records = []

    rows = inventory.to_dict("records")

    allowed_cross_year = {
        (2022, 2023),
        (2023, 2024),
        (2022, 2024),
    }

    for a, b in combinations(rows, 2):

        ta = pd.Timestamp(a["date"])
        tb = pd.Timestamp(b["date"])

        if tb <= ta:
            continue

        ya = ta.year
        yb = tb.year

        # Geometry first.
        if not same_geometry(a, b):
            continue

        common_pol = common_polarization(
            a["polarization"],
            b["polarization"],
        )

        if not common_pol:
            continue

        temporal_days = (
            tb - ta
        ).total_seconds() / 86400.0

        year_pair = (ya, yb)

        if ya == yb:

            # Useful short-term / seasonal interferograms.
            #
            # Require at least one repeat cycle and do not let
            # within-year network become unnecessarily huge.
            if temporal_days < 13:
                continue

            if temporal_days > 112:
                continue

            interval = f"{ya}_within_year"
            pair_type = "WITHIN_YEAR"

        elif year_pair in allowed_cross_year:

            interval = f"{ya}_{yb}"
            pair_type = "CROSS_YEAR"

        else:
            continue

        season_diff = seasonal_difference_days(
            ta,
            tb,
        )

        repeat_multiple = max(
            1,
            round(
                temporal_days
                / ALOS2_REPEAT_DAYS
            )
        )

        expected_repeat_days = (
            repeat_multiple
            * ALOS2_REPEAT_DAYS
        )

        repeat_residual = abs(
            temporal_days
            - expected_repeat_days
        )

        both_summer = (
            summer_month(ta.month)
            and summer_month(tb.month)
        )

        same_month = (
            ta.month == tb.month
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        if pair_type == "CROSS_YEAR":

            year_gap = yb - ya

            # ALOS-2 repeat-compatible "year":
            # 26 * 14 days = 364 days.
            target_days = (
                364.0 * year_gap
            )

            temporal_target_difference = abs(
                temporal_days
                - target_days
            )

        else:

            # For within-year pairs prefer shorter repeat-compatible
            # intervals, but without forcing only 14-day pairs.
            target_days = temporal_days

            temporal_target_difference = 0.0

        score = 0.0

        # Seasonal similarity is very important in glaciers/moraines.
        score += season_diff * 2.0

        # Repeat-cycle residual is a strong penalty.
        score += repeat_residual * 25.0

        # Annual / multiannual target proximity.
        score += temporal_target_difference * 0.15

        if both_summer:
            score -= 15.0

        if same_month:
            score -= 10.0

        if "HH" in pol_set(common_pol):
            score -= 5.0

        # WBS HH is straightforward for our current GMTSAR workflow.
        if (
            a["beam_mode"] == "WBS"
            and b["beam_mode"] == "WBS"
            and "HH" in pol_set(common_pol)
        ):
            score -= 5.0

        # ----------------------------------------------------
        # Quality flag
        # ----------------------------------------------------

        if (
            repeat_residual <= 0.25
            and (
                pair_type == "WITHIN_YEAR"
                or season_diff <= 14
            )
        ):
            quality = "HIGH"

        elif (
            repeat_residual <= 1.0
            and (
                pair_type == "WITHIN_YEAR"
                or season_diff <= 45
            )
        ):
            quality = "MEDIUM"

        else:
            quality = "LOW"

        records.append(
            {
                "pair_type": pair_type,
                "interval": interval,

                "scene_1": a["scene"],
                "date_1": a["date"],
                "year_1": ya,

                "scene_2": b["scene"],
                "date_2": b["date"],
                "year_2": yb,

                "temporal_baseline_days":
                    temporal_days,

                "season_difference_days":
                    season_diff,

                "repeat_cycle_days":
                    ALOS2_REPEAT_DAYS,

                "nearest_repeat_cycle_multiple":
                    repeat_multiple,

                "expected_repeat_cycle_days":
                    expected_repeat_days,

                "repeat_cycle_residual_days":
                    repeat_residual,

                "path": a["path"],
                "frame": a["frame"],

                "orbit_direction":
                    a["orbit_direction"],

                "beam_mode":
                    a["beam_mode"],

                "common_polarization":
                    common_pol,

                "look_direction":
                    a["look_direction"],

                "both_summer":
                    both_summer,

                "same_month":
                    same_month,

                "quality_flag":
                    quality,

                "priority_score":
                    round(score, 4),

                # Must be measured later from orbit processing.
                "perpendicular_baseline_m":
                    np.nan,

                # Must be measured after interferogram generation.
                "coherence_status":
                    "UNTESTED",

                "geometry_verified":
                    True,

                "url_1":
                    a["url"],

                "url_2":
                    b["url"],
            }
        )

    pairs = pd.DataFrame(records)

    if pairs.empty:
        raise RuntimeError(
            "No geometry-compatible ALOS-2 pairs found."
        )

    quality_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    pairs["_quality_order"] = (
        pairs["quality_flag"]
        .map(quality_order)
        .fillna(99)
    )

    pairs = (
        pairs
        .sort_values(
            [
                "pair_type",
                "interval",
                "path",
                "frame",
                "orbit_direction",
                "beam_mode",
                "_quality_order",
                "priority_score",
                "temporal_baseline_days",
            ]
        )
        .drop(columns="_quality_order")
        .reset_index(drop=True)
    )

    return pairs


# ============================================================
# SELECT MULTIPLE PAIRS
# ============================================================

def select_multiple_pairs(pairs):

    cross = pairs[
        pairs["pair_type"]
        == "CROSS_YEAR"
    ].copy()

    if cross.empty:
        return cross

    quality_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    cross["_quality_order"] = (
        cross["quality_flag"]
        .map(quality_order)
        .fillna(99)
    )

    cross = cross.sort_values(
        [
            "interval",
            "path",
            "frame",
            "orbit_direction",
            "beam_mode",
            "_quality_order",
            "priority_score",
        ]
    )

    group_cols = [
        "interval",
        "path",
        "frame",
        "orbit_direction",
        "beam_mode",
        "common_polarization",
    ]

    # IMPORTANT:
    # Do not select only one pair.
    #
    # Keep several best independent candidate interferograms
    # for every geometry and interannual interval.
    selected = (
        cross
        .groupby(
            group_cols,
            dropna=False,
            group_keys=False,
        )
        .head(
            TOP_PAIRS_PER_GEOMETRY_INTERVAL
        )
        .copy()
    )

    selected["selected"] = True

    selected = selected.drop(
        columns="_quality_order"
    )

    return selected


# ============================================================
# DOWNLOAD SHORTLIST
# ============================================================

def build_download_shortlist(
    inventory,
    selected,
):

    if selected.empty:
        return pd.DataFrame()

    selected_scenes = set(
        selected["scene_1"]
    ) | set(
        selected["scene_2"]
    )

    download = inventory[
        inventory["scene"].isin(
            selected_scenes
        )
    ].copy()

    download = download.sort_values(
        [
            "date",
            "path",
            "frame",
        ]
    )

    download["download_required"] = True

    return download


# ============================================================
# OUTPUT SUMMARY
# ============================================================

def print_summary(
    inventory,
    pairs,
    selected,
):

    print()
    print("=" * 72)
    print("ALOS-2 2022-2024 INVENTORY")
    print("=" * 72)

    yearly = (
        inventory
        .groupby("year")
        .size()
        .reindex(
            [2022, 2023, 2024],
            fill_value=0,
        )
    )

    for year, count in yearly.items():
        print(
            f"{year}: {count} acquisitions"
        )

    print()
    print("=" * 72)
    print("GEOMETRY GROUPS")
    print("=" * 72)

    geometry_cols = [
        "path",
        "frame",
        "orbit_direction",
        "beam_mode",
        "polarization",
        "look_direction",
    ]

    geom = (
        inventory
        .groupby(
            geometry_cols,
            dropna=False,
        )
        .agg(
            acquisitions=(
                "scene",
                "count",
            ),
            years=(
                "year",
                lambda x:
                    ",".join(
                        map(
                            str,
                            sorted(set(x))
                        )
                    ),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "acquisitions",
                "path",
                "frame",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
    )

    print(
        geom.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("PAIR COUNTS")
    print("=" * 72)

    print(
        pairs
        .groupby(
            [
                "pair_type",
                "interval",
                "quality_flag",
            ]
        )
        .size()
        .to_string()
    )

    print()
    print("=" * 72)
    print("SELECTED MULTIPLE CROSS-YEAR PAIRS")
    print("=" * 72)

    if selected.empty:

        print("No cross-year pair selected.")

    else:

        cols = [
            "interval",
            "date_1",
            "date_2",
            "path",
            "frame",
            "orbit_direction",
            "beam_mode",
            "common_polarization",
            "temporal_baseline_days",
            "season_difference_days",
            "repeat_cycle_residual_days",
            "quality_flag",
            "priority_score",
        ]

        print(
            selected[
                cols
            ].to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("TOSOR ALOS-2 DInSAR NETWORK 2022-2024")
    print("=" * 72)

    print(
        f"Study point: "
        f"{LAT:.6f}, {LON:.6f}"
    )

    print(
        f"Period: "
        f"{START} -> {END}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    cmr = cmr_granule_search()

    inventory = asf_metadata(cmr)

    # --------------------------------------------------------
    # Pairing
    # --------------------------------------------------------

    pairs = build_pair_network(
        inventory
    )

    selected = select_multiple_pairs(
        pairs
    )

    download = build_download_shortlist(
        inventory,
        selected,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pairs.to_csv(
        OUT /
        "alos2_2022_2024_all_compatible_pairs.csv",
        index=False,
    )

    for interval in [
        "2022_2023",
        "2023_2024",
        "2022_2024",
    ]:

        part = pairs[
            pairs["interval"]
            == interval
        ]

        part.to_csv(
            OUT /
            f"alos2_{interval}_pairs.csv",
            index=False,
        )

    within = pairs[
        pairs["pair_type"]
        == "WITHIN_YEAR"
    ]

    within.to_csv(
        OUT /
        "alos2_2022_2024_within_year_pairs.csv",
        index=False,
    )

    selected.to_csv(
        OUT /
        "alos2_2022_2024_selected_multiple_pairs.csv",
        index=False,
    )

    download.to_csv(
        OUT /
        "alos2_2022_2024_download_shortlist.csv",
        index=False,
    )

    print_summary(
        inventory,
        pairs,
        selected,
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    for name in [
        "alos2_2022_2024_inventory.csv",
        "alos2_2022_2024_all_compatible_pairs.csv",
        "alos2_2022_2023_pairs.csv",
        "alos2_2023_2024_pairs.csv",
        "alos2_2022_2024_pairs.csv",
        "alos2_2022_2024_within_year_pairs.csv",
        "alos2_2022_2024_selected_multiple_pairs.csv",
        "alos2_2022_2024_download_shortlist.csv",
    ]:
        print(
            OUT / name
        )

    print()
    print("NEXT SCIENTIFIC STEP:")
    print(
        "Measure perpendicular baseline and coherence "
        "for the selected pairs before interpreting deformation."
    )


if __name__ == "__main__":
    main()
