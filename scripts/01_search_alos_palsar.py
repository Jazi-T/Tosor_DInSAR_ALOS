#!/usr/bin/env python3

"""
Search ALOS/PALSAR and ALOS-2/PALSAR-2 SAR scenes
covering the Tosor Glacier-Lake-Moraine system.

Study area:
    Tosor Lake, Kyrgyz Republic
    41.961825 N, 77.373620 E

Search periods:
    ALOS/PALSAR:
        2009-01-01 to 2010-12-31

    ALOS-2/PALSAR-2:
        2023-01-01 to 2025-12-31

The script:
    1. Searches NASA Earthdata CMR / ASF catalog.
    2. Saves complete metadata as JSON.
    3. Extracts useful SAR metadata to CSV.
    4. Generates preliminary interferometric pair candidates.
    5. Does NOT download SAR scenes.
"""

from pathlib import Path
from itertools import combinations
import json
import math

import requests
import pandas as pd


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TOSOR STUDY AREA
# ============================================================

TOSOR_LAT = 41.961825
TOSOR_LON = 77.373620

# CMR point syntax is:
# longitude,latitude
TOSOR_POINT = f"{TOSOR_LON},{TOSOR_LAT}"


# ============================================================
# NASA EARTHDATA / ASF COLLECTIONS
# ============================================================

COLLECTIONS = {

    "ALOS_PALSAR_2009_2010": {
        "collection_id": "C1206485527-ASF",
        "mission": "ALOS",
        "sensor": "PALSAR",
        "product": "Level 1.1 SLC",
        "start": "2009-01-01T00:00:00Z",
        "end": "2010-12-31T23:59:59Z",
        "repeat_cycle_days": 46,
    },

    "ALOS2_PALSAR2_2023_2025": {
        "collection_id": "C3315903479-ASF",
        "mission": "ALOS-2",
        "sensor": "PALSAR-2",
        "product": "Level 1 ScanSAR",
        "start": "2023-01-01T00:00:00Z",
        "end": "2025-12-31T23:59:59Z",
        "repeat_cycle_days": 14,
    },
}


CMR_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.umm_json"
)

PAGE_SIZE = 2000


# ============================================================
# CMR SEARCH
# ============================================================

def search_cmr(collection_id, start, end):
    """
    Search NASA CMR for granules intersecting the Tosor point.
    """

    all_items = []
    page_num = 1

    print("\nSearching CMR")
    print(f"Collection: {collection_id}")
    print(f"Period:     {start} -> {end}")
    print(f"Point:      {TOSOR_POINT}")
    print("-" * 70)

    while True:

        params = {
            "collection_concept_id": collection_id,
            "point": TOSOR_POINT,
            "temporal": f"{start},{end}",
            "page_size": PAGE_SIZE,
            "page_num": page_num,
        }

        headers = {
            "Client-Id": "tosor-dinsar-research",
            "User-Agent": "Tosor-DInSAR-Research/1.0",
        }

        response = requests.get(
            CMR_URL,
            params=params,
            headers=headers,
            timeout=90,
        )

        response.raise_for_status()

        data = response.json()

        hits = data.get("hits", 0)
        items = data.get("items", [])

        print(
            f"Page {page_num}: "
            f"{len(items)} records "
            f"(CMR total hits: {hits})"
        )

        all_items.extend(items)

        if len(items) < PAGE_SIZE:
            break

        page_num += 1

    return all_items


# ============================================================
# HELPERS
# ============================================================

def safe_first(value, default=""):
    if isinstance(value, list) and value:
        return value[0]
    return default


def build_attribute_dictionary(umm):
    """
    Convert CMR AdditionalAttributes into a simpler dictionary.
    """

    result = {}

    attributes = umm.get("AdditionalAttributes", [])

    for attribute in attributes:

        name = str(
            attribute.get("Name", "")
        ).strip()

        values = attribute.get("Values", [])

        if not name:
            continue

        if isinstance(values, list):
            value = ";".join(
                str(v) for v in values
            )
        else:
            value = str(values)

        result[name] = value

    return result


def find_attribute(attributes, keywords):
    """
    Search AdditionalAttributes without relying on one exact
    provider-specific metadata field name.
    """

    for key, value in attributes.items():

        normalized = (
            key.lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        for keyword in keywords:

            if keyword.lower() in normalized:
                return value

    return ""


def get_download_url(umm):
    """
    Find GET DATA URL if present in CMR metadata.
    """

    related_urls = umm.get("RelatedUrls", [])

    for item in related_urls:

        url_type = str(
            item.get("Type", "")
        ).upper()

        url = item.get("URL", "")

        if "GET DATA" in url_type and url:
            return url

    return ""


def get_platform_info(umm):

    platforms = umm.get("Platforms", [])

    if not platforms:
        return "", ""

    platform = platforms[0]

    platform_name = platform.get(
        "ShortName",
        ""
    )

    instruments = platform.get(
        "Instruments",
        []
    )

    instrument_names = []

    for instrument in instruments:

        name = instrument.get(
            "ShortName",
            ""
        )

        if name:
            instrument_names.append(name)

    return (
        platform_name,
        ";".join(instrument_names)
    )


def get_temporal_info(umm):

    temporal = umm.get(
        "TemporalExtent",
        {}
    )

    range_time = temporal.get(
        "RangeDateTime",
        {}
    )

    start = range_time.get(
        "BeginningDateTime",
        ""
    )

    end = range_time.get(
        "EndingDateTime",
        ""
    )

    return start, end


def get_orbit_info(umm):

    domains = umm.get(
        "OrbitCalculatedSpatialDomains",
        []
    )

    if not domains:
        return "", "", ""

    domain = domains[0]

    orbit = domain.get(
        "OrbitNumber",
        ""
    )

    equator_time = domain.get(
        "EquatorCrossingDateTime",
        ""
    )

    equator_lon = domain.get(
        "EquatorCrossingLongitude",
        ""
    )

    return (
        orbit,
        equator_time,
        equator_lon
    )


# ============================================================
# CONVERT CMR METADATA TO TABLE
# ============================================================

def items_to_dataframe(items, mission_info):

    rows = []

    for item in items:

        meta = item.get(
            "meta",
            {}
        )

        umm = item.get(
            "umm",
            {}
        )

        attributes = build_attribute_dictionary(
            umm
        )

        platform, instrument = (
            get_platform_info(umm)
        )

        start_time, end_time = (
            get_temporal_info(umm)
        )

        (
            orbit_number,
            equator_time,
            equator_lon,
        ) = get_orbit_info(umm)

        granule_id = umm.get(
            "GranuleUR",
            meta.get("native-id", "")
        )

        # Try several possible metadata names.
        # Different ASF/JAXA collections may use
        # slightly different attribute names.

        path = find_attribute(
            attributes,
            [
                "path number",
                "path",
                "track",
            ],
        )

        frame = find_attribute(
            attributes,
            [
                "frame number",
                "frame",
            ],
        )

        beam_mode = find_attribute(
            attributes,
            [
                "beam mode",
                "beam",
            ],
        )

        polarization = find_attribute(
            attributes,
            [
                "polarization",
                "polarisation",
            ],
        )

        flight_direction = find_attribute(
            attributes,
            [
                "flight direction",
                "orbit direction",
                "ascending descending",
            ],
        )

        look_direction = find_attribute(
            attributes,
            [
                "look direction",
                "look",
            ],
        )

        processing_level = find_attribute(
            attributes,
            [
                "processing level",
                "product level",
            ],
        )

        row = {
            "mission": mission_info["mission"],
            "sensor": mission_info["sensor"],
            "expected_product":
                mission_info["product"],

            "granule_id": granule_id,

            "concept_id":
                meta.get("concept-id", ""),

            "acquisition_start":
                start_time,

            "acquisition_end":
                end_time,

            "platform":
                platform,

            "instrument":
                instrument,

            "orbit_number":
                orbit_number,

            "equator_crossing_time":
                equator_time,

            "equator_crossing_longitude":
                equator_lon,

            "path":
                path,

            "frame":
                frame,

            "beam_mode":
                beam_mode,

            "polarization":
                polarization,

            "flight_direction":
                flight_direction,

            "look_direction":
                look_direction,

            "processing_level":
                processing_level,

            "download_url":
                get_download_url(umm),

            # Keep original additional metadata
            # so nothing useful is lost.
            "additional_attributes":
                json.dumps(
                    attributes,
                    ensure_ascii=False,
                ),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:

        df["acquisition_start"] = (
            pd.to_datetime(
                df["acquisition_start"],
                utc=True,
                errors="coerce",
            )
        )

        df = df.sort_values(
            "acquisition_start"
        ).reset_index(drop=True)

    return df


# ============================================================
# INTERFEROMETRIC PAIR CANDIDATES
# ============================================================

def normalize_group_value(value):

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def create_pair_candidates(
    df,
    mission_name,
    repeat_cycle_days,
):
    """
    Generate preliminary pairs.

    Important:
    These are only candidate pairs.
    Perpendicular baseline and actual coherence
    must still be checked later.
    """

    if len(df) < 2:
        return pd.DataFrame()

    pair_rows = []

    # Metadata that should ideally match
    # for interferometric processing.
    geometry_columns = [
        "path",
        "frame",
        "beam_mode",
        "polarization",
        "flight_direction",
        "look_direction",
    ]

    for i, j in combinations(
        range(len(df)),
        2,
    ):

        scene1 = df.iloc[i]
        scene2 = df.iloc[j]

        date1 = scene1[
            "acquisition_start"
        ]

        date2 = scene2[
            "acquisition_start"
        ]

        if pd.isna(date1) or pd.isna(date2):
            continue

        temporal_baseline = abs(
            (date2 - date1).total_seconds()
            / 86400.0
        )

        # ----------------------------------------------------
        # Compare geometry metadata
        # ----------------------------------------------------

        compared_fields = []
        mismatched_fields = []

        for column in geometry_columns:

            value1 = normalize_group_value(
                scene1.get(column, "")
            )

            value2 = normalize_group_value(
                scene2.get(column, "")
            )

            # Only compare a field when metadata exists
            # for both scenes.
            if value1 and value2:

                compared_fields.append(column)

                if value1 != value2:
                    mismatched_fields.append(
                        column
                    )

        same_geometry_metadata = (
            len(mismatched_fields) == 0
        )

        # ----------------------------------------------------
        # Check repeat-cycle alignment
        # ----------------------------------------------------

        if repeat_cycle_days:

            repeat_number = max(
                1,
                round(
                    temporal_baseline
                    / repeat_cycle_days
                ),
            )

            expected_days = (
                repeat_number
                * repeat_cycle_days
            )

            cycle_difference = abs(
                temporal_baseline
                - expected_days
            )

        else:

            repeat_number = ""
            expected_days = ""
            cycle_difference = ""

        year1 = date1.year
        year2 = date2.year

        if year1 == year2:
            pair_type = "same_year"
        else:
            pair_type = "cross_year"

        pair_rows.append({

            "mission":
                mission_name,

            "scene_1":
                scene1["granule_id"],

            "date_1":
                date1,

            "scene_2":
                scene2["granule_id"],

            "date_2":
                date2,

            "year_1":
                year1,

            "year_2":
                year2,

            "pair_type":
                pair_type,

            "temporal_baseline_days":
                round(
                    temporal_baseline,
                    2,
                ),

            "repeat_cycle_days":
                repeat_cycle_days,

            "repeat_cycle_multiple":
                repeat_number,

            "expected_cycle_days":
                expected_days,

            "cycle_difference_days":
                round(
                    cycle_difference,
                    2,
                )
                if isinstance(
                    cycle_difference,
                    (int, float),
                )
                else "",

            "same_geometry_metadata":
                same_geometry_metadata,

            "metadata_compared":
                ",".join(
                    compared_fields
                ),

            "metadata_mismatches":
                ",".join(
                    mismatched_fields
                ),

            "path_1":
                scene1.get("path", ""),

            "path_2":
                scene2.get("path", ""),

            "frame_1":
                scene1.get("frame", ""),

            "frame_2":
                scene2.get("frame", ""),

            "beam_1":
                scene1.get(
                    "beam_mode",
                    "",
                ),

            "beam_2":
                scene2.get(
                    "beam_mode",
                    "",
                ),

            "polarization_1":
                scene1.get(
                    "polarization",
                    "",
                ),

            "polarization_2":
                scene2.get(
                    "polarization",
                    "",
                ),

            "direction_1":
                scene1.get(
                    "flight_direction",
                    "",
                ),

            "direction_2":
                scene2.get(
                    "flight_direction",
                    "",
                ),
        })

    pairs = pd.DataFrame(pair_rows)

    if pairs.empty:
        return pairs

    # Put likely candidates at the top:
    # 1. no known geometry mismatch
    # 2. closest to repeat-cycle multiple
    # 3. shorter temporal baseline

    pairs = pairs.sort_values(
        by=[
            "same_geometry_metadata",
            "cycle_difference_days",
            "temporal_baseline_days",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    return pairs.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TOSOR ALOS / PALSAR SEARCH")
    print("=" * 70)

    print(
        f"Study point: "
        f"{TOSOR_LAT} N, "
        f"{TOSOR_LON} E"
    )

    summary = []

    for name, info in COLLECTIONS.items():

        print("\n")
        print("=" * 70)
        print(name)
        print("=" * 70)

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        items = search_cmr(
            collection_id=info[
                "collection_id"
            ],
            start=info["start"],
            end=info["end"],
        )

        print(
            f"\nFound {len(items)} scenes."
        )

        # ----------------------------------------------------
        # Save raw JSON
        # ----------------------------------------------------

        raw_json_path = (
            METADATA_DIR
            / f"{name}_raw_cmr.json"
        )

        with open(
            raw_json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                items,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Inventory
        # ----------------------------------------------------

        df = items_to_dataframe(
            items,
            info,
        )

        inventory_path = (
            OUTPUT_DIR
            / f"{name}_inventory.csv"
        )

        df.to_csv(
            inventory_path,
            index=False,
        )

        print(
            f"Inventory saved:\n"
            f"  {inventory_path}"
        )

        # ----------------------------------------------------
        # Pair candidates
        # ----------------------------------------------------

        pairs = create_pair_candidates(
            df=df,
            mission_name=info[
                "mission"
            ],
            repeat_cycle_days=info[
                "repeat_cycle_days"
            ],
        )

        pair_path = (
            OUTPUT_DIR
            / f"{name}_pair_candidates.csv"
        )

        pairs.to_csv(
            pair_path,
            index=False,
        )

        print(
            f"Pair candidates saved:\n"
            f"  {pair_path}"
        )

        # ----------------------------------------------------
        # Print inventory
        # ----------------------------------------------------

        if df.empty:

            print(
                "\nNo scenes found."
            )

        else:

            display_columns = [
                "acquisition_start",
                "granule_id",
                "path",
                "frame",
                "beam_mode",
                "polarization",
                "flight_direction",
            ]

            existing_columns = [
                c for c in display_columns
                if c in df.columns
            ]

            print(
                "\nSCENE INVENTORY"
            )

            print(
                df[
                    existing_columns
                ].to_string(
                    index=False
                )
            )

        # ----------------------------------------------------
        # Print best preliminary pairs
        # ----------------------------------------------------

        if not pairs.empty:

            print(
                "\nTOP PRELIMINARY PAIRS"
            )

            pair_display = [
                "date_1",
                "date_2",
                "temporal_baseline_days",
                "repeat_cycle_multiple",
                "cycle_difference_days",
                "same_geometry_metadata",
            ]

            print(
                pairs[
                    pair_display
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

        summary.append({
            "dataset": name,
            "scenes_found":
                len(df),
            "pairs_generated":
                len(pairs),
        })

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    summary_path = (
        OUTPUT_DIR
        / "search_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT:"
        "\nThe generated pairs are only preliminary candidates."
        "\nDo NOT start DInSAR processing yet."
        "\nWe still need to check:"
        "\n  - path/track"
        "\n  - frame"
        "\n  - beam mode"
        "\n  - polarization"
        "\n  - ascending/descending geometry"
        "\n  - perpendicular baseline"
        "\n  - seasonal conditions"
        "\n  - expected coherence"
    )


if __name__ == "__main__":
    main()