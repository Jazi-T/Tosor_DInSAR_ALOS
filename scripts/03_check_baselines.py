from pathlib import Path
import requests
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"

ALOS_FILE = TABLE_DIR / "alos_ranked_dinsar_pairs.csv"

OUTPUT_FILE = (
    TABLE_DIR /
    "alos_baseline_checked_pairs.csv"
)

BASELINE_URL = (
    "https://api.daac.asf.alaska.edu/"
    "services/search/baseline"
)


def normalize_alos_scene(scene):

    if pd.isna(scene):
        return ""

    scene = str(scene).strip()

    suffixes = [
        ".zip",
        ".ZIP",
        "-L1.1",
        "-L1.5",
    ]

    for suffix in suffixes:
        if scene.endswith(suffix):
            scene = scene[:-len(suffix)]

    return scene


def get_feature_identifier(properties):

    possible_keys = [
        "sceneName",
        "granuleName",
        "fileID",
        "fileName",
        "name",
    ]

    values = []

    for key in possible_keys:

        value = properties.get(key)

        if value:
            values.append(str(value))

    return values


def query_baseline(reference_scene):

    params = {
        "reference": reference_scene,
        "processingLevel": "L1.1",
        "output": "geojson",
    }

    print("\nASF Baseline query:")
    print("Reference:", reference_scene)

    try:
        response = requests.get(
            BASELINE_URL,
            params=params,
            timeout=120,
        )
    except requests.RequestException as exc:
        print(f"ASF request failed (no response): {exc}")
        return None

    print("HTTP status:", response.status_code)

    if not response.ok:

        print("ASF response:")
        print(response.text)

        return None

    try:
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("features"), list):
            raise ValueError("Expected GeoJSON features")
        return data

    except Exception:

        print("Response was not valid JSON:")
        print(response.text)

        return None


def match_secondary(
    geojson,
    secondary_scene,
    secondary_date,
):

    secondary_norm = normalize_alos_scene(
        secondary_scene
    ).upper()

    features = geojson.get(
        "features",
        []
    )

    # --------------------------------------------------
    # First: scene ID
    # --------------------------------------------------

    for feature in features:

        props = feature.get(
            "properties",
            {}
        )

        identifiers = (
            get_feature_identifier(props)
        )

        for identifier in identifiers:

            normalized = (
                normalize_alos_scene(
                    identifier
                )
                .upper()
            )

            if (
                secondary_norm == normalized
                or secondary_norm in normalized
                or normalized in secondary_norm
            ):

                return feature, "scene_id"

    # --------------------------------------------------
    # Fallback: acquisition date
    # --------------------------------------------------

    target_date = pd.to_datetime(
        secondary_date,
        utc=True,
        errors="coerce",
    )

    if pd.isna(target_date):
        return None, ""

    possible_date_keys = [
        "startTime",
        "sceneDate",
        "acquisitionDate",
    ]

    for feature in features:

        props = feature.get(
            "properties",
            {}
        )

        for key in possible_date_keys:

            value = props.get(key)

            if not value:
                continue

            date = pd.to_datetime(
                value,
                utc=True,
                errors="coerce",
            )

            if pd.isna(date):
                continue

            if (
                date.date()
                == target_date.date()
            ):
                return (
                    feature,
                    "acquisition_date",
                )

    return None, ""


def main():

    df = pd.read_csv(ALOS_FILE)

    results = []

    print("=" * 80)
    print("ALOS/PALSAR BASELINE CHECK")
    print("=" * 80)

    for i, row in df.iterrows():

        ref_original = row["scene_1"]
        sec_original = row["scene_2"]

        reference = normalize_alos_scene(
            ref_original
        )

        secondary = normalize_alos_scene(
            sec_original
        )

        print("\n" + "-" * 80)
        print(f"PAIR {i + 1}")
        print("Original reference :", ref_original)
        print("ASF reference      :", reference)
        print("Original secondary :", sec_original)
        print("ASF secondary      :", secondary)
        print(
            "Dates              :",
            row["date_1"],
            "->",
            row["date_2"],
        )

        output = row.to_dict()

        output["reference_scene_asf"] = (
            reference
        )

        output["secondary_scene_asf"] = (
            secondary
        )

        output[
            "asf_temporal_baseline_days"
        ] = None

        output[
            "perpendicular_baseline_m"
        ] = None

        output["baseline_status"] = ""
        output["match_method"] = ""

        geojson = query_baseline(
            reference
        )

        if geojson is None:

            output[
                "baseline_status"
            ] = "api_query_failed"

            results.append(output)
            continue

        print(
            "Baseline stack size:",
            len(
                geojson.get(
                    "features",
                    []
                )
            )
        )

        feature, method = match_secondary(
            geojson,
            secondary,
            row["date_2"],
        )

        if feature is None:

            print(
                "Secondary scene was not "
                "found in baseline stack."
            )

            output[
                "baseline_status"
            ] = "secondary_not_found"

            results.append(output)
            continue

        props = feature.get(
            "properties",
            {}
        )

        temporal = props.get(
            "temporalBaseline"
        )

        perpendicular = props.get(
            "perpendicularBaseline"
        )

        output[
            "asf_temporal_baseline_days"
        ] = temporal

        output[
            "perpendicular_baseline_m"
        ] = perpendicular

        output["baseline_status"] = (
            "success" if pd.notna(pd.to_numeric(temporal, errors="coerce"))
            and pd.notna(pd.to_numeric(perpendicular, errors="coerce"))
            else "baseline_missing"
        )
        output["asf_scene_identifier"] = next(iter(get_feature_identifier(props)), "")
        output["asf_acquisition_date"] = next(
            (props[k] for k in ("startTime", "sceneDate", "acquisitionDate") if props.get(k)), ""
        )
        output["asf_path"] = props.get("pathNumber", props.get("path"))
        output["asf_frame"] = props.get("frameNumber", props.get("frame"))

        output[
            "match_method"
        ] = method

        print(
            "Temporal baseline     :",
            temporal,
            "days"
        )

        print(
            "Perpendicular baseline:",
            perpendicular,
            "m"
        )

        results.append(output)

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    cols = [
        "date_1",
        "date_2",
        "reference_scene_asf",
        "secondary_scene_asf",
        "asf_temporal_baseline_days",
        "perpendicular_baseline_m",
        "baseline_status",
    ]

    print(
        result_df[
            [
                c for c in cols
                if c in result_df.columns
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    alos2 = pd.read_csv(TABLE_DIR / "alos2_ranked_dinsar_pairs.csv")
    alos2["perpendicular_baseline_m"] = pd.NA
    alos2["baseline_status"] = "requires_orbit_metadata"
    alos2["baseline_notes"] = "Calculate from ALOS-2 orbit/leader metadata."
    alos2.to_csv(TABLE_DIR / "alos2_baseline_review_pairs.csv", index=False)
    print(f"ALOS-2: {len(alos2)} pairs require orbit metadata.")
    print("No SAR scenes were downloaded.")


if __name__ == "__main__":
    main()