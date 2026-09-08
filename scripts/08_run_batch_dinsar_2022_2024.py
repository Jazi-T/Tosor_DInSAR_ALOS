#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import asf_search as asf


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path("/home/jazi/Desktop/Tosor_DInSAR_ALOS")

NETWORK_SCRIPT = PROJECT / "scripts/07_build_alos2_network_2022_2024.py"

TABLES = PROJECT / "output/tables"

INVENTORY_CSV = TABLES / "alos2_2022_2024_inventory.csv"

SELECTED_CSV = (
    TABLES /
    "alos2_2022_2024_selected_multiple_pairs.csv"
)

# Heavy data are already symlinked to the 8 TB disk.
RAW_ROOT = PROJECT / "data/raw/ALOS2_2022_2024"

ZIP_ROOT = RAW_ROOT / "zips"
EXTRACT_ROOT = RAW_ROOT / "extracted"

PROCESS_ROOT = (
    PROJECT /
    "processing/ALOS2_batch_2022_2024"
)

SUMMARY_CSV = (
    TABLES /
    "alos2_2022_2024_dinsar_results.csv"
)


# ============================================================
# CURRENT DEM
# ============================================================

# We already created this DEM.
DEM_SOURCE = (
    PROJECT /
    "processing/ALOS2_2023_2024/topo/dem.grd"
)

# Fallback bounds if DEM must be recreated.
DEM_W = 72.8170
DEM_E = 78.2580
DEM_S = 40.9430
DEM_N = 45.0560


# ============================================================
# TOSOR ROI
# ============================================================

ROI_W = 77.30
ROI_E = 77.45
ROI_S = 41.90
ROI_N = 42.02

ROI = (
    f"{ROI_W}/"
    f"{ROI_E}/"
    f"{ROI_S}/"
    f"{ROI_N}"
)


# ============================================================
# PROCESSING PARAMETERS
# ============================================================

# First quality threshold.
COHERENCE_THRESHOLD = 0.20

# At least this fraction of Tosor pixels must exceed threshold.
MIN_GOOD_PERCENT = 1.0

# 1 = parallel F1-F5 processing
PARALLEL = 1

# Only workflow currently validated for our GMTSAR setup.
ALLOWED_BEAM = "WBS"

# Require HH.
REQUIRED_POL = "HH"

# Process all selected compatible WBS/HH pairs.
# Set to an integer later if you intentionally want to limit runs.
MAX_PAIRS = None


# ============================================================
# SHELL ENVIRONMENT
# ============================================================

GMTSAR_ENV = PROJECT / "scripts/gmtsar_env.sh"

FITOFFSET = PROJECT / "scripts/fitoffset.csh"


def shell(command: str,
          cwd: Path | None = None,
          log: Path | None = None,
          check: bool = True):

    """
    Run command under bash with GMTSAR environment.
    """

    prefix = f"""
set -o pipefail
source "{GMTSAR_ENV}"
export PATH="{PROJECT}/scripts:$PATH"
export LC_ALL=C
"""

    full = prefix + "\n" + command

    if log:

        log.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        full += (
            f'\n 2>&1 | tee "{log}"'
        )

    result = subprocess.run(
        [
            "bash",
            "-lc",
            full,
        ],
        cwd=str(cwd or PROJECT),
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {command}"
        )

    return result.returncode


def capture(command: str,
            cwd: Path | None = None):

    prefix = f"""
source "{GMTSAR_ENV}"
export PATH="{PROJECT}/scripts:$PATH"
export LC_ALL=C
"""

    return subprocess.check_output(
        [
            "bash",
            "-lc",
            prefix + "\n" + command,
        ],
        cwd=str(cwd or PROJECT),
        text=True,
    ).strip()


# ============================================================
# 1. BUILD / UPDATE NETWORK
# ============================================================

def build_network():

    print()
    print("=" * 70)
    print("1. BUILD ALOS-2 NETWORK 2022-2024")
    print("=" * 70)

    subprocess.run(
        [
            "python3",
            str(NETWORK_SCRIPT),
        ],
        cwd=PROJECT,
        check=True,
    )

    if not INVENTORY_CSV.exists():
        raise RuntimeError(
            f"Missing inventory: {INVENTORY_CSV}"
        )

    if not SELECTED_CSV.exists():
        raise RuntimeError(
            f"Missing selected pair table: {SELECTED_CSV}"
        )


# ============================================================
# 2. LOAD SELECTED PAIRS
# ============================================================

def load_pairs():

    inventory = pd.read_csv(
        INVENTORY_CSV
    )

    pairs = pd.read_csv(
        SELECTED_CSV
    )

    # ALOS-2 ScanSAR workflow we are using now:
    # WBS + common HH.
    pairs = pairs[
        pairs["beam_mode"]
        .astype(str)
        .str.upper()
        .eq(ALLOWED_BEAM)
    ].copy()

    pairs = pairs[
        pairs["common_polarization"]
        .astype(str)
        .str.contains(
            REQUIRED_POL,
            regex=False,
            na=False,
        )
    ].copy()

    pairs = pairs.sort_values(
        [
            "interval",
            "quality_flag",
            "priority_score",
        ]
    ).reset_index(drop=True)

    if MAX_PAIRS is not None:
        pairs = pairs.head(
            MAX_PAIRS
        )

    if pairs.empty:
        raise RuntimeError(
            "No WBS/HH selected pairs found."
        )

    print()
    print("=" * 70)
    print("2. PAIRS TO PROCESS")
    print("=" * 70)

    cols = [
        "interval",
        "date_1",
        "date_2",
        "path",
        "frame",
        "orbit_direction",
        "beam_mode",
        "common_polarization",
        "quality_flag",
    ]

    print(
        pairs[cols].to_string(
            index=False
        )
    )

    return inventory, pairs


# ============================================================
# PRODUCT NAME
# ============================================================

def get_product_row(
    inventory: pd.DataFrame,
    scene: str,
):

    rows = inventory[
        inventory["scene"]
        .astype(str)
        .eq(str(scene))
    ]

    if rows.empty:
        raise RuntimeError(
            f"Scene absent from inventory: {scene}"
        )

    return rows.iloc[0]


def product_stem(row):

    filename = str(
        row["file_name"]
    )

    if filename.endswith(".zip"):
        filename = filename[:-4]

    return filename


# ============================================================
# 3. EARTHDATA SESSION
# ============================================================

def earthdata_session():

    token = os.environ.get(
        "EARTHDATA_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "\nEARTHDATA_TOKEN is not set.\n\n"
            "Before running this script:\n"
            "read -r -s -p "
            "\"Earthdata token: \" EARTHDATA_TOKEN\n"
            "echo\n"
            "export EARTHDATA_TOKEN\n"
        )

    session = (
        asf.ASFSession()
        .auth_with_token(token)
    )

    return session


# ============================================================
# 4. DOWNLOAD
# ============================================================

def download_scene(
    scene: str,
    product_name: str,
    session,
):

    ZIP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        ZIP_ROOT /
        f"{product_name}.zip"
    )

    if target.exists():

        print(
            f"Already downloaded: "
            f"{target.name}"
        )

        return target

    print(
        f"Downloading: {scene}"
    )

    results = asf.granule_search(
        [scene]
    )

    if len(results) != 1:

        # Sometimes ASF prefers the full product identifier.
        results = asf.granule_search(
            [product_name]
        )

    if len(results) != 1:
        raise RuntimeError(
            f"{scene}: ASF returned "
            f"{len(results)} products"
        )

    results[0].download(
        path=str(ZIP_ROOT),
        session=session,
    )

    if not target.exists():

        candidates = list(
            ZIP_ROOT.glob(
                f"*{scene}*.zip"
            )
        )

        if len(candidates) == 1:
            target = candidates[0]

    if not target.exists():
        raise RuntimeError(
            f"Downloaded ZIP not found: "
            f"{product_name}"
        )

    print(
        f"Download complete: "
        f"{target.name} "
        f"{target.stat().st_size / 1024**3:.2f} GiB"
    )

    return target


# ============================================================
# 5. EXTRACT
# ============================================================

def extract_scene(
    zipfile: Path,
    product_name: str,
):

    destination = (
        EXTRACT_ROOT /
        product_name
    )

    complete_flag = (
        destination /
        ".extraction_complete"
    )

    if complete_flag.exists():
        return destination

    if destination.exists():
        shutil.rmtree(
            destination
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Extracting: {zipfile.name}"
    )

    subprocess.run(
        [
            "unzip",
            "-q",
            str(zipfile),
            "-d",
            str(destination),
        ],
        check=True,
    )

    complete_flag.touch()

    return destination


# ============================================================
# FIND RAW CEOS FILE
# ============================================================

def find_one(
    root: Path,
    pattern: str,
):

    matches = list(
        root.rglob(pattern)
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{pattern}: expected one file, "
            f"found {len(matches)} under {root}"
        )

    return matches[0]


# ============================================================
# 6. PREPARE PAIR DIRECTORY
# ============================================================

def pair_id(
    row,
    master_product,
    secondary_product,
):

    d1 = pd.Timestamp(
        row["date_1"]
    ).strftime("%Y%m%d")

    d2 = pd.Timestamp(
        row["date_2"]
    ).strftime("%Y%m%d")

    orbit = str(
        row["orbit_direction"]
    ).upper()[:3]

    path = int(
        row["path"]
    )

    frame = int(
        row["frame"]
    )

    return (
        f"{d1}_{d2}_"
        f"P{path}_F{frame}_"
        f"{orbit}_WBS_HH"
    )


def safe_symlink(
    source: Path,
    destination: Path,
):

    if destination.exists() \
       or destination.is_symlink():
        destination.unlink()

    destination.symlink_to(
        source.resolve()
    )


def prepare_pair(
    pid: str,
    master_product: str,
    secondary_product: str,
    master_extract: Path,
    secondary_extract: Path,
):

    work = (
        PROCESS_ROOT /
        pid
    )

    raw = work / "raw"
    topo = work / "topo"
    logs = work / "logs"
    results = work / "results"

    for folder in [
        raw,
        topo,
        logs,
        results,
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    for product, source_root in [
        (
            master_product,
            master_extract,
        ),
        (
            secondary_product,
            secondary_extract,
        ),
    ]:

        led = find_one(
            source_root,
            f"LED-{product}",
        )

        safe_symlink(
            led,
            raw / led.name,
        )

        for swath in range(
            1,
            6,
        ):

            img = find_one(
                source_root,
                f"IMG-HH-{product}-F{swath}",
            )

            safe_symlink(
                img,
                raw / img.name,
            )

    if not DEM_SOURCE.exists():
        raise RuntimeError(
            f"DEM missing: {DEM_SOURCE}"
        )

    safe_symlink(
        DEM_SOURCE,
        topo / "dem.grd",
    )

    return work


# ============================================================
# 7. CONFIG
# ============================================================

def create_config(
    work: Path,
):

    config = (
        work /
        "config.alos2.txt"
    )

    shell(
        (
            "pop_config.csh "
            "ALOS2_SCAN "
            "> config.alos2.txt"
        ),
        cwd=work,
    )

    text = config.read_text()

    replacements = {
        "proc_stage": "1",
        "topo_phase": "1",

        # First generate interferogram + coherence.
        "threshold_snaphu": "0",

        # Geocode phase/coherence.
        "threshold_geocode": ".10",

        "correct_iono": "0",
        "iono_skip_est": "1",

        "mask_water": "1",
        "defomax": "0",
    }

    for key, value in (
        replacements.items()
    ):

        pattern = re.compile(
            rf"^[ \t]*{re.escape(key)}"
            rf"[ \t]*=.*$",
            flags=re.MULTILINE,
        )

        replacement = (
            f"{key} = {value}"
        )

        if pattern.search(text):
            text = pattern.sub(
                replacement,
                text,
            )
        else:
            text += (
                f"\n{replacement}\n"
            )

    config.write_text(text)

    return config


# ============================================================
# 8. GMTSAR P2P
# ============================================================

def run_p2p(
    work: Path,
    master_product: str,
    secondary_product: str,
):

    merge = work / "merge"

    corr_ll = (
        merge /
        "corr_ll.grd"
    )

    if corr_ll.exists():

        print(
            "Interferogram already exists."
        )

        return

    log = (
        work /
        "logs/p2p.log"
    )

    command = f"""
p2p_ALOS2_SCAN_Frame.csh \
"IMG-HH-{master_product}" \
"IMG-HH-{secondary_product}" \
config.alos2.txt \
{PARALLEL}
"""

    shell(
        command,
        cwd=work,
        log=log,
        check=True,
    )

    if not corr_ll.exists():
        raise RuntimeError(
            "p2p finished but "
            "merge/corr_ll.grd is missing"
        )


# ============================================================
# 9. TOSOR COHERENCE
# ============================================================

def coherence_analysis(
    work: Path,
):

    merge = work / "merge"
    results = work / "results"

    corr = (
        merge /
        "corr_ll.grd"
    )

    phase = (
        merge /
        "phasefilt_ll.grd"
    )

    corr_roi = (
        results /
        "corr_tosor_ll.grd"
    )

    phase_roi = (
        results /
        "phasefilt_tosor_ll.grd"
    )

    shell(
        f"""
gmt grdcut "{corr}" \
-R{ROI} \
-G"{corr_roi}"

gmt grdcut "{phase}" \
-R{ROI} \
-G"{phase_roi}"
""",
        cwd=work,
    )

    xyz = capture(
        f"""
gmt grd2xyz \
"{corr_roi}" \
-s
""",
        cwd=work,
    )

    values = []

    for line in xyz.splitlines():

        parts = line.split()

        if len(parts) < 3:
            continue

        try:
            value = float(
                parts[2]
            )
        except ValueError:
            continue

        if (
            math.isfinite(value)
            and 0 <= value <= 1
        ):
            values.append(
                value
            )

    if not values:

        return {
            "mean_coherence": np.nan,
            "median_coherence": np.nan,
            "good_pct_020": 0.0,
            "good_pct_030": 0.0,
            "good_pct_050": 0.0,
        }

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean_coherence":
            float(np.mean(arr)),

        "median_coherence":
            float(np.median(arr)),

        "good_pct_020":
            float(
                100 *
                np.mean(arr >= 0.20)
            ),

        "good_pct_030":
            float(
                100 *
                np.mean(arr >= 0.30)
            ),

        "good_pct_050":
            float(
                100 *
                np.mean(arr >= 0.50)
            ),
    }


# ============================================================
# 10. UNWRAP
# ============================================================

def unwrap(
    work: Path,
):

    merge = work / "merge"

    unwrap_grid = (
        merge /
        "unwrap.grd"
    )

    if not unwrap_grid.exists():

        shell(
            f"""
snaphu.csh \
{COHERENCE_THRESHOLD} \
0
""",
            cwd=merge,
            log=work / "logs/snaphu.log",
        )

    if not unwrap_grid.exists():
        raise RuntimeError(
            "unwrap.grd not created"
        )

    unwrap_ll = (
        merge /
        "unwrap_ll.grd"
    )

    if not unwrap_ll.exists():

        shell(
            """
proj_ra2ll.csh \
trans.dat \
unwrap.grd \
unwrap_ll.grd
""",
            cwd=merge,
        )

    if not unwrap_ll.exists():
        raise RuntimeError(
            "unwrap_ll.grd not created"
        )


# ============================================================
# 11. WAVELENGTH
# ============================================================

def find_wavelength(
    work: Path,
):

    prm_files = list(
        work.glob(
            "F1/**/*.PRM"
        )
    )

    if not prm_files:

        prm_files = list(
            work.glob(
                "F1/*.PRM"
            )
        )

    for prm in prm_files:

        for line in (
            prm
            .read_text(
                errors="ignore"
            )
            .splitlines()
        ):

            if (
                line.strip()
                .startswith(
                    "radar_wavelength"
                )
            ):

                parts = (
                    line
                    .replace("=", " ")
                    .split()
                )

                for token in (
                    parts[::-1]
                ):

                    try:
                        return float(
                            token
                        )
                    except ValueError:
                        continue

    raise RuntimeError(
        "radar_wavelength not found"
    )


# ============================================================
# 12. LOS
# ============================================================

def make_los(
    work: Path,
):

    merge = work / "merge"
    results = work / "results"

    wavelength = (
        find_wavelength(
            work
        )
    )

    # Positive LOS = toward satellite.
    scale = (
        -wavelength /
        (4 * math.pi)
    )

    full = (
        results /
        "los_m_full_ll.grd"
    )

    roi_m = (
        results /
        "los_m_tosor_ll.grd"
    )

    roi_cm = (
        results /
        "los_cm_tosor_ll.grd"
    )

    shell(
        f"""
gmt grdmath \
"{merge / 'unwrap_ll.grd'}" \
{scale:.15f} \
MUL \
= "{full}"

gmt grdcut \
"{full}" \
-R{ROI} \
-G"{roi_m}"

gmt grdmath \
"{roi_m}" \
100 \
MUL \
= "{roi_cm}"
""",
        cwd=work,
    )

    tif_ll = (
        results /
        "los_cm_tosor_4326.tif"
    )

    tif_utm = (
        results /
        "los_cm_tosor_32643.tif"
    )

    shell(
        f"""
gdal_translate \
-q \
-of GTiff \
-a_srs EPSG:4326 \
"{roi_cm}" \
"{tif_ll}"

gdalwarp \
-overwrite \
-q \
-t_srs EPSG:32643 \
-r bilinear \
-dstnodata -9999 \
"{tif_ll}" \
"{tif_utm}"
""",
        cwd=work,
    )

    return {
        "wavelength_m":
            wavelength,

        "los_tif":
            str(tif_utm),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    build_network()

    inventory, pairs = (
        load_pairs()
    )

    session = (
        earthdata_session()
    )

    PROCESS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_rows = []

    for index, row in (
        pairs.iterrows()
    ):

        print()
        print()
        print("=" * 70)
        print(
            f"PAIR "
            f"{index + 1} / "
            f"{len(pairs)}"
        )
        print("=" * 70)

        scene1 = str(
            row["scene_1"]
        )

        scene2 = str(
            row["scene_2"]
        )

        inv1 = get_product_row(
            inventory,
            scene1,
        )

        inv2 = get_product_row(
            inventory,
            scene2,
        )

        product1 = (
            product_stem(
                inv1
            )
        )

        product2 = (
            product_stem(
                inv2
            )
        )

        pid = pair_id(
            row,
            product1,
            product2,
        )

        print(
            "Pair ID:",
            pid,
        )

        print(
            row["date_1"],
            "->",
            row["date_2"],
        )

        status = {
            "pair_id":
                pid,

            "interval":
                row["interval"],

            "scene_1":
                scene1,

            "scene_2":
                scene2,

            "date_1":
                row["date_1"],

            "date_2":
                row["date_2"],

            "path":
                row["path"],

            "frame":
                row["frame"],

            "orbit_direction":
                row["orbit_direction"],

            "beam_mode":
                row["beam_mode"],

            "status":
                "STARTED",
        }

        try:

            zip1 = download_scene(
                scene1,
                product1,
                session,
            )

            zip2 = download_scene(
                scene2,
                product2,
                session,
            )

            extract1 = extract_scene(
                zip1,
                product1,
            )

            extract2 = extract_scene(
                zip2,
                product2,
            )

            work = prepare_pair(
                pid,
                product1,
                product2,
                extract1,
                extract2,
            )

            create_config(
                work
            )

            run_p2p(
                work,
                product1,
                product2,
            )

            coherence = (
                coherence_analysis(
                    work
                )
            )

            status.update(
                coherence
            )

            print()
            print(
                "Mean coherence:",
                coherence[
                    "mean_coherence"
                ],
            )

            print(
                "Pixels >= 0.20:",
                coherence[
                    "good_pct_020"
                ],
                "%",
            )

            if (
                coherence[
                    "good_pct_020"
                ]
                <
                MIN_GOOD_PERCENT
            ):

                print(
                    "PAIR REJECTED: "
                    "coherence too low."
                )

                status["status"] = (
                    "LOW_COHERENCE"
                )

                result_rows.append(
                    status
                )

                pd.DataFrame(
                    result_rows
                ).to_csv(
                    SUMMARY_CSV,
                    index=False,
                )

                continue

            unwrap(
                work
            )

            los = make_los(
                work
            )

            status.update(
                los
            )

            status["status"] = (
                "LOS_CREATED"
            )

            print()
            print(
                "LOS created:"
            )

            print(
                los["los_tif"]
            )

        except Exception as exc:

            status["status"] = (
                "FAILED"
            )

            status["error"] = (
                str(exc)
            )

            print()
            print(
                "FAILED:",
                exc,
            )

        result_rows.append(
            status
        )

        pd.DataFrame(
            result_rows
        ).to_csv(
            SUMMARY_CSV,
            index=False,
        )

    print()
    print("=" * 70)
    print("BATCH DINSAR COMPLETE")
    print("=" * 70)

    results = pd.DataFrame(
        result_rows
    )

    print(
        results[
            [
                "pair_id",
                "interval",
                "orbit_direction",
                "status",
                "mean_coherence",
                "good_pct_020",
            ]
        ].to_string(
            index=False
        )
        if "mean_coherence"
        in results.columns
        else results.to_string(
            index=False
        )
    )

    print()
    print(
        "Summary:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Generated rasters are LOS displacement."
    )

    print(
        "Positive = toward satellite; "
        "negative = away from satellite."
    )

    print(
        "Stable-bedrock referencing and "
        "ascending/descending decomposition "
        "are the next analysis stage before "
        "claiming vertical uplift/subsidence."
    )


if __name__ == "__main__":
    main()