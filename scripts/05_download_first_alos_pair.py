#!/usr/bin/env python3
"""Interactively download only the verified 2009-08-22 / 2010-08-25 ALOS pair.

Credentials stay in memory. Archives are neither extracted nor committed.
Each product.download call is synchronous; effective processes = 1.
"""
from pathlib import Path
from getpass import getpass
import hashlib
import sys
import pandas as pd
import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'output/tables'
MANIFEST = TABLES / 'alos_2009_2010_download_manifest.csv'
DATES = ('2009-08-22', '2010-08-25')
FIELDS = {'path': 'pathNumber', 'frame': 'frameNumber', 'beam_mode': 'beamModeType',
          'polarization': 'polarization', 'orbit_direction': 'flightDirection'}


def norm(value):
    return str(value).strip().upper().replace('+', ';')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    shortlist = pd.read_csv(TABLES / 'final_dinsar_pair_shortlist.csv', dtype=str, keep_default_na=False)
    selected = shortlist[(shortlist.mission == 'ALOS') &
                         (shortlist.date_1.str[:10] == DATES[0]) & (shortlist.date_2.str[:10] == DATES[1])]
    require(len(selected) == 1, 'Expected exactly one historical ALOS pair in shortlist.')
    pair = selected.iloc[0]
    inventory = pd.read_csv(TABLES / 'alos_palsar_2009_2010_inventory.csv', dtype=str, keep_default_na=False)
    scenes = []
    for side, date in enumerate(DATES, 1):
        scene_id = pair[f'scene_{side}']
        matches = inventory[inventory.granule_id == scene_id]
        require(len(matches) == 1, f'Inventory cannot uniquely resolve {scene_id}.')
        scene = matches.iloc[0]
        require(scene.mission == 'ALOS' and scene.sensor == 'PALSAR', 'Mission/sensor conflict.')
        require(norm(scene.processing_level) in ('1.1', 'L1.1'), 'Product is not verified Level 1.1.')
        require(scene.acquisition_start[:10] == date, 'Inventory date conflict.')
        for field in FIELDS:
            require(bool(scene[field]) and norm(scene[field]) == norm(pair[field]), f'{field} conflict or missing metadata.')
            for suffix in (1, 2):
                key = f"{'beam' if field == 'beam_mode' else field}_{suffix}"
                if key in pair:
                    require(norm(pair[key]) == norm(scene[field]), f'{key} shortlist conflict.')
        scenes.append(scene)
    # Resolve both before authentication or any download; never choose the first fuzzy hit.
    results = asf.product_search([s.granule_id for s in scenes])
    prepared = []
    for scene, date in zip(scenes, DATES):
        matches = [p for p in results if p.properties.get('fileID') == scene.granule_id]
        require(len(matches) == 1, f'ASF did not uniquely resolve {scene.granule_id}.')
        product = matches[0]
        props = product.properties
        require(props.get('platform') == 'ALOS' and props.get('sensor') == 'PALSAR', 'ASF mission/sensor conflict.')
        require(norm(props.get('processingLevel')) in ('L1.1', '1.1'), 'ASF product is not Level 1.1.')
        require(str(props.get('startTime', ''))[:10] == date, 'ASF acquisition date conflict.')
        for field, asf_key in FIELDS.items():
            require(norm(props.get(asf_key, '')) == norm(scene[field]), f'ASF {field} conflict.')
        filename = props.get('fileName', '')
        require(bool(filename) and Path(filename).name == filename and filename.endswith('.zip'), 'Invalid ASF archive filename.')
        require(str(props.get('url', '')).startswith('https://'), 'Missing HTTPS download URL.')
        directory = ROOT / 'data/raw/ALOS_2009_2010' / date
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        require(not target.is_symlink(), 'Refusing a symlink archive target.')
        print(f'\nAcquisition date: {date}\nScene ID: {scene.granule_id}')
        for key in ('fileID', 'processingLevel', 'pathNumber', 'frameNumber', 'beamModeType',
                    'polarization', 'flightDirection', 'url', 'fileName', 'bytes'):
            print(f'{key}: {props.get(key, "unavailable")}')
        prepared.append((product, scene.granule_id, date, target))
    print('\nREADY TO DOWNLOAD HISTORICAL ALOS PAIR', flush=True)
    return prepared


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as archive:
        for chunk in iter(lambda: archive.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    prepared = prepare()
    # Fail closed without collecting credentials over a logged/noninteractive channel.
    if not sys.stdin.isatty():
        raise RuntimeError('Interactive terminal required for Earthdata username/password. Run this script in your Ubuntu terminal.')
    username = input('Earthdata username: ')
    password = getpass('Earthdata password: ')
    try:
        session = asf.ASFSession().auth_with_creds(username, password)
    except Exception:
        raise RuntimeError('Earthdata authentication failed; verify credentials and Earthdata access.') from None
    finally:
        del username, password
    records = []
    for product, scene, date, target in prepared:
        record = dict(acquisition_date=date, scene_id=scene, local_file=str(target.relative_to(ROOT)),
                      file_size_bytes=0, sha256='', download_status='failed')
        try:
            # Sequential synchronous product download: processes = 1 (no worker pool).
            if target.exists():
                expected = product.properties.get('bytes')
                require(expected and target.stat().st_size == int(expected),
                        f'Existing file cannot be verified by ASF size: {target}. Move it aside before retrying.')
            else:
                product.download(path=str(target.parent), filename=target.name, session=session)
            require(target.is_file() and target.stat().st_size > 0, f'Download missing or empty: {target.name}')
            require(target.name == product.properties['fileName'], 'Downloaded filename mismatch.')
            expected = product.properties.get('bytes')
            if expected:
                require(target.stat().st_size == int(expected), 'Downloaded size differs from ASF metadata.')
            record.update(file_size_bytes=target.stat().st_size, sha256=sha256(target), download_status='success')
        except Exception as exc:
            records.append(record)
            pd.DataFrame(records).to_csv(MANIFEST, index=False)
            # Do not serialize network exceptions that could include authentication details.
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f'Download or checksum failed for {scene} ({type(exc).__name__}); archive not verified.') from None
        records.append(record)
        pd.DataFrame(records).to_csv(MANIFEST, index=False)
    print('\n' + '='*60 + '\nALOS HISTORICAL PAIR DOWNLOAD\n' + '='*60)
    for row in records:
        print(f"\n{row['acquisition_date'][:4]} scene:\ndate: {row['acquisition_date']}\nscene: {row['scene_id']}\nlocal file: {row['local_file']}\nsize: {row['file_size_bytes']}\nSHA256: {row['sha256']}")
    print('\nBoth historical ALOS Level 1.1 products downloaded successfully.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
