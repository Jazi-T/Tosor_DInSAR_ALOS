#!/usr/bin/env python3
"""Discover ASF collections and search Tosor scene metadata only.

CMR API: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
Point queries use longitude,latitude. No product downloads, pairs or baselines.
Raw JSON retains complete collection and granule records and query provenance.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CMR = 'https://cmr.earthdata.nasa.gov/search'
LATITUDE, LONGITUDE = 41.961825, 77.373620
POINT = f'{LONGITUDE},{LATITUDE}'
DATASETS = [
    ('ALOS', 'PALSAR', 2009, 2010, 'alos_palsar_2009_2010'),
    ('ALOS-2', 'PALSAR-2', 2023, 2025, 'alos2_palsar2_2023_2025'),
]
COLUMNS = ['mission', 'sensor', 'scene_name', 'granule_id', 'acquisition_start',
           'acquisition_end', 'processing_level', 'path', 'frame', 'absolute_orbit_number',
           'orbit_direction', 'beam_mode', 'polarization', 'look_direction',
           'collection_concept_id', 'granule_concept_id', 'download_url']


def search(session, resource, params):
    """Harvest every page using CMR Search After, deduplicating concept IDs."""
    results, seen, tokens = [], set(), set()
    headers = {}
    while True:
        response = session.get(f'{CMR}/{resource}.umm_json',
                               params={**params, 'page_size': 2000}, headers=headers, timeout=90)
        if not response.ok:
            raise RuntimeError(f'CMR HTTP {response.status_code}: {response.text}')
        payload = response.json()
        if not isinstance(payload.get('items'), list):
            raise ValueError(f'Unexpected CMR response: {payload}')
        for item in payload['items']:
            concept = item['meta']['concept-id']
            if concept not in seen:
                seen.add(concept)
                results.append(item)
        token = response.headers.get('CMR-Search-After')
        if not token or not payload['items']:
            if len(results) < payload.get('hits', 0):
                raise RuntimeError('CMR pagination ended before all reported hits were retrieved.')
            return results
        if token in tokens:
            raise RuntimeError('CMR repeated a pagination token.')
        tokens.add(token)
        headers['CMR-Search-After'] = token


def canonical(value):
    return re.sub(r'[^A-Z0-9]', '', str(value).upper())


def matching_collections(collections, mission, sensor):
    selected = []
    for item in collections:
        umm = item['umm']
        matches = any(canonical(p.get('ShortName', '')) == canonical(mission) and
                      any(canonical(i.get('ShortName', '')) == canonical(sensor)
                          for i in p.get('Instruments', []))
                      for p in umm.get('Platforms', []))
        if not matches:
            continue
        # ALOS collections sometimes label ProcessingLevel.Id as just "1".
        # Require explicit 1.1 evidence in the collection title/short name.
        description = ' '.join(str(umm.get(k, '')) for k in ('ShortName', 'EntryTitle'))
        level = umm.get('ProcessingLevel', {}).get('Id', '')
        if mission == 'ALOS' and not (
            str(level).upper() in ('1.1', 'L1.1') or
            re.search(r'(?:LEVEL\s*|L)1\.1(?!\d)', description, re.I)
        ):
            continue
        selected.append(item)
    if not selected:
        raise RuntimeError(f'No matching current ASF collection found for {mission}/{sensor}.')
    return selected


def attributes(umm):
    result = {}
    for attr in umm.get('AdditionalAttributes', []):
        values = attr.get('Values', [])
        result[canonical(attr.get('Name', ''))] = (
            ';'.join(str(v) for v in values) if isinstance(values, list) else str(values))
    return result


def extract(item, collection, mission, sensor):
    umm, meta = item['umm'], item['meta']
    attrs = attributes(umm)
    def get(*keys):
        return next((attrs[canonical(k)] for k in keys if attrs.get(canonical(k))), '')
    temporal = umm.get('TemporalExtent', {})
    dates = temporal.get('RangeDateTime', {})
    identifiers = umm.get('DataGranule', {}).get('Identifiers', [])
    granule = umm.get('GranuleUR', meta.get('native-id', ''))
    scene = next((i['Identifier'] for i in identifiers
                  if i.get('IdentifierType') == 'ProducerGranuleId'), '')
    orbit = next((d['OrbitNumber'] for d in umm.get('OrbitCalculatedSpatialDomains', [])
                  if 'OrbitNumber' in d), '')
    urls = [u.get('URL', '') for u in umm.get('RelatedUrls', [])
            if u.get('Type') == 'GET DATA' and u.get('URL')]
    level = get('PROCESSING_TYPE', 'PROCESSING_LEVEL', 'PRODUCT_LEVEL')
    if not level:
        level = collection['umm'].get('ProcessingLevel', {}).get('Id', '')
    if mission == 'ALOS' and level in ('1', 'L1', ''):
        level = 'L1.1'  # Explicitly verified at collection discovery.
    platforms = umm.get('Platforms', [])
    instruments = [i.get('ShortName', '') for p in platforms for i in p.get('Instruments', [])]
    return dict(zip(COLUMNS, [
        mission, ';'.join(instruments) or sensor,
        scene or get('SCENE_NAME') or granule, granule,
        dates.get('BeginningDateTime', temporal.get('SingleDateTime', '')) or get('ACQUISITION_DATE'),
        dates.get('EndingDateTime', ''), level,
        get('PATH_NUMBER', 'PATH', 'TRACK_NUMBER', 'TRACK'),
        get('FRAME_NUMBER', 'FRAME'), orbit or get('ORBIT_NUMBER', 'ABSOLUTE_ORBIT_NUMBER'),
        get('ASCENDING_DESCENDING', 'FLIGHT_DIRECTION', 'ORBIT_DIRECTION'),
        get('BEAM_MODE', 'BEAM_MODE_TYPE'), get('POLARIZATION', 'POLARISATION'),
        get('LOOK_DIRECTION'), meta.get('collection-concept-id', collection['meta']['concept-id']),
        meta.get('concept-id', ''), ';'.join(urls),
    ]))


def main():
    table_dir, metadata_dir = ROOT / 'output/tables', ROOT / 'data/metadata'
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({'Client-Id': 'tosor-catalog-search', 'User-Agent': 'Tosor-Catalog/1.0'})
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        collections = search(session, 'collections', {'provider': 'ASF'})
        for mission, sensor, year1, year2, stem in DATASETS:
            selected = matching_collections(collections, mission, sensor)
            rows, records, queries = [], [], []
            for collection in selected:
                concept = collection['meta']['concept-id']
                print(f"Resolved {concept}: {collection['umm'].get('EntryTitle', '')}")
                params = {'collection_concept_id': concept, 'point': POINT,
                          'temporal': f'{year1}-01-01T00:00:00Z,{year2}-12-31T23:59:59Z'}
                items = search(session, 'granules', params)
                queries.append(params)
                records.extend(items)
                rows.extend(extract(item, collection, mission, sensor) for item in items)
            df = pd.DataFrame(rows, columns=COLUMNS)
            if not df.empty:
                df = df.drop_duplicates('granule_concept_id').sort_values('acquisition_start', kind='stable')
            raw = {'retrieved_at': datetime.now(timezone.utc).isoformat(),
                   'queries': queries, 'collections': selected, 'items': records}
            (metadata_dir / f'{stem}_cmr.json').write_text(
                json.dumps(raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            df.to_csv(table_dir / f'{stem}_inventory.csv', index=False)
            print(f'\n{mission}/{sensor} {year1}–{year2}\nTotal scenes found: {len(df)}')
            print(df[['acquisition_start', 'granule_id', 'processing_level', 'path', 'frame',
                      'beam_mode', 'polarization', 'orbit_direction']].to_string(index=False))
    print('\nMetadata search complete. No SAR products downloaded.')


if __name__ == '__main__':
    main()
