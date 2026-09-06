#!/usr/bin/env python3
"""Discover ALOS-2 collections across CMR providers and query Tosor in 2025.

Only metadata endpoints are accessed. Product phase suitability is conservative;
conflicting or absent evidence is UNCERTAIN. No inference from scene IDs.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
TABLES, METADATA = ROOT / 'output/tables', ROOT / 'data/metadata'
spec = importlib.util.spec_from_file_location('catalog', ROOT / 'scripts/01_search_alos_scenes.py')
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
POINT = '77.373620,41.961825'
TEMPORAL = '2025-01-01T00:00:00Z,2025-12-31T23:59:59Z'
COL_COLUMNS = ['collection_concept_id', 'short_name', 'entry_title', 'version', 'provider',
               'platform', 'instrument', 'processing_level', 'collection_temporal_extent',
               'relevance_evidence', 'granules_found_2025']
GRAN_COLUMNS = ['collection_short_name', 'collection_concept_id', 'mission', 'platform', 'sensor',
                'granule_id', 'native_id', 'granule_concept_id', 'acquisition_start', 'acquisition_end',
                'processing_level', 'product_type', 'path', 'track', 'frame', 'orbit_number',
                'orbit_direction', 'beam_mode', 'polarization', 'look_direction', 'download_url',
                'insar_suitability', 'suitability_evidence']


def platforms(umm):
    ps = umm.get('Platforms', [])
    return (';'.join(p.get('ShortName', '') for p in ps),
            ';'.join(i.get('ShortName', '') for p in ps for i in p.get('Instruments', [])))


def relevance(umm):
    platform, instrument = platforms(umm)
    text = ' '.join([platform, instrument, umm.get('ShortName', ''), umm.get('EntryTitle', ''),
                     umm.get('Abstract', '')])
    if catalog.canonical(instrument) == 'CIRC':
        return ''  # Returned metadata identifies the Compact Infrared Camera, not SAR.
    found = re.findall(r'\b(?:ALOS[- _]?2|PALSAR[- _]?2)\b', text, re.I)
    # Deliberately inclusive: abstracts mentioning ALOS-2 are also searched,
    # including derived/non-phase products, which are assessed separately.
    return ';'.join(sorted(set(found)))


def suitability(level, text):
    positive = str(level).strip().upper() in ('1.1', 'L1.1', 'LEVEL 1.1') or bool(re.search(
        r'\b(?:Level\s*|L)1\.1\b|\bSLC\b|single[- ]look complex|\bcomplex (?:SAR|data|image|product)|phase[- ]preserv|retains? phase', text, re.I))
    negative = bool(re.search(r'amplitude[- ]only|intensity[- ]only|\bdetected\b|without phase|no phase|'
                             r'phase (?:information )?(?:is )?(?:lost|removed)|backscatter mosaic', text, re.I))
    if positive and not negative:
        return 'LIKELY_INSAR_SUITABLE'
    if negative and not positive:
        return 'NOT_INSAR_SUITABLE'
    return 'UNCERTAIN'


def extract(item, collection):
    umm, meta, cu = item['umm'], item['meta'], collection['umm']
    row = catalog.extract(item, collection, '', '')
    attrs = catalog.attributes(umm)
    def attr(*keys):
        return next((attrs[catalog.canonical(k)] for k in keys if attrs.get(catalog.canonical(k))), '')
    platform, sensor = platforms(umm)
    if not platform or not sensor:
        cp, cs = platforms(cu)
        platform, sensor = platform or cp, sensor or cs
    level = attr('PROCESSING_TYPE', 'PROCESSING_LEVEL', 'PRODUCT_LEVEL') or cu.get('ProcessingLevel', {}).get('Id', '')
    product_type = attr('PRODUCT_TYPE', 'GRANULE_TYPE', 'PROCESSING_TYPE_DISPLAY')
    # Prefer product-specific descriptions, avoiding arbitrary abstract mentions of SLC inputs.
    descriptions = [attr('PROCESSING_DESCRIPTION'), attr('PRODUCT_DESCRIPTION'), product_type]
    if not any(descriptions):
        descriptions = [cu.get('EntryTitle', ''), cu.get('ProcessingLevel', {}).get('ProcessingLevelDescription', '')]
    evidence = f'processing_level={level}; ' + '; '.join(filter(None, descriptions))
    return dict(zip(GRAN_COLUMNS, [
        cu.get('ShortName', ''), meta.get('collection-concept-id', collection['meta']['concept-id']),
        platform, platform, sensor, row['granule_id'], meta.get('native-id', ''), meta.get('concept-id', ''),
        row['acquisition_start'], row['acquisition_end'], level, product_type, row['path'],
        attr('TRACK_NUMBER', 'TRACK', 'PATH_NUMBER', 'PATH'), row['frame'], row['absolute_orbit_number'],
        row['orbit_direction'], row['beam_mode'], row['polarization'], row['look_direction'],
        row['download_url'], suitability(level, ' '.join(descriptions)), evidence,
    ]))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    queries = [{'keyword': term} for term in ('ALOS-2', 'PALSAR-2', 'PALSAR2', 'ALOS2')]
    queries += [{'platform': 'ALOS-2'}, {'instrument': 'PALSAR-2'}, {'provider': 'ASF'}]
    discovered, query_records = {}, []
    with requests.Session() as session:
        session.headers.update({'Client-Id': 'tosor-alos2-discovery', 'User-Agent': 'Tosor-CMR/1.0'})
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504])))
        for params in queries:
            items = catalog.search(session, 'collections', params)
            print(f'Discovery {params}: {len(items)} collections', flush=True)
            query_records.append({'params': params, 'concept_ids': [i['meta']['concept-id'] for i in items]})
            discovered.update({i['meta']['concept-id']: i for i in items})
        relevant = [i for i in discovered.values() if relevance(i['umm'])]
        relevant.sort(key=lambda i: i['meta']['concept-id'])
        raw_discovery = {'retrieved_at': stamp, 'queries': query_records, 'items': list(discovered.values()),
                         'relevant_concept_ids': [i['meta']['concept-id'] for i in relevant]}
        (METADATA / 'alos2_palsar2_collection_discovery.json').write_text(json.dumps(raw_discovery, indent=2)+'\n')
        print('\n' + '=' * 40 + '\nALOS-2 / PALSAR-2 COLLECTION DISCOVERY\n' + '=' * 40)
        for item in relevant:
            print(item['umm'].get('ShortName'), item['meta']['concept-id'],
                  item['meta'].get('provider-id'), item['umm'].get('EntryTitle'), sep=' | ')
        rows, collection_rows, searches = [], [], []
        print('\n' + '=' * 40 + '\nTOSOR 2025 SEARCH\n' + '=' * 40)
        for collection in relevant:
            umm, meta = collection['umm'], collection['meta']
            params = {'collection_concept_id': meta['concept-id'], 'point': POINT, 'temporal': TEMPORAL}
            items = catalog.search(session, 'granules', params)
            print(f"collection: {umm.get('ShortName')}\ngranules found over Tosor in 2025: {len(items)}", flush=True)
            searches.append({'params': params, 'count': len(items), 'items': items})
            rows.extend(extract(i, collection) for i in items)
            platform, instrument = platforms(umm)
            collection_rows.append(dict(zip(COL_COLUMNS, [meta['concept-id'], umm.get('ShortName', ''),
                umm.get('EntryTitle', ''), umm.get('Version', ''), meta.get('provider-id', ''), platform,
                instrument, umm.get('ProcessingLevel', {}).get('Id', ''),
                json.dumps(umm.get('TemporalExtents', [])), relevance(umm), len(items)])))
    df = pd.DataFrame(rows, columns=GRAN_COLUMNS).drop_duplicates('granule_concept_id')
    if not df.empty:
        df = df.sort_values('acquisition_start')
    candidates = df[df['insar_suitability'] != 'NOT_INSAR_SUITABLE']
    pd.DataFrame(collection_rows, columns=COL_COLUMNS).to_csv(TABLES / 'alos2_palsar2_collections_discovered.csv', index=False)
    df.to_csv(TABLES / 'alos2_palsar2_2025_all_collections.csv', index=False)
    candidates.to_csv(TABLES / 'alos2_palsar2_2025_insar_candidates.csv', index=False)
    (METADATA / 'alos2_palsar2_2025_all_collections_cmr.json').write_text(json.dumps(
        {'retrieved_at': stamp, 'searches': searches}, indent=2)+'\n')
    print(df[['acquisition_start', 'granule_id', 'collection_short_name', 'processing_level',
              'beam_mode', 'polarization', 'orbit_direction', 'insar_suitability']].to_string(index=False))
    print(f'Total ALOS-2/PALSAR-2 collections checked: {len(relevant)}')
    print(f'Total 2025 scenes over Tosor: {len(df)}')
    print(f"Likely InSAR-suitable scenes: {(df['insar_suitability'] == 'LIKELY_INSAR_SUITABLE').sum()}")
    print(f"Uncertain scenes: {(df['insar_suitability'] == 'UNCERTAIN').sum()}")
    if df.empty:
        print('No ALOS-2/PALSAR-2 2025 scenes over the Tosor point were found in the searched NASA Earthdata CMR/ASF collections.')
    print('JAXA G-Portal availability requires a separate check. No SAR data downloaded.')


if __name__ == '__main__':
    main()
