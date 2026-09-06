#!/usr/bin/env python3
"""Evaluate cross-year SAR pairs using catalog geometry only.

Season distance compares month/day on the common leap-year 2000 calendar
(366-day circle), avoiding a spurious shift between leap and non-leap years.
Priority score is an ordinal rank within each mission/year combination:
compatible geometry and requested years are prerequisites; smaller season
and repeat-cycle differences, then shorter temporal baseline rank first.
No downloads, perpendicular-baseline queries, or baseline estimates.
"""
from pathlib import Path
from itertools import product
import importlib.util
import json
import math
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'output/tables'
PERIODS = [('ALOS', 2009, 2010, 46, 'alos_2009_2010'),
           ('ALOS-2', 2023, 2024, 14, 'alos2_2023_2024'),
           ('ALOS-2', 2024, 2025, 14, 'alos2_2024_2025'),
           ('ALOS-2', 2023, 2025, 14, 'alos2_2023_2025')]
GEOMETRY = ['mission', 'path', 'frame', 'orbit_direction', 'beam_mode',
            'polarization', 'look_direction']
COLUMNS = ['mission', 'scene_1', 'date_1', 'scene_2', 'date_2', 'year_1', 'year_2']
for field in ['path', 'frame', 'orbit_direction', 'beam', 'polarization', 'look_direction']:
    COLUMNS.extend([f'{field}_1', f'{field}_2'])
COLUMNS += ['temporal_baseline_days', 'season_difference_days', 'season_match_class',
            'repeat_cycle_days', 'nearest_repeat_cycle_multiple', 'expected_repeat_cycle_days',
            'repeat_cycle_difference_days', 'pair_status', 'metadata_missing',
            'metadata_mismatches', 'priority_score']


def normalize(value, field):
    if pd.isna(value) or str(value).strip().upper() in ('', 'NAN', 'NONE', 'UNKNOWN', 'NOT PROVIDED'):
        return ''
    value = str(value).strip().upper()
    if field in ('path', 'frame'):
        try:
            return str(int(value)) if float(value).is_integer() else value
        except ValueError:
            return value
    if field == 'polarization':
        return ';'.join(sorted(set(re.split(r'[+;,/\s]+', value))))
    if field == 'look_direction':
        return {'R': 'RIGHT', 'L': 'LEFT'}.get(value, value)
    return value


def load_inventory(mission, stem):
    df = pd.read_csv(TABLES / f'{stem}_inventory.csv', dtype=str, keep_default_na=False)
    # Reuse the catalog extractor only to recover missing fields from actual UMM.
    spec = importlib.util.spec_from_file_location('catalog', ROOT / 'scripts/01_search_alos_scenes.py')
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)
    raw = json.loads((ROOT / 'data/metadata' / f'{stem}_cmr.json').read_text())
    collections = {c['meta']['concept-id']: c for c in raw['collections']}
    recovered = {}
    for item in raw['items']:
        collection = collections[item['meta']['collection-concept-id']]
        row = catalog.extract(item, collection, mission, 'PALSAR' if mission == 'ALOS' else 'PALSAR-2')
        recovered[row['granule_id']] = row
    for index, row in df.iterrows():
        source = recovered.get(row['granule_id'], {})
        for field in GEOMETRY:
            value = normalize(row.get(field, ''), field)
            if not value:
                value = normalize(source.get(field, ''), field)
            df.loc[index, field] = value
    if not df['mission'].eq(mission).all():
        raise ValueError(f'{stem}: unexpected or missing mission')
    df['date'] = pd.to_datetime(df['acquisition_start'], utc=True, format='mixed', errors='raise')
    if df['date'].isna().any() or df['granule_id'].eq('').any():
        raise ValueError(f'{stem}: acquisition dates and scene identifiers are required')
    return df.drop_duplicates('granule_id')


def seasonal_difference(date1, date2):
    first = date1.replace(year=2000).dayofyear
    second = date2.replace(year=2000).dayofyear
    diff = abs(first - second)
    return min(diff, 366 - diff)


def evaluate(first, second, cycle):
    missing, mismatches = [], []
    for field in GEOMETRY:
        a, b = first[field], second[field]
        if not a or not b:
            missing.append(field)
        elif a != b:
            mismatches.append(field)
    status = 'REJECT_GEOMETRY' if mismatches else 'INCOMPLETE_METADATA' if missing else 'COMPATIBLE'
    date1, date2 = first['date'], second['date']
    days = (date2 - date1).total_seconds() / 86400
    multiple = max(1, math.floor(days / cycle + 0.5))
    season = seasonal_difference(date1, date2)
    season_class = ('excellent seasonal match' if season <= 14 else 'good seasonal match' if season <= 30
                    else 'acceptable seasonal match' if season <= 45 else 'poor seasonal match')
    result = {'mission': first['mission'], 'scene_1': first['granule_id'], 'date_1': date1.isoformat(),
              'scene_2': second['granule_id'], 'date_2': date2.isoformat(),
              'year_1': date1.year, 'year_2': date2.year,
              'temporal_baseline_days': days, 'season_difference_days': season,
              'season_match_class': season_class, 'repeat_cycle_days': cycle,
              'nearest_repeat_cycle_multiple': multiple, 'expected_repeat_cycle_days': multiple * cycle,
              'repeat_cycle_difference_days': abs(days - multiple * cycle),
              'pair_status': status, 'metadata_missing': ';'.join(missing),
              'metadata_mismatches': ';'.join(mismatches), 'priority_score': None}
    for field in GEOMETRY[1:]:
        name = 'beam' if field == 'beam_mode' else field
        result[f'{name}_1'], result[f'{name}_2'] = first[field], second[field]
    return result


def rank(evaluated):
    compatible = evaluated[evaluated['pair_status'] == 'COMPATIBLE'].sort_values(
        ['season_difference_days', 'repeat_cycle_difference_days', 'temporal_baseline_days',
         'scene_1', 'scene_2'], kind='stable')
    # Dense ordinal scores preserve ties on all scientific ranking criteria.
    keys = list(zip(compatible['season_difference_days'], compatible['repeat_cycle_difference_days'],
                    compatible['temporal_baseline_days']))
    scores = {key: len(set(keys)) - i for i, key in enumerate(sorted(set(keys)))}
    for index, key in zip(compatible.index, keys):
        evaluated.loc[index, 'priority_score'] = scores[key]
    return evaluated.loc[compatible.index].copy()


def main():
    inventories = {
        'ALOS': load_inventory('ALOS', 'alos_palsar_2009_2010'),
        'ALOS-2': load_inventory('ALOS-2', 'alos2_palsar2_2023_2025'),
    }
    all_results = []
    display = ['date_1', 'date_2', 'scene_1', 'scene_2', 'path_1', 'frame_1', 'beam_1',
               'polarization_1', 'orbit_direction_1', 'temporal_baseline_days',
               'season_difference_days', 'repeat_cycle_difference_days', 'priority_score']
    top = []
    for mission, year1, year2, cycle, stem in PERIODS:
        df = inventories[mission]
        first = df[df['date'].dt.year == year1]
        second = df[df['date'].dt.year == year2]
        rows = [evaluate(a, b, cycle) for (_, a), (_, b) in product(first.iterrows(), second.iterrows())]
        evaluated = pd.DataFrame(rows, columns=COLUMNS)
        compatible = rank(evaluated)
        compatible.to_csv(TABLES / f'{stem}_compatible_pairs.csv', index=False)
        all_results.append(evaluated)
        title = f'{"ALOS/PALSAR" if mission == "ALOS" else "ALOS-2"} {year1}–{year2}'
        print('\n' + '=' * 40 + '\n' + title + '\n' + '=' * 40)
        print(f'total possible pairs: {len(evaluated)}')
        for label, status in [('compatible pairs', 'COMPATIBLE'), ('rejected geometry pairs', 'REJECT_GEOMETRY'),
                              ('incomplete metadata pairs', 'INCOMPLETE_METADATA')]:
            print(f'{label}: {(evaluated["pair_status"] == status).sum()}')
        print(compatible[display].to_string(index=False) if len(compatible) else 'No compatible pairs.')
        top.append((title, compatible.head(3)))
    pd.concat(all_results, ignore_index=True).to_csv(TABLES / 'all_dinsar_pair_evaluation.csv', index=False)
    for title, pairs in top:
        print(f'\nTOP 3 — {title}')
        print(pairs[display].to_string(index=False) if len(pairs) else 'No compatible pairs.')
    print('\nPriority scores are ordinal within each period, not coherence probabilities.')
    print('No SAR downloads or perpendicular-baseline calculations performed.')


if __name__ == '__main__':
    main()
