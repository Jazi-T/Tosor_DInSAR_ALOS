#!/usr/bin/env python3
"""Prepare a minimal scene manifest from ranked pairs without downloading data.

Keep every ranked ALOS pair; select up to two ALOS-2 pairs per target
comparison by descending priority score, preserving input order for ties.
Perpendicular baseline is not used to select or discard pairs.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / 'output' / 'tables'
YEAR_PAIRS = ((2023, 2024), (2024, 2025), (2023, 2025))
MANIFEST_COLUMNS = ['mission', 'scene_id', 'acquisition_date', 'year', 'path',
                    'frame', 'beam', 'polarization', 'orbit_direction', 'required_by_pairs']


def read_pairs(path, mission):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'scene_1', 'scene_2', 'date_1', 'date_2', 'priority_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df['mission'] = mission
    for side in (1, 2):
        if df[f'scene_{side}'].str.strip().eq('').any():
            raise ValueError(f'{path}: empty scene identifier')
        dates = pd.to_datetime(df[f'date_{side}'], utc=True, format='mixed', errors='raise')
        if dates.isna().any():
            raise ValueError(f'{path}: missing acquisition date')
        df[f'year_{side}'] = dates.dt.year
    df['priority_score'] = pd.to_numeric(df['priority_score'], errors='raise')
    df['year_combination'] = df.apply(
        lambda row: f"{min(row['year_1'], row['year_2'])}–{max(row['year_1'], row['year_2'])}",
        axis=1,
    ) if not df.empty else pd.Series(dtype=str)
    return df


def select_pairs(alos, alos2):
    selected = [alos.copy()]
    for year1, year2 in YEAR_PAIRS:
        group = alos2[alos2['year_combination'] == f'{year1}–{year2}']
        selected.append(group.sort_values('priority_score', ascending=False, kind='stable').head(2))
    shortlist = pd.concat(selected, ignore_index=True)
    shortlist.insert(0, 'pair_id', [f'PAIR_{i:03d}' for i in range(1, len(shortlist) + 1)])
    return shortlist


def build_manifest(shortlist):
    scenes = {}
    for _, pair in shortlist.iterrows():
        for side in (1, 2):
            key = (pair['mission'], pair[f'scene_{side}'].strip())
            record = {
                'mission': key[0], 'scene_id': key[1],
                'acquisition_date': pd.to_datetime(pair[f'date_{side}'], utc=True).isoformat(),
                'year': int(pair[f'year_{side}']),
                **{dest: pair.get(f'{src}_{side}', '') for dest, src in
                   [('path', 'path'), ('frame', 'frame'), ('beam', 'beam'),
                    ('polarization', 'polarization'), ('orbit_direction', 'direction')]},
            }
            if key not in scenes:
                scenes[key] = {**record, 'required_by_pairs': []}
            else:
                for field, value in record.items():
                    old = scenes[key][field]
                    if old != '' and value != '' and old != value:
                        raise ValueError(f'Conflicting {field} metadata for {key}: {old!r} vs {value!r}')
                    if old == '':
                        scenes[key][field] = value
            if pair['pair_id'] not in scenes[key]['required_by_pairs']:
                scenes[key]['required_by_pairs'].append(pair['pair_id'])
    rows = [{**record, 'required_by_pairs': ';'.join(record['required_by_pairs'])}
            for record in scenes.values()]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def print_candidates(title, pairs):
    print(f'\n{title}')
    if pairs.empty:
        print('No ranked candidates available.')
        return
    columns = ['pair_id', 'scene_1', 'date_1', 'scene_2', 'date_2', 'priority_score']
    print(pairs[columns].to_string(index=False))


def main():
    alos = read_pairs(TABLE_DIR / 'alos_ranked_dinsar_pairs.csv', 'ALOS')
    alos2 = read_pairs(TABLE_DIR / 'alos2_ranked_dinsar_pairs.csv', 'ALOS-2')
    shortlist = select_pairs(alos, alos2)
    manifest = build_manifest(shortlist)
    shortlist.to_csv(TABLE_DIR / 'final_pair_shortlist.csv', index=False)
    manifest.to_csv(TABLE_DIR / 'sar_download_manifest.csv', index=False)

    print_candidates('ALOS FINAL CANDIDATES', shortlist[shortlist['mission'] == 'ALOS'])
    for year1, year2 in YEAR_PAIRS:
        print_candidates(f'ALOS-2 {year1}–{year2} FINAL CANDIDATES', shortlist[
            (shortlist['mission'] == 'ALOS-2') &
            (shortlist['year_combination'] == f'{year1}–{year2}')])
    print('\nUNIQUE SCENES REQUIRED')
    print(manifest.to_string(index=False))
    for mission in ('ALOS', 'ALOS-2'):
        print(f"Total unique {mission} scenes: {(manifest['mission'] == mission).sum()}")
    print('\nSaved final_pair_shortlist.csv and sar_download_manifest.csv in output/tables/.')
    print('No perpendicular-baseline filtering applied. No SAR data downloaded.')


if __name__ == '__main__':
    main()
