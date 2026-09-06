#!/usr/bin/env python3
"""Review inventory geometry and shortlist preliminary Tosor pairs offline.

HIGH: verified geometry, July–September on both dates, season <=14 days,
and repeat-cycle residual <=1 day. MEDIUM: verified geometry, both dates
July–September, season <=45 days and cycle residual <=3 days. Otherwise LOW.
Unverified/conflicting pairs are never shortlisted. Within quality tiers:
prefer summer, smaller seasonal distance, August-to-August, then cycle
residual and temporal separation. August preference breaks seasonal ties.
These are transparent screening rules, not measured coherence or Bperp.
"""
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'output/tables'
spec = importlib.util.spec_from_file_location('pairs', ROOT / 'scripts/02_find_dinsar_pairs.py')
pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairs)
SETTINGS = [('ALOS', 'alos_2009_2010', 'alos_palsar_2009_2010', 2009, 2010, 46, 2),
            ('ALOS-2', 'alos2_2023_2024', 'alos2_palsar2_2023_2025', 2023, 2024, 14, 3)]
DISPLAY = ['mission', 'scene_1', 'date_1', 'scene_2', 'date_2', 'path', 'frame',
           'beam_mode', 'polarization', 'orbit_direction', 'look_direction',
           'temporal_baseline_days', 'season_difference_days', 'repeat_cycle_difference_days',
           'quality_flag', 'selection_reason']


def review(candidate, inventory, mission, year1, year2, cycle):
    conflicts = []
    scenes = []
    for side in (1, 2):
        matches = inventory[inventory['granule_id'] == candidate[f'scene_{side}']]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve {candidate[f'scene_{side}']} in inventory")
        scene = matches.iloc[0]
        scenes.append(scene)
        if pd.to_datetime(candidate[f'date_{side}'], utc=True) != scene['date']:
            conflicts.append(f'date_{side}: pair table differs from inventory')
        for field in pairs.GEOMETRY:
            key = 'mission' if field == 'mission' else f"{'beam' if field == 'beam_mode' else field}_{side}"
            value = pairs.normalize(candidate.get(key, ''), field)
            if value and value != scene[field]:
                conflicts.append(f'{key}: pair table differs from inventory')
    first, second = scenes
    result = pairs.evaluate(first, second, cycle)
    if first['mission'] != mission or second['mission'] != mission or (
        first['date'].year, second['date'].year) != (year1, year2):
        conflicts.append('mission or target year combination differs')
    valid = result['pair_status'] == 'COMPATIBLE' and not conflicts
    summer = first['date'].month in (7, 8, 9) and second['date'].month in (7, 8, 9)
    august = first['date'].month == second['date'].month == 8
    season, residual = result['season_difference_days'], result['repeat_cycle_difference_days']
    quality = 'LOW_PRIORITY'
    if valid and summer and season <=14 and residual <=1:
        quality = 'HIGH_PRIORITY'
    elif valid and summer and season <=45 and residual <=3:
        quality = 'MEDIUM_PRIORITY'
    reasons = [f"Inventory geometry {'verified' if valid else 'not verified'}",
               f"{'both' if summer else 'not both'} acquisitions in July–September",
               f'{season}-day seasonal difference', f'{residual:.6f}-day repeat-cycle residual']
    if august:
        reasons.append('August-to-August comparison')
    if mission == 'ALOS':
        reasons.append('requested historical comparison; year-long separation is accepted')
    if conflicts:
        reasons.extend(conflicts)
    if result['metadata_missing']:
        reasons.append('missing: ' + result['metadata_missing'])
    if result['metadata_mismatches']:
        reasons.append('geometry mismatches: ' + result['metadata_mismatches'])
    reasons.append('perpendicular baseline and coherence remain untested')
    result.update(quality_flag=quality, selection_reason='; '.join(reasons),
                  metadata_conflicts='; '.join(conflicts), geometry_verified=valid,
                  both_summer=summer, both_august=august, selected=False)
    for field in pairs.GEOMETRY[1:]:
        # Shared fields are populated only when inventory values actually agree.
        result[field] = first[field] if first[field] and first[field] == second[field] else ''
    return result


def shortlist(reviewed, count):
    eligible = reviewed[reviewed['geometry_verified']].copy()
    eligible['quality_order'] = eligible['quality_flag'].map(
        {'HIGH_PRIORITY': 0, 'MEDIUM_PRIORITY': 1, 'LOW_PRIORITY': 2})
    return eligible.sort_values(
        ['quality_order', 'both_summer', 'season_difference_days', 'both_august',
         'repeat_cycle_difference_days', 'temporal_baseline_days', 'scene_1', 'scene_2'],
        ascending=[True, False, True, False, True, True, True, True], kind='stable'
    ).head(count).drop(columns='quality_order')


def main():
    reviews, selections = [], []
    for mission, stem, inventory_stem, year1, year2, cycle, count in SETTINGS:
        candidates = pd.read_csv(TABLES / f'{stem}_compatible_pairs.csv', dtype=str, keep_default_na=False)
        inventory = pairs.load_inventory(mission, inventory_stem)
        reviewed = pd.DataFrame([review(row, inventory, mission, year1, year2, cycle)
                                 for _, row in candidates.iterrows()])
        print(f'\nALL REVIEWED {mission} CANDIDATES')
        print(reviewed.to_string(index=False))
        if mission == 'ALOS':
            special = reviewed[(reviewed['date_1'].str[:10] == '2009-08-22') &
                               (reviewed['date_2'].str[:10] == '2010-08-25')]
            print('\nSPECIAL HISTORICAL CANDIDATE')
            print(special.to_string(index=False) if not special.empty else 'Not present among input candidates.')
        selected = shortlist(reviewed, count)
        reviewed.loc[selected.index, 'selected'] = True
        selected['selected'] = True
        if len(selected) < count:
            print(f'Only {len(selected)} verified pairs available; requested {count}.')
        reviews.append(reviewed)
        selections.append(selected)
    pd.concat(reviews, ignore_index=True).to_csv(TABLES / 'dinsar_pair_quality_review.csv', index=False)
    final = pd.concat(selections, ignore_index=True)
    final.to_csv(TABLES / 'final_dinsar_pair_shortlist.csv', index=False)
    for mission, _, _, year1, year2, _, _ in SETTINGS:
        print('\n' + '='*60)
        print(f"FINAL {'ALOS/PALSAR' if mission == 'ALOS' else 'ALOS-2/PALSAR-2'} {year1}-{year2} SHORTLIST")
        print('='*60)
        print(final[final['mission'] == mission][DISPLAY].to_string(index=False))
    unique = {}
    for _, row in final.iterrows():
        for side in (1, 2):
            unique[(row['mission'], row[f'scene_{side}'])] = row[f'date_{side}']
    print('\nUNIQUE SAR SCENES REQUIRED')
    for (mission, scene), date in sorted(unique.items()):
        print(f'{mission} | {scene} | {date}')
    for mission in ('ALOS', 'ALOS-2'):
        print(f'Number of unique {mission} scenes: {sum(m == mission for m, _ in unique)}')
    print(f'Total unique SAR products that would need to be downloaded: {len(unique)}')
    print('No SAR downloads or baseline API queries performed.')


if __name__ == '__main__':
    main()
