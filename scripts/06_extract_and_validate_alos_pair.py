#!/usr/bin/env python3
"""Verify manifest SHA256, safely extract ZIPs, and inspect CEOS structure.

Validation compares supplied summary metadata and inventory, not a full
binary CEOS leader/orbit interpretation. Original archives remain untouched.
"""
from pathlib import Path, PurePosixPath
import hashlib
import re
import shutil
import stat
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'output/tables'
DATES = ('2009-08-22', '2010-08-25')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_members(archive, destination):
    members, seen = [], set()
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        require(not name.is_absolute() and '..' not in name.parts and
                '\\' not in info.filename and ':' not in info.filename,
                f'Unsafe ZIP path: {info.filename}')
        mode = info.external_attr >> 16
        require(stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                f'Unsupported ZIP special file: {info.filename}')
        target = destination.joinpath(*name.parts)
        require(target.resolve().is_relative_to(destination.resolve()), 'ZIP path escapes destination')
        require(target not in seen, f'Duplicate ZIP path: {info.filename}')
        seen.add(target)
        members.append((info, target))
    return members


def file_type(path):
    return next((prefix for prefix in ('LED', 'IMG', 'VOL', 'TRL')
                 if path.name.upper().startswith(prefix+'-')), 'ANCILLARY')


def main():
    manifest = pd.read_csv(TABLES / 'alos_2009_2010_download_manifest.csv', dtype=str)
    inventory = pd.read_csv(TABLES / 'alos_palsar_2009_2010_inventory.csv', dtype=str)
    prepared = []
    # Verify BOTH archives before extracting either one.
    for date in DATES:
        rows = manifest[manifest.acquisition_date == date]
        require(len(rows)==1, f'Expected one manifest row for {date}')
        row = rows.iloc[0]
        folder = ROOT / 'data/raw/ALOS_2009_2010' / date
        archive = (ROOT / row.local_file).resolve()
        require(archive.parent == folder.resolve() and archive.suffix.lower()=='.zip', 'Manifest archive outside date folder')
        require(archive.is_file() and archive.stat().st_size>0, f'Missing/empty archive: {archive}')
        require(archive.stat().st_size == int(row.file_size_bytes), f'Size mismatch: {archive}')
        print(f'Checking SHA256: {archive}', flush=True)
        require(checksum(archive)==row.sha256.lower(), f'SHA256 mismatch: {archive}')
        matches = inventory[inventory.granule_id == row.scene_id]
        require(len(matches)==1, 'Scene not uniquely found in inventory')
        destination = folder / 'extracted'
        require(not destination.is_symlink(), 'Extraction directory is a symlink')
        require(not destination.exists() or not any(destination.iterdir()),
                f'Extraction directory is nonempty; refusing to mix/overwrite files: {destination}')
        with zipfile.ZipFile(archive) as z:
            safe_members(z, destination)
        prepared.append((date,row.scene_id,archive,destination,matches.iloc[0]))
    files, validations = [], []
    for date,scene,archive,destination,inv in prepared:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for info,target in safe_members(z,destination):
                if info.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src,dst,8*1024*1024)
                    require(target.stat().st_size==info.file_size, 'Extracted size mismatch')
        paths=sorted(p for p in destination.rglob('*') if p.is_file())
        groups={kind:[p for p in paths if file_type(p)==kind] for kind in ('LED','IMG','VOL','TRL','ANCILLARY')}
        notes=[]
        valid=bool(groups['LED'] and groups['IMG']) and all(p.stat().st_size>0 for p in groups['LED']+groups['IMG'])
        summaries=[p for p in paths if p.name.lower()=='summary.txt']
        if len(summaries)!=1:
            valid=False; notes.append('Missing or ambiguous summary.txt; product identity not confirmed')
        else:
            summary=dict(re.findall(r'^([^=\r\n]+)="(.*)"\s*$',summaries[0].read_text().strip(),re.M))
            checks={
                'scene identity':summary.get('Scs_SceneID')==inv.scene_name,
                'satellite':summary.get('Lbi_Satellite')==inv.mission=='ALOS',
                'sensor':summary.get('Lbi_Sensor')==inv.sensor=='PALSAR',
                'processing level':summary.get('Lbi_ProcessLevel')=='1.1' and inv.processing_level in ('1.1','L1.1'),
                'observation date':summary.get('Lbi_ObservationDate')==date.replace('-',''),
                'format':summary.get('Pdi_ProductFormat')=='CEOS',
            }
            listed={v for k,v in summary.items() if re.fullmatch(r'Pdi_L15ProductFileName\d+',k)}
            actual={p.name for p in paths if file_type(p)!='ANCILLARY'}
            checks['summary CEOS file list']=listed==actual
            tokens={p.name.split('-')[1] for p in groups['IMG'] if len(p.name.split('-'))>2}
            expected=set(re.split(r'[+;,/\s]+',inv.polarization))
            checks['IMG polarization labels versus inventory']=tokens==expected
            for label,ok in checks.items():
                notes.append(f'{label}: {"consistent" if ok else "CONFLICT"}')
            valid=valid and all(checks.values())
            notes.append('Polarization filename labels cross-checked with inventory and summary file list; binary CEOS records not decoded')
        for p in paths:
            files.append(dict(acquisition_date=date,scene_id=scene,file_type=file_type(p),filename=p.name,
                              absolute_path=str(p.resolve()),size_bytes=p.stat().st_size))
        validations.append(dict(acquisition_date=date,scene_id=scene,archive=str(archive),sha256_verified=True,
            led_count=len(groups['LED']),img_count=len(groups['IMG']),vol_count=len(groups['VOL']),
            trl_count=len(groups['TRL']),product_structure_valid=valid,validation_notes='; '.join(notes)))
        print('\n'+'='*60+f'\n{date} ALOS CEOS PRODUCT\n'+'='*60)
        print(f'Total extracted files: {len(paths)}')
        for kind,ps in groups.items():
            print(f'{kind}:')
            for p in ps: print(f'{p.resolve()} ({p.stat().st_size} bytes)')
        print('; '.join(notes))
    pd.DataFrame(files).to_csv(TABLES/'alos_2009_2010_extracted_files.csv',index=False)
    pd.DataFrame(validations).to_csv(TABLES/'alos_2009_2010_ceos_validation.csv',index=False)
    print('\n'+'='*60+'\nHISTORICAL ALOS PAIR VALIDATION\n'+'='*60)
    for row in validations:
        print(f"{row['acquisition_date'][:4]} product valid: {'YES' if row['product_structure_valid'] else 'NO'}")
    require(all(r['product_structure_valid'] for r in validations),'One or more products failed validation; see CSV notes')
    print('Historical ALOS Level 1.1 pair is ready for GMTSAR preprocessing.')


if __name__=='__main__':
    main()
