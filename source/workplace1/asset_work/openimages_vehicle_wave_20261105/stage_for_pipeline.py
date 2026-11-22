"""Stage selected Open Images photographs as auditable inputs for the H queue."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / 'web_vehicles_20261103'
INITIAL = {'bus': 240, 'truck': 75, 'van': 75}

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def main():
    counts = {}
    for category, n in INITIAL.items():
        rows = [json.loads(line) for line in (ROOT / f'{category}_selected.jsonl').read_text().splitlines()]
        staged = 0
        for row in rows[:n]:
            assert row['status'] == 'downloaded'
            source_id = f"oi_{row['split']}_{row['image_id']}"
            folder = DEST / 'sources' / source_id
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / 'source.jpg'
            if source.is_symlink():
                assert source.resolve() == Path(row['path']).resolve()
            else:
                assert not source.exists()
                os.symlink(row['path'], source)
            acquisition = folder / 'acquisition.json'
            payload = {
                'source_id':source_id, 'category':category, 'status':'downloaded',
                'image_path':str(source), 'sha256':row['sha256'],
                'download_url':row['image_url'],
                'page_url':row['source_page_url'] or row['image_url'],
                'metadata':{
                    'dataset':'Open Images V7', 'split':row['split'],
                    'image_id':row['image_id'], 'license':row['license'],
                    'author':row['author'], 'original_url':row['original_url'],
                    'title':row['title'], 'annotation_source':row['annotation_source'],
                    'annotation_bbox':row['bbox'], 'quality_flags':row['quality_flags'],
                    'occluded':row['occluded'], 'large_target':row['large_target'],
                    'width':row['width'], 'height':row['height'],
                    'input_manifest':str(ROOT / f'{category}_selected.jsonl')},
                'staged_at':time.time()
            }
            if acquisition.exists():
                old = json.loads(acquisition.read_text())
                assert old['sha256'] == row['sha256'] and old['category'] == category
            else:
                save(acquisition, payload)
            staged += 1
        counts[category] = staged
    rider=[]
    for category in ('motorcycle','bicycle'):
        rows = [json.loads(line) for line in (ROOT / f'{category}_selected.jsonl').read_text().splitlines()]
        eligible = [row for row in rows if row['rider_present'] and row['large_target']
                    and all(value == '0' for value in row['quality_flags'].values())]
        eligible.sort(key=lambda row:(not row['occluded'],row['bbox_area']),reverse=True)
        rider.extend(eligible[:150])
    with (ROOT/'rider_whole_candidates.jsonl').open('w') as out:
        for row in rider:
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(ROOT/'staging_status.json',{'status':'completed','staged_H_vehicle_photos':counts,
                                    'rider_whole_candidates':len(rider),
                                    'rider_condition':'person overlaps vehicle; requires visual confirmation of complete fixed rider and vehicle',
                                    'reserve_photos_per_category':{c:300-n for c,n in INITIAL.items()},
                                    'staged_at':time.time()})

if __name__ == '__main__':
    main()
