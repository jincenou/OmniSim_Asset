"""Commit the visually verified, single retry under the existing object identity."""
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CHUNK = ROOT/'chunks/waymo_waymo_val_033_0120'
UID = 'waymo_val_033_t000754'
GLOBAL_UID = 'waymo:waymo_val_033:' + UID
sys.path.insert(0, str(ROOT))
from runtime import LIBRARY, read_json, write_json, sha256, apply_identity_curation
from asset_extraction.bank import publish


def main():
    with (WORK/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'finalized.json').exists():
            print(json.dumps(read_json(WORK/'finalized.json')))
            return
        ready = read_json(WORK/'ready_for_visual_review.json')
        retry = read_json(ready['curation'])
        generated = read_json(ready['generation'])
        row = generated['assets'][0]
        assert row['obj_id']==UID and row['selected_attempt']==1
        assert row['real_images']==[] and row['stages']['H']['real_count']==0
        flux = row['stages']['flux']['images']
        assert row['stages']['H']['input_images']==[x['path'] for x in flux]
        assert all(sha256(x['path'])==x['sha256'] for x in flux)
        assert sha256(row['raw_path'])==row['raw_sha256']
        record = deepcopy(retry['assets'][UID])
        assert record['status']=='keep'
        entry = record['entry']
        evidence = WORK/'shape_retry'/UID/'curation/asset_00000/axis_render/evidence.jpg'
        review = dict(uid=GLOBAL_UID, reviewed_at=time.time(), reviewer='Codex visual inspection',
            verdict='keep', evidence=str(evidence), evidence_sha256=sha256(evidence),
            reason='All 16 views show a complete enclosed commercial van body, front and rear ends, cargo volume and four-wheel layout. The cabin-only body defect is corrected; small texture and trim differences are tolerated.',
            limitations='Source is a small low-resolution crop. This validates category/body completeness, not exact source model, measured dimensions, or physical contact accuracy.',
            real_count=0, flux_images=flux, original_review=str(WORK/'visual_decision.json'))
        write_json(WORK/'final_visual_review.json', review)
        record['manual_body_review_final'] = review
        entry['provenance']['batch_id'] = ROOT.name
        entry['provenance']['manual_body_correction'] = str(WORK/'final_visual_review.json')
        fixed = apply_identity_curation(dict(assets={UID:record}), 'waymo', ROOT.name)
        record = fixed['assets'][UID]
        # Fail closed if another process has since published the same native identity.
        if record['status'] != 'keep':
            raise ValueError('Corrected asset no longer eligible after native identity deduplication')
        old_publication = read_json(WORK/'withdrawn_publication_receipt.json')
        destination = LIBRARY/'releases/rigid'/old_publication['entry']['asset_path']
        archive = WORK/'original_published_payload.ply'
        with (LIBRARY/'.rigid.lock').open('a') as bank_lock:
            fcntl.flock(bank_lock, fcntl.LOCK_EX)
            if destination.exists() and sha256(destination)==old_publication['output_sha256']:
                if archive.exists():
                    assert sha256(archive)==old_publication['output_sha256']
                    # Preserve the payload without deleting either original chunk output.
                    destination.unlink()
                else:
                    destination.rename(archive)
            elif destination.exists():
                assert sha256(destination)==entry['canonical']['sha256']
        added = publish(fixed, LIBRARY, dataset='waymo', scene_id='waymo_val_033')
        assert len(added)==1 and added[0]['uid']==GLOBAL_UID
        published = added[0]
        assert sha256(LIBRARY/published['canonical']['canonical_path'])==published['canonical']['sha256']
        for relative, backup in [('completed.json','quarantined_completed.json'),
                                 ('curation/curation.json','quarantined_curation.json'),
                                 ('selected_generation.json','quarantined_selected_generation.json'),
                                 ('technical_validity.json','quarantined_technical_validity.json')]:
            if not (WORK/backup).exists():
                write_json(WORK/backup, read_json(CHUNK/relative))
        state = read_json(CHUNK/'curation/curation.json')
        original_other = {k:v for k,v in state['assets'].items() if k!=UID}
        state['assets'][UID] = record
        state['counts'] = dict(queries=len(state['assets']), **Counter(x['status'] for x in state['assets'].values()))
        selected = read_json(CHUNK/'selected_generation.json')
        prior_row = next(x for x in selected['assets'] if x['obj_id']==UID)
        replacement = dict(row, original_item_index=prior_row.get('original_item_index', prior_row['item_index']))
        selected['assets'] = [replacement if x['obj_id']==UID else x for x in selected['assets']]
        selected['counts'] = dict(queries=len(selected['assets']), generated=sum(x['status']=='generated' for x in selected['assets']))
        technical = read_json(CHUNK/'technical_validity.json')
        render = entry['quality']['rendering']
        pixels = render.get('foreground_pixels', [])
        assert render.get('all_16_nonempty') and len(pixels)==16 and all(x>0 for x in pixels)
        for x in technical['assets']:
            if x['obj_id']==UID:
                x.update(status=record['status'], generated=True, valid_16=True, rendering=render,
                         selected_attempt=1, manual_body_review=str(WORK/'final_visual_review.json'))
        receipt = read_json(CHUNK/'completed.json')
        receipt['published'] = [x for x in receipt['published'] if x['uid']!=GLOBAL_UID]
        receipt['published'].append(dict(uid=GLOBAL_UID,
            canonical_path=str(LIBRARY/published['canonical']['canonical_path']),
            sha256=published['canonical']['sha256'], source_obj_id=UID, status='keep', category='van'))
        receipt.update(counts=state['counts'], generation_counts=selected['counts'],
            kept=sum(x['status']=='keep' for x in receipt['published']),
            manual_body_review=dict(record=str(WORK/'final_visual_review.json'), updated=time.time(), status='completed'))
        assert receipt['queries']==read_json(WORK/'previous_completed.json')['queries']==24
        assert {k:v for k,v in state['assets'].items() if k!=UID}==original_other
        write_json(CHUNK/'selected_generation.json', selected)
        write_json(CHUNK/'curation/curation.json', state)
        write_json(CHUNK/'technical_validity.json', technical)
        write_json(CHUNK/'completed.json', receipt)
        result = dict(uid=GLOBAL_UID, status='completed', completed=time.time(),
            selected_attempt=1, queries_before=24, queries_after=24,
            canonical_path=receipt['published'][-1]['canonical_path'],
            sha256=published['canonical']['sha256'], original_payload_preserved=str(archive))
        write_json(WORK/'finalized.json', result)
        write_json(WORK/'status.json', result)
        print(json.dumps(result))


if __name__=='__main__':
    main()
