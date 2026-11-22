"""Select same-official-identity alternative windows; no admission threshold changes."""
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scan_native import ROOT, load_tracks, sha


def main():
    output = ROOT/'native_mask_retry_queue.json'
    if output.exists():
        raise RuntimeError('Keep existing selection immutable')
    cfg = json.loads((ROOT/'SCREEN_PROTOCOL.json').read_text())
    assert sha(ROOT/'SCREEN_PROTOCOL.json') == (ROOT/'SCREEN_PROTOCOL.sha256').read_text().strip()
    inventory = {(r['split'], r['scene']):r for r in json.loads((ROOT/'inventory.json').read_text())['scenes']}
    risks = json.loads((ROOT/'input_risk_inventory.json').read_text())['rows']
    selected, decisions, caches = [], [], {}
    for risk in risks:
        if ((risk['empty_binary_mask_fraction'] or 0)<0.5 or
            risk['identity_scope']=='predicted_camera_local_not_official' or
            '/samples/' not in risk['package']):
            continue
        primary = Path(risk['package'])
        original = json.loads((primary/'READY.json').read_text())['candidate']
        key = (original['split'],original['scene'])
        if key not in caches:
            caches[key] = load_tracks(inventory[key])
        tracks, cache = caches[key]
        associated = original.get('associated_lidar_track_id')
        tracks = [t for t in tracks if
                  (bool(associated) and t.get('associated_lidar_track_id') == associated) or
                  (t['camera_index']==original['camera_index'] and t['track_id']==original['track_id'])]
        old_frames = {o['frame_index'] for o in original['observations']}
        options, rejected = [], Counter()
        for length in cfg['native_lengths']:
            for track in tracks:
                obs = sorted(track['observations'],key=lambda o:o['frame_index'])
                for start in range(len(obs)-length+1):
                    part=obs[start:start+length]
                    indices=[o['frame_index'] for o in part]
                    if np.any(np.diff(indices)!=1):
                        continue
                    same_camera=track['camera_index']==original['camera_index']
                    if same_camera and len(old_frames.intersection(indices))/min(len(old_frames),length)>0.5:
                        rejected['overlaps_primary_more_than_half']+=1
                        continue
                    boxes=np.asarray([o['bbox_xyxy'] for o in part]);height=boxes[:,3]-boxes[:,1]
                    if not np.isfinite(boxes).all() or np.any(height<=0) or np.any(boxes[:,2]<=boxes[:,0]):
                        rejected['invalid_box']+=1;continue
                    if height.min()<cfg['minimum_height_px'] or np.median(height)<cfg['median_height_px']:
                        rejected['resolution']+=1;continue
                    if np.any(boxes[:,:2]<0) or np.any(boxes[:,2:]>[1920,1280]):
                        rejected['outside_image']+=1;continue
                    dt=np.diff([o['timestamp_micros'] for o in part])/1e6
                    if np.any((dt<.08)|(dt>.12)):
                        rejected['timestamp_gap']+=1;continue
                    if not all((Path(inventory[key]['processed_path'])/'images'/f"{o['frame_index']:03d}_{track['camera_index']}.jpg").is_file() for o in part):
                        rejected['missing_pixels']+=1;continue
                    difficulty=float(np.mean([o['difficulty']>=2 for o in part]))
                    score=(-difficulty,float(np.median(height)),float(height.min()),-indices[0],-track['camera_index'])
                    options.append((score,track,part))
            if options:
                break
        decision={'primary':str(primary),'identity_key':original['identity_key'],
                  'eligible_alternatives_at_selected_length':len(options),
                  'window_reasons_not_people':dict(rejected),'identity_rejected':False}
        if options:
            score,track,part=max(options,key=lambda o:o[0])
            identity=original['segment']+':'+(track.get('associated_lidar_track_id') or f"cam{track['camera_index']}:{track['track_id']}")
            assert identity==original['identity_key']
            candidate=copy.deepcopy(original)
            candidate.update(observations=part,native_frame_count=len(part),output_slots=(len(part)-1)*5//2+1,
                             camera_index=track['camera_index'],track_id=track['track_id'],
                             review_id=original['review_id']+'_nalt1',height_median_px=score[1],
                             height_min_px=score[2],difficulty2_fraction=-score[0],
                             track_cache=str(cache),track_cache_sha256=sha(cache),
                             sample_root=str(ROOT/'retry'/key[0]/key[1]/original['review_id']/'native_alternative1/clip0'),
                             retry_of=str(primary),generation_allowed=False,
                             selection_rule='Existing moderate metadata bounds and original length/difficulty/resolution ranking; distinct time window or official same-identity camera. Risk only prioritizes additional preparation, not rejection.')
            assert not Path(candidate['sample_root']).parent.exists()
            selected.append(candidate)
            decision.update(state='selected_for_input_preparation',camera=track['camera_index'],
                            frames=[part[0]['frame_index'],part[-1]['frame_index']])
        else:
            decision['state']='no_diverse_alternative_under_frozen_metadata_rules'
        decisions.append(decision)
    assert len(decisions)==13
    assert len({c['identity_key'] for c in selected})==len(selected)
    result={'utc':datetime.now(timezone.utc).isoformat(),'candidates':selected,'decisions':decisions,
            'generation_allowed':False,'screen_protocol_sha256':sha(ROOT/'SCREEN_PROTOCOL.json'),
            'risk_snapshot_sha256':sha(ROOT/'input_risk_inventory.json'),
            'selection_script_sha256':sha(Path(__file__)),
            'limits':['Same identity, not additional people.',
                      'Measured exposure, crop/framing and conditioning checks remain mandatory during preparation.',
                      'Alternative success and quality are unknown until executed and reviewed.']}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'reviewed_native_keys':len(decisions),'alternatives':len(selected),
                      'selected':[(c['review_id'],c['camera_index'],c['observations'][0]['frame_index'],c['observations'][-1]['frame_index']) for c in selected]},ensure_ascii=False))


if __name__=='__main__':
    main()
