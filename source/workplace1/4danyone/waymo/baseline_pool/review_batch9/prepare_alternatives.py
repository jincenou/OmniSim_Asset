"""Select same-identity later windows with the unchanged baseline thresholds."""
import copy,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import scan_native as native
from screen_predicted import select_windows
sha=native.sha;write=native.write
cfg=json.loads((R/'SCREEN_PROTOCOL.json').read_text());assert sha(R/'SCREEN_PROTOCOL.json')==(R/'SCREEN_PROTOCOL.sha256').read_text().strip()
selected=json.loads((R/'review_batch9/selection.json').read_text())['selected'];decisions=[]
# Select from same original track; newly exposed frames require review after preparation.
for row in selected:
 p=Path(row['package']);old=json.loads((p/'READY.json').read_text())['candidate'];assert (p/'review.json').is_file()
 if row['split']=='train':
  tracks=json.loads(Path(old['track_cache']).read_text());assert sha(Path(old['track_cache']))==old['track_cache_sha256']
  track=next(t for t in tracks['tracks'] if t['track_id']==old['track_id'])
  observations=[x for x in track['observations'] if 105<=x['frame_index']<=129]
  choices,rejections,_=select_windows(observations,cfg)
  candidate=copy.deepcopy(old) if choices else None
  if candidate:candidate.update(choices[0])
  prefix='review9_predicted_retry';limits=[105,129]
 else:
  inventory=json.loads((R/'inventory.json').read_text())['scenes'];scene=next(x for x in inventory if x['split']==row['split'] and x['scene']==row['scene'])
  tracks,cache=native.load_tracks(scene);target=copy.deepcopy(next(t for t in tracks if t['track_id']==old['track_id'] and t['camera_index']==old['camera_index']))
  target['observations']=[x for x in target['observations'] if 169<=x['frame_index']<=197]
  original=native.load_tracks
  try:
   native.load_tracks=lambda _:([target],cache)
   found=native.screen(scene,cfg)
  finally:native.load_tracks=original
  choices=found['candidates'];candidate=copy.deepcopy(old) if choices else None
  if candidate:candidate.update(choices[0])
  prefix='review9_native_retry';limits=[169,197]
 decision=dict(primary=str(p),allowed_native_frames=limits,eligible_windows=len(choices),screen_protocol_unchanged=True)
 if candidate:
  candidate.update(review_id=row['review_id']+'_window1',sample_root=str(R/'retry'/row['split']/row['scene']/row['review_id']/'window1/clip0'),retry_of=str(p),visual_state='unreviewed',generation_allowed=False,selection_rule='Same identity and camera: train105-129 avoids terminal truncation, val169-197 avoids initial occlusion. Contains source frames outside the earlier review; full new review required. Original thresholds unchanged.')
  candidate.pop('alternative_frame_windows',None)
  decision['selected_frames']=[candidate['observations'][0]['frame_index'],candidate['observations'][-1]['frame_index']]
  out=R/(prefix+'_queue.json');assert not out.exists()
  write(out,dict(created_at=time.time(),candidates=[candidate],decision=decision,screen_protocol_sha256=sha(R/'SCREEN_PROTOCOL.json'),new_people=0,generation_allowed=False))
 decisions.append(decision)
write(R/'review_batch9/alternative_selection.json',dict(decisions=decisions,at=time.time(),new_people=0))
print(json.dumps(decisions))
