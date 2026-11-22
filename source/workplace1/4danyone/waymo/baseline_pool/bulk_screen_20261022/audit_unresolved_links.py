"""Catalogue old auxiliary proposals; never count proposals as unique people."""
import json,csv,collections,time,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;P=B.parent
rows=[r for r in json.loads((B/'results.json').read_text()) if r['group']=='priority_generation_candidate' and r['identity_scope']!='official_2d3d_association'];cache={};output=[]
conf=json.loads((P/'predicted_identity_summary.json').read_text())['conflicts'];confkeys=set()
for x in conf:
 for ids in x['overlap_frames'].values():confkeys.update(ids)
for r in rows:
 key=(r['split'],r['scene'])
 if key not in cache:
  p=P/'predicted_identity'/key[0]/(key[1]+'.json');p=p if p.exists() else P/'identity_validation'/key[0]/(key[1]+'.json');a=json.loads(p.read_text()) if p.exists() else None;cache[key]=(p,a)
 p,a=cache[key];c=json.loads((Path(r['package'])/'READY.json').read_text())['candidate'];links=[] if a is None else [x for x in a.get('links',[]) if x.get('identity_key')==r['identity_key'] and x.get('camera_index')==r['camera'] and x.get('track_id')==c['track_id']];link=links[0] if len(links)==1 else None
 if a is None:state='no_cached_scene_diagnostic'
 elif not link:state='no_unique_cached_track_diagnostic'
 elif r['identity_key'] in confkeys:state='same_camera_overlap_conflict'
 elif link.get('state')=='proposed_association':state='unverified_old_proposal'
 else:state=link.get('state','unknown')
 output.append(dict(split=r['split'],scene=r['scene'],review_id=r['review_id'],identity_key=r['identity_key'],camera=r['camera'],identity_scope=r['identity_scope'],diagnostic_state=state,scene_diagnostic_state=a.get('state') if a else None,proposed_world_track=link.get('proposed_world_track') if link else None,time_basis=link.get('time_basis') if link else None,cached_diagnostic=str(p) if a else None,selected_package=r['package'],selected_window_start=r['window_start'],selected_window_end=r['window_end'],scores_recomputed_for_selected_window=False,automatic_merge=False,physical_identity_verified=False))
with (B/'unresolved_link_status.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
a=dict(at=time.time(),windows=len(output),scenes=len(cache),states=dict(collections.Counter(r['diagnostic_state'] for r in output)),time_bases=dict(collections.Counter(r['time_basis'] for r in output)),scene_diagnostic_states=dict(collections.Counter(r['scene_diagnostic_state'] for r in output)),automatic_merge=False,unique_people=None,scope='Coverage of old cached track proposals against current 2702 unresolved candidates; old scores may refer to different windows and are not recomputed or admitted',source_results_sha256=hashlib.sha256((B/'results.json').read_bytes()).hexdigest());(B/'UNRESOLVED_LINK_STATUS.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
