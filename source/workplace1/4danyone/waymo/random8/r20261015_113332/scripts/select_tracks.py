"""Freeze input-only sampling protocol before observing any new model outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,random
import numpy as np

RUN=Path(__file__).resolve().parents[1];W=RUN.parents[1];R=W/'round2';PUBLIC=Path('/mnt/public_data/waymo')
def save(path,d):path.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def identity(row):
 assoc=row.get('associated_lidar_track_id');segment=row.get('segment')
 if assoc and segment:return segment+':'+assoc
 return f"{row.get('split','val')}:{row.get('scene')}:{row.get('camera_index',row.get('camera'))}:{row.get('track_id',row.get('track'))}"
def aliases(row):return (str(row.get('split','val')),str(row.get('scene')),str(row.get('camera_index',row.get('camera'))),str(row.get('track_id',row.get('track'))))
def main():
 protocol={'frozen_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'selection_seed':20261015,'generation_seed':42,'training_seed':42,'pool':'202 existing native camera-label cache scenes,processed val RGB present;not allWaymo','camera_indices':[0,1,2],'native_window_preference':[49,25],'min_height_px':80,'median_height_px':120,'image_edge_margin_px':8,'native_size':[1920,1280],'max_difficulty2_fraction':0.2,'native_delta_s_range':[0.08,0.12],'max_10hz_time_deviation_s':0.001,'window_rule':'Try49then25;within length maximize median native bboxheight,then minimumheight,then earliestframe; across officially associated cameras same rule. No ego/action/output-based ranking.','ego_motion_filter':None,'input_pixel_existence_check':'all selected native frames must exist','ordering':'stable official association track key;camera-local identity if association unavailable;random.Random(20261015).shuffle;first8main,next8reserve','selection_probability':'8/N for main within qualified deduplicated pool;16/N preregisteredattempts','historical_exclusion':'All generated/preprocessed/inferred/failed source experiments plus prior20input-audited tracks,legacy candidate-manifest tracks and catalog. Config-only deferred candidates are not treated as used.','model_outputs_seen':False}
 assert not (RUN/'selection_protocol.json').exists();save(RUN/'selection_protocol.json',protocol)
 excluded=set();local=set();evidence=[]
 def exclude(row,origin):
  if not isinstance(row,dict):return
  if row.get('track_id',row.get('track')) is None:return
  excluded.add(identity(row));local.add(aliases(row));evidence.append({'identity_key':identity(row),'camera_local_key':aliases(row),'source':origin})
 for row in json.loads((W/'candidate_manifest.json').read_text()):exclude(row,'legacy_candidate_manifest')
 for row in json.loads((R/'sample_manifest.json').read_text())['candidates']:exclude(row,'prior20_input_audit')
 for row in json.loads((W/'objects/catalog.json').read_text())['objects']:exclude(row,'delivered_catalog')
 for parent in [W/'experiments',R/'experiments']:
  for folder in sorted(parent.iterdir()):
   if not folder.is_dir():continue
   attempted=any((folder/x).exists() for x in ['native.json','inputs/main.mp4','inputs_native/frame_mapping.json','preflight','outputs','experiment_status.json'])
   if not attempted:continue
   for name in ['candidate.json','native.json']:
    path=folder/name
    if path.exists():exclude(json.loads(path.read_text()),str(path))
 candidates=[];reject={};cache_files=sorted((W/'reconstruction/track_reserve/native').glob('*.json'))
 for path in cache_files:
  scene=json.loads(path.read_text());poses=np.asarray(scene['poses']);scene_id=scene['scene']
  for track in scene['tracks']:
   row={'split':'val','scene':scene_id,'segment':scene['segment'],**{k:v for k,v in track.items() if k!='observations'}}
   key=identity(row)
   if key in excluded or aliases(row) in local:continue
   observations=track['observations'];groups=np.split(np.arange(len(observations)),np.flatnonzero(np.diff([o['frame_index'] for o in observations])!=1)+1)
   windows=[]
   for length in [49,25]:
    for group in groups:
     for offset in range(len(group)-length+1):
      part=[observations[int(k)] for k in group[offset:offset+length]];boxes=np.asarray([o['bbox_xyxy'] for o in part]);height=boxes[:,3]-boxes[:,1]
      if height.min()<80 or np.median(height)<120:continue
      if boxes[:,0].min()<8 or boxes[:,1].min()<8 or boxes[:,2].max()>1912 or boxes[:,3].max()>1272:continue
      if np.mean([o['difficulty']>=2 for o in part])>.2:continue
      ts=np.asarray([o['timestamp_micros'] for o in part],dtype=np.int64);dt=np.diff(ts)/1e6
      if dt.min()<.08 or dt.max()>.12 or np.max(np.abs((ts-ts[0])/1e6-np.arange(length)/10))>.001:continue
      windows.append((float(np.median(height)),float(height.min()),-part[0]['frame_index'],part))
    if windows:break
   if not windows:continue
   median,minimum,_,part=max(windows,key=lambda x:x[:3]);a,b=part[0]['frame_index'],part[-1]['frame_index'];cam=row['camera_index']
   if not all((PUBLIC/'val'/scene_id/'images'/f"{o['frame_index']:03d}_{cam}.jpg").is_file() for o in part):continue
   translation=float(np.linalg.norm(np.diff(poses[a:b+1,:3,3],axis=0),axis=1).sum());rot=poses[a:b+1,:3,:3]@poses[a,:3,:3].T;angle=float(np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1))).max())
   candidates.append({**row,'identity_key':key,'identity_scope':'official2D3Dassociation' if row.get('associated_lidar_track_id') else 'camera_local_track_only','observations':part,'native_frame_count':len(part),'output_slots':(len(part)-1)*5//2+1,'duration_s':(part[-1]['timestamp_micros']-part[0]['timestamp_micros'])/1e6,'height_median_px':median,'height_min_px':minimum,'ego_path_m':translation,'ego_rotation_deg':angle,'ego_group':'moving' if translation>1 or angle>2 else 'near_static_or_intermediate','human_action':'unknown_pending_native_review','accessory':'unknown','raw_tfrecord':scene['raw'],'cache':str(path)})
 unique={}
 for row in sorted(candidates,key=lambda x:(-x['native_frame_count'],-x['height_median_px'],-x['height_min_px'],x['observations'][0]['frame_index'],x['camera_index'])):unique.setdefault(row['identity_key'],row)
 pool=sorted(unique.values(),key=lambda x:x['identity_key']);rng=random.Random(20261015);rng.shuffle(pool);assert len(pool)>=16
 chosen=[]
 for i,row in enumerate(pool[:16]):chosen.append({**row,'sample_id':f"N{i+1:02d}",'cohort':'main' if i<8 else 'reserve','ordered_rank':i+1,'main_selection_probability':8/len(pool)})
 save(RUN/'selection.json',{'state':'frozen_input_qualified_random_tracks','pool_size':len(pool),'cached_scenes':len(cache_files),'excluded_keys':len(excluded),'selection_seed':20261015,'main':chosen[:8],'reserve':chosen[8:],'pool_order_sha256':hashlib.sha256('\n'.join(x['identity_key'] for x in pool).encode()).hexdigest()})
 save(RUN/'research/exclusion_evidence.json',evidence);save(RUN/'research/qualified_pool.json',pool)
 save(RUN/'research/selection_summary.json',{'main':[{k:x[k] for k in ['sample_id','scene','camera_index','track_id','native_frame_count','output_slots','height_median_px','ego_path_m','ego_rotation_deg']} for x in chosen[:8]],'reserve':[x['identity_key'] for x in chosen[8:]],'pool_size':len(pool)})
 print(json.dumps(load_summary:=json.loads((RUN/'research/selection_summary.json').read_text())),flush=True)
if __name__=='__main__':main()
