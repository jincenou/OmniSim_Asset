"""Check official candidate provenance without promoting inferred associations."""
from pathlib import Path
import collections,csv,hashlib,json,time
R=Path(__file__).resolve().parent
P=R.parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 rows=json.loads((R/'results.json').read_text());official=[x for x in rows if x['group']=='priority_generation_candidate' and x['identity_scope']=='official_2d3d_association']
 scenes={};evidence=[];seen=set()
 for row in official:
  key=(row['split'],row['scene'])
  if key not in scenes:
   scan=P/'native_scan'/key[0]/(key[1]+'.json');s=json.loads(scan.read_text());source=Path(s['source']);assert sha(source)==s['source_sha256'],source
   data=json.loads(source.read_text());tracks=data['tracks'] if isinstance(data,dict) else data
   index={}
   for t in tracks:
    cam=t.get('camera_index') if key[0]=='val' else {'FRONT':0,'FRONT_LEFT':1,'FRONT_RIGHT':2}.get(t.get('camera_name'))
    index[(cam,t['track_id'])]=t
   scenes[key]=(index,str(source),s['source_sha256'])
  package=Path(row['package']);assert sha(package/'READY.json')==row['ready_sha256']
  c=json.loads((package/'READY.json').read_text())['candidate'];index,source,digest=scenes[key];t=index[(c['camera_index'],c['track_id'])]
  lid=t['associated_lidar_track_id'];assert lid and lid==c['associated_lidar_track_id']
  assert row['identity_key']==c['segment']+':'+lid
  obs={o['frame_index']:o for o in t['observations']}
  for o in c['observations']:
   original=obs[o['frame_index']];assert original['timestamp_micros']==o['timestamp_micros']
  assert row['identity_key'] not in seen,'More than one selected window per official scene identity'
  seen.add(row['identity_key'])
  evidence.append(dict(split=key[0],scene=key[1],review_id=row['review_id'],identity_key=row['identity_key'],camera=row['camera'],track_id=c['track_id'],package=str(package),label_source=source,label_source_sha256=digest,selected_observations_verified=len(c['observations']),input=row['input'],pose=row['pose'],quality_accepted=False,generation_allowed=False))
 with (R/'official_identity_candidates.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(evidence[0]));writer.writeheader();writer.writerows(evidence)
 result=dict(at=time.time(),state='official_candidate_keys_matched_unchanged_label_sources',official_candidate_windows=len(evidence),distinct_scene_identity_keys=len(seen),source_scenes=len(scenes),split_counts=dict(collections.Counter(r['split'] for r in evidence)),checked_observations=sum(r['selected_observations_verified'] for r in evidence),sources=[dict(split=k[0],scene=k[1],path=v[1],sha256=v[2]) for k,v in scenes.items()],results_sha256=sha(R/'results.json'),csv_sha256=sha(R/'official_identity_candidates.csv'),scope='Match prepared candidate track and exact timestamps to existing official-label exports. Not a fresh TFRecord parse, visual identity validation, cross-scene identity proof or generated quality approval.',unresolved_candidates=3259-len(evidence),quality_accepted_people=None,whole_pool_unique_people=None)
 (R/'IDENTITY_SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('sources',)},ensure_ascii=False))
if __name__=='__main__':main()
