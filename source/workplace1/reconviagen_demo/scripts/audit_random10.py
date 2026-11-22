"""Reload all new payloads, decode every demo frame, and record actual validity."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from asset_adapter import inspect
ROOT=Path(__file__).resolve().parents[1]
tasks=json.loads((ROOT/'inputs/random10_experiment_matrix.json').read_text());records=[]
for task in tasks:
 case=task['case_id'];v=task['variant'];seed=task['seed'];run=ROOT/'results'/case/f'{v}_seed{seed}';asset=run/'asset'
 validation,gs=inspect(asset/'canonical.ply');heading=json.loads((asset/'heading_review.json').read_text())
 assert heading['front_after']=='+X'
 audit=json.loads((run/'evaluation/render_audit.json').read_text());assert audit['valid_export'] and audit['valid_16_views']
 for prefix,folder,alpha_pattern in [('orbit',run/'evaluation','orbit_{:02}_alpha.png'),('standalone',asset,'alpha_{:02}.png')]:
  counts=[int((np.array(Image.open(folder/alpha_pattern.format(i)))>127).sum()) for i in range(16)]
  assert all(n>0 for n in counts),(case,v,prefix)
 reader=imageio.get_reader(str(asset/'turntable.mp4'));n=sum(1 for frame in reader);reader.close();assert n==96
 # Seed is part of the experiment identity; avoid index collisions for extras.
 uid=f'car_reconviagen_{case}_{v}_seed{seed}'
 for name in ['metadata.json','library.json']:
  data=json.loads((asset/name).read_text());entry=data if name=='metadata.json' else data['assets'][0]
  entry.update(uid=uid,global_uid='waymo:'+uid);(asset/name).write_text(json.dumps(data,indent=2))
 record=dict(case_id=case,variant=v,seed=seed,gaussians=validation['gaussian_count'],canonical_sha256=hashlib.sha256((asset/'canonical.ply').read_bytes()).hexdigest(),orbit_views_nonempty=16,decoded_video_frames=n,heading_confidence=heading.get('confidence',1.),matched_source_foreground=audit['matched_foreground_pixels'],status='generated_reloaded_heading_reviewed_rendered',production_approved=False,scene_insertion_verified=False)
 records.append(record);task['status']=record['status'];print(case,v,'PASS',flush=True)
videos=[]
for case in sorted({t['case_id'] for t in tasks}):
 path=ROOT/'results'/case/'comparison_six_routes.mp4';reader=imageio.get_reader(str(path));n=sum(1 for frame in reader);reader.close();assert n==96
 videos.append(dict(case_id=case,path=str(path),decoded_frames=n))
(ROOT/'inputs/random10_experiment_matrix.json').write_text(json.dumps(tasks,indent=2))
(ROOT/'docs/random10_deliverable_audit.json').write_text(json.dumps(dict(status='PASS',primary_assets=len(records),actual_nonempty_metric_orbit_views=16*len(records),decoded_asset_video_frames=sum(r['decoded_video_frames'] for r in records),comparisons=videos,assets=records,time_unix=time.time(),scene_insertion='Not run for this new batch; previous three-object scene insertion demos remain separate'),indent=2))
print('AUDIT COMPLETE',len(records),len(videos),flush=True)
