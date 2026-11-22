"""Read-only delivery audit against actual payloads and completed scene evidence."""
from pathlib import Path
import json,hashlib,subprocess
W=Path(__file__).resolve().parents[1];root=W.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for q in json.loads((W/'sources/web_expansion_20261103/queue.json').read_text()):
 name=q['name'];run=W/'runs'/name;dest=root/'rigid/motorcycle'/('rider_web_'+name);row=dict(identity=name,package=str(dest.relative_to(root)),status='pending_package');rows.append(row)
 if not (dest/'package.json').exists():continue
 try:
  pack=json.loads((dest/'package.json').read_text());cfg=json.loads((dest/'placement.json').read_text());quality=json.loads((dest/'quality.json').read_text());ins=dest/'insertion/near_follow';meta=json.loads((ins/'meta.json').read_text())
  for f in pack['files']:
   p=dest/f;assert not Path(f).is_absolute() and p.resolve().is_relative_to(dest.resolve()) and p.is_file(),f
  assert cfg['forward']=='+X' and cfg['up']=='+Z' and cfg['fixed_pose'] is True and cfg['apply_scale_once'] is True
  assert sha(dest/cfg['ply'])==meta['asset_sha256']==sha(run/'selected/canonical.ply')
  assert sha(dest/cfg['trajectory_files']['near_follow'])==meta['trajectory_sha256']
  assert json.loads((dest/'load_validation.json').read_text())['passed']
  assert float(cfg['metric_scale'])>0 and all(float(x)>0 for x in cfg['dimensions_lwh_m'])
  video=ins/'insert_splatad_6cam.mp4';probe=json.loads(subprocess.check_output(['/home/wmj/miniconda3/envs/sim/bin/ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)]));frames=int(probe['streams'][0]['nb_read_frames']);assert frames==meta['n_frames']==24
  v=[r for r in json.loads((ins/'visibility_diagnostics.json').read_text()) if r['sensor']=='FRONT'];assert len(v)==24
  clipped=sum(r['edge_clipped'] for r in v);visible=sum(r['visible_pixels_01']>0 for r in v)
  row.update(status='artifact_checks_passed',quality_status=quality['status'],asset_sha256=meta['asset_sha256'],decoded_frame_count=frames,front_visible_frames=visible,front_edge_clipped_frames=clipped,visual_review_required=quality['status']!='demo_accepted')
 except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
p=W/'reports/web_expansion_delivery_audit.json';p.write_text(json.dumps(dict(scope='File portability, hash binding, numerical load and video/visibility evidence. Does not replace visual review.',assets=rows),indent=2));print(json.dumps({s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))}))
