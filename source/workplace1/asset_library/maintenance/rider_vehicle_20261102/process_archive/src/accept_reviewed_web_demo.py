"""Record explicit manual scene review for named candidates, gated by bound evidence."""
from pathlib import Path
import sys,json,hashlib
W=Path(__file__).resolve().parents[1];root=W.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in sys.argv[1:]:
 run=W/'runs'/name;dest=root/'rigid/motorcycle'/('rider_web_'+name);ins=run/'insertion_near_follow';meta=json.loads((ins/'meta.json').read_text());asset_sha=sha(dest/'asset.ply')
 assert (run/'insert_near_follow.exit').read_text().strip()=='0' and meta['asset_sha256']==asset_sha==sha(run/'selected/canonical.ply')
 assert meta['n_frames']==24 and meta['trajectory_sha256']==sha(dest/'trajectories/near_follow.npz')
 v=[x for x in json.loads((ins/'visibility_diagnostics.json').read_text()) if x['sensor']=='FRONT'];assert len(v)==24 and all(x['visible_pixels_01']>0 and not x['edge_clipped'] for x in v)
 assert json.loads((dest/'load_validation.json').read_text())['passed'];assert json.loads((dest/'support_validation.json').read_text())['support_difference_m']<.04
 review=dict(status='passed_for_internal_demo',reviewer='Codex manual visual inspection of source, all 16 orbit views and near-follow frames 0/12/23',asset_sha256=asset_sha,trajectory_sha256=meta['trajectory_sha256'],reviewed_frames=[dict(frame=i,sha256=sha(ins/'frames'/f'f{i:03d}.png')) for i in [0,12,23]],front_visible_frames=24,front_edge_clipped_frames=0,minimum_visible_mass_fraction=min(x['visible_mass_fraction'] for x in v),findings='Complete fixed rider/scooter assembly remains visible, with stable scale and travel direction and no gross split in reviewed samples.',limitations=['No fitted contact/cast shadow; lighting is not photorealistic','Single image does not demonstrate temporal rigidity of the source','Face, back clothing, markings and hidden mechanical details synthesized','No deliberate foreground occluder in this per-asset near-follow test; shared renderer occlusion was separately demonstrated with unu'])
 for p in [ins/'visual_review.json',dest/'insertion/near_follow/visual_review.json']:p.write_text(json.dumps(review,indent=2))
 p=dest/'quality.json';q=json.loads(p.read_text());q.update(status='demo_accepted',requires=[],production_release=False,acceptance_scope='Internal fixed-pose whole-assembly Gaussian insertion demo; not exact person/product reconstruction or production photorealism',completed_evidence=review);q['limitations']=list(dict.fromkeys(q['limitations']+review['limitations']));p.write_text(json.dumps(q,indent=2))
 p=run/'selection_review.json';s=json.loads(p.read_text());s['status']='demo_accepted';s['acceptance_scope']=q['acceptance_scope'];p.write_text(json.dumps(s,indent=2));(dest/'selection_review.json').write_text(json.dumps(s,indent=2))
 p=dest/'package.json';d=json.loads(p.read_text());d['status']='demo_accepted';d['files']=sorted(str(f.relative_to(dest)) for f in dest.rglob('*') if f.is_file() and f!=p);p.write_text(json.dumps(d,indent=2));print('ACCEPTED_INTERNAL_DEMO',name)
