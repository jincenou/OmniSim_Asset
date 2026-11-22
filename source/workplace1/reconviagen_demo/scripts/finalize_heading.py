"""Apply visually audited heading, preserving first export and rendering anew."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
from asset_adapter import ROOT,export
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--flip',action='store_true');p.add_argument('--confidence',type=float,default=1.);p.add_argument('--note',default='Visual front/rear light, grille and cabin check');a=p.parse_args()
out=ROOT/'results'/a.case/f'{a.variant}_seed{a.seed}';asset=out/'asset';archive=out/'asset_heading_unverified'
if not a.flip:
    # The canonical geometry already faces +X. Preserve it byte-for-byte and
    # accept its existing renders instead of exporting an identical payload.
    review_path=asset/'heading_review.json'
    if review_path.exists():raise FileExistsError(review_path)
    original={name:json.loads((asset/name).read_text()) for name in ['metadata.json','library.json','experiment.json']}
    (asset/'heading_initial_metadata.json').write_text(json.dumps(original,indent=2))
    for name in ['metadata.json','library.json']:
        data=original[name]
        entry=data if name=='metadata.json' else data['assets'][0]
        entry['canonical']['heading']={'front':'+X','confidence':a.confidence}
        (asset/name).write_text(json.dumps(data,indent=2))
    original['experiment.json']['heading_verified']=True
    (asset/'experiment.json').write_text(json.dumps(original['experiment.json'],indent=2))
    review_path.write_text(json.dumps(dict(evidence=str(asset/'turntable.jpg'),observed_front_before='+X',applied_yaw_flip_deg=0,front_after='+X',confidence=a.confidence,review=a.note+'; existing canonical payload and renders retained unchanged',canonical_sha256=hashlib.sha256((asset/'canonical.ply').read_bytes()).hexdigest()),indent=2))
    sys.exit(0)
if archive.exists():raise FileExistsError(archive)
asset.rename(archive)
for f in archive.glob('*.json'):f.write_text(f.read_text().replace(str(asset),str(archive)))
case=json.loads((ROOT/'inputs'/a.case/'manifest.json').read_text());source=out/'finite.ply'
if not source.exists():source=out/'raw.ply'
export(source,asset,case,a.variant,'-y' if a.variant in ('B0','B0_real1','B0_real4','B0_repeat') else 'z',flip=a.flip,heading_verified=True)
env=os.environ.copy();env.update(PATH='/home/wmj/miniconda3/envs/sim/bin:'+env['PATH'],CPATH='/home/wmj/miniconda3/envs/splatad/include',TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch_extensions'),MAX_JOBS='4')
subprocess.run(['/home/wmj/miniconda3/envs/sim/bin/python',str(ROOT/'scripts/render_orbit.py'),'--ply',str(asset/'canonical.ply'),'--output',str(asset),'--length',str(case['metric_dimensions_lwh'][0]),'--distance','11'],env=env,check=True)
(asset/'heading_review.json').write_text(json.dumps(dict(evidence=str(archive/'turntable.jpg'),observed_front_before='+X' if not a.flip else '-X',applied_yaw_flip_deg=180 if a.flip else 0,front_after='+X',confidence=a.confidence,review=a.note),indent=2))
if a.confidence!=1.:
    for name in ['metadata.json','library.json']:
        data=json.loads((asset/name).read_text());entry=data if name=='metadata.json' else data['assets'][0]
        entry['canonical']['heading']['confidence']=a.confidence
        entry['defects'].append(a.note);(asset/name).write_text(json.dumps(data,indent=2))
