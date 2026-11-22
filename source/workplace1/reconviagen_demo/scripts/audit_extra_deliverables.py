"""Decode scene and ablation deliverables and record their actual scope."""
from pathlib import Path
import json
import imageio.v2 as imageio
from PIL import Image
from asset_adapter import ROOT, inspect

def video(path, expected):
    with imageio.get_reader(path) as reader:
        n=sum(1 for _ in reader)
    if n!=expected:raise ValueError((path,n,expected))
    return dict(path=str(path),decoded_frames=n)

scenes=[]
for case in ['fresh_blue_suv','fresh_black_pickup','fresh_silver_suv']:
    base=ROOT/'results'/case
    metas=[]
    for variant in ['B0','R']:
        out=base/f'{variant}_seed42/scene0104'
        meta=json.loads((out/'meta.json').read_text());metas.append(meta)
        assert meta['n_frames']==40
        assert all(not a['missing'] and not a['unexpected'] for a in meta['checkpoint_load_audit'])
        evidence=video(out/'insert_splatad_6cam.mp4',40)
        review=dict(case=case,variant=variant,renderer='actual SplatAD Gaussian append and depth sorting',
            checkpoint_keys_match=True,trajectory_continuity='visually reviewed',
            ground_contact='visual floating impression persists; not approved',
            shadows='no new cast/contact shadow simulation',
            production_approved=False,video=evidence)
        (out/'quality_review.json').write_text(json.dumps(review,indent=2));scenes.append(review)
    for key in ['config','traj_npz','step','n_frames','start_frame','target_length_m']:
        assert metas[0][key]==metas[1][key],key
    video(base/'scene0104_comparison_B0_R.mp4',40)

ablations=[]
for tag in ['R1_seed42','R2_seed42','R_seed42','R_seed7']:
    asset=ROOT/'results/fresh_black_pickup'/tag/'asset'
    stats,_=inspect(asset/'canonical.ply')
    assert json.loads((asset/'experiment.json').read_text())['heading_verified']
    ablations.append(dict(tag=tag,loader=stats,video=video(asset/'turntable.mp4',96)))
video(ROOT/'results/fresh_black_pickup/ablation_count_seed.mp4',96)

refinements=[]
for v in ['R','F']:
    out=ROOT/'results/fresh_black_pickup'/f'{v}_refined_seed42'
    if not (out/'inference.json').exists():
        refinements.append(dict(variant=v,status='not_completed'));continue
    meta=json.loads((out/'inference.json').read_text())
    assert meta['status']=='refinement_succeeded_downstream_pending'
    asset=out/'asset';stats,_=inspect(asset/'canonical.ply')
    assert json.loads((asset/'experiment.json').read_text())['heading_verified']
    refinements.append(dict(variant=v,status='refined_exported_reloaded_rendered',
        loader=stats,video=video(asset/'turntable.mp4',96),inference=meta))

paired=[]
if all(x['status']=='refined_exported_reloaded_rendered' for x in refinements):
    for tag in ['R_refine_base','R_refined','F_refine_base','F_refined']:
        asset=ROOT/'results/fresh_black_pickup'/f'{tag}_seed42/asset'
        stats,_=inspect(asset/'canonical.ply')
        assert json.loads((asset/'experiment.json').read_text())['heading_verified']
        for i in range(16):
            assert Image.open(asset/f'view_{i:02}.png').size==(512,512)
            assert Image.open(asset/f'alpha_{i:02}.png').getextrema()[1]>0
        paired.append(dict(tag=tag,views=16,loader=stats,video=video(asset/'turntable.mp4',96)))
    video(ROOT/'results/fresh_black_pickup/refinement_R_F_comparison.mp4',96)
result=dict(scenes=scenes,ablations=ablations,refinements=refinements,paired_refinement_assets=paired,
    production_quality_approved=False,scope='File integrity and actual renderer compatibility, distinct from visual quality')
(ROOT/'docs/extra_deliverable_audit.json').write_text(json.dumps(result,indent=2))
print('Verified six scene videos, three scene comparisons, four count/seed assets; refinement statuses:',[(r['variant'],r['status']) for r in refinements])
