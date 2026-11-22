"""Report file-backed experiment stages without treating PLY as full success."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for case in ['service_van','fresh_blue_suv','fresh_black_pickup','fresh_silver_suv']:
    for variant in ['B0','R','F','H']:
        out=ROOT/'results'/case/f'{variant}_seed42'
        infer=out/'inference.json';meta=out/'asset/metadata.json';scene=out/'scene0104/meta.json'
        row=dict(case=case,variant=variant,seed=42,output=str(out),inference='pending',
            original_loader_reload='pending',orbit='pending',scene='pending',quality='unreviewed')
        if (out/'failure.txt').exists():row.update(inference='failed',failure=str(out/'failure.txt'))
        if infer.exists():
            row.update(inference='succeeded',**{k:v for k,v in json.loads(infer.read_text()).items()
                if k in ['gaussian_count','load_seconds','inference_seconds','peak_cuda_allocated_bytes']})
        if (out/'asset/experiment.json').exists():
            x=json.loads((out/'asset/experiment.json').read_text())
            if x.get('reloaded_validation'):row['original_loader_reload']='succeeded'
            row['heading_verified']=x.get('heading_verified',False)
        if (out/'asset/turntable.mp4').exists() and (out/'asset/render_config.json').exists():row['orbit']='rendered'
        if scene.exists() and (out/'scene0104/insert_splatad_6cam.mp4').exists():
            row['scene']='rendered_needs_visual_review'
            if (out/'scene0104/quality_review.json').exists():row['scene']='rendered_reviewed_contact_not_approved'
        if meta.exists():row['quality']=json.loads(meta.read_text()).get('status','unreviewed')
        rows.append(row)
extras=[]
for tag in ['R1_seed42','R2_seed42','R_seed7','R_refined_seed42','F_refined_seed42']:
    out=ROOT/'results/fresh_black_pickup'/tag
    row=dict(case='fresh_black_pickup',tag=tag,output=str(out),status='not_completed')
    if (out/'asset/heading_review.json').exists() and (out/'asset/turntable.mp4').exists():row['status']='generated_reloaded_heading_reviewed_rendered'
    if (out/'asset/quality_review.json').exists():row['quality_review']=str(out/'asset/quality_review.json')
    extras.append(row)
result=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),experiments=rows,additional_experiments=extras,
    superseded_cases=['silver_sedan_03','white_sedan'],superseded_reason='User requested fresh selection and regenerated FLUX; archived weak inputs retained',
    setup='metadata-assisted; common training-view 3D-box length per object',
    metric_status='N/A until an input-only pose registration is validated; FLUX is not ground truth')
(ROOT/'docs/experiment_status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
