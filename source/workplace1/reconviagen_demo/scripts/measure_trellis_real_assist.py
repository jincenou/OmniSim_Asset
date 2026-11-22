"""Summarize measured resources and box-proportion diagnostics; no GT claims."""
from pathlib import Path
import json,statistics

ROOT=Path(__file__).resolve().parents[1]
cases=['fresh_blue_suv','fresh_black_pickup','fresh_silver_suv']
variants=['B0','B0_real1','B0_real4','H']
rows=[]
for case in cases:
    ref=json.loads((ROOT/'inputs'/case/'manifest.json').read_text())['metric_dimensions_lwh']
    for v in variants:
        out=ROOT/'results'/case/f'{v}_seed42'
        i=json.loads((out/'inference.json').read_text())
        e=json.loads((out/'asset/experiment.json').read_text())
        dims=e['metric_dimensions_from_uniform_scale']
        assert e['heading_verified']
        rows.append(dict(case=case,variant=v,seconds=i['inference_seconds'],
            peak_gib=i['peak_cuda_allocated_bytes']/2**30,gaussians=e['reloaded_validation']['gaussian_count'],
            reference_lwh_m=ref,dimensions_lwh_m=dims,
            width_signed_pct=100*(dims[1]/ref[1]-1),height_signed_pct=100*(dims[2]/ref[2]-1),
            render=str(out/'asset/turntable.mp4')))
summary=[]
for v in variants:
    selected=[r for r in rows if r['variant']==v]
    summary.append(dict(variant=v,mean_seconds=statistics.mean(r['seconds'] for r in selected),
        mean_peak_gib=statistics.mean(r['peak_gib'] for r in selected),
        mean_gaussians=statistics.mean(r['gaussians'] for r in selected),
        width_mape_pct=statistics.mean(abs(r['width_signed_pct']) for r in selected),
        height_mape_pct=statistics.mean(abs(r['height_signed_pct']) for r in selected)))
result=dict(rows=rows,summary=summary,seed=42,
    timing_scope='Model inference only; no loading, preprocessing, FLUX, export or render',
    dimension_scope='0.5/99.5-percentile Gaussian centers, uniform input-box-length scaling; width/height box-proportion diagnostic, not held-out accuracy',
    sample_size=3,heldout_used=False,heldout_metrics='N/A',
    inference_mode='Original TRELLIS equal-per-image multidiffusion; compare H as a different-backend contextual control',
    scene_scope='Only existing B0 has scene insertion; six new mixed TRELLIS assets are orbit-tested')
(ROOT/'docs/trellis_real_assist_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
