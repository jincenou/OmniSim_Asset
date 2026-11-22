"""Finalize user-defined gates plus reference-only engineering ROI before training."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
NEW=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    assert not (NEW/'PLAN.json').exists();assert not any((NEW/'runs').iterdir())
    proofs=['research/probe/report.json','research/capacity_probe.json','research/control_probe.json','research/probe/asset/replay_validation.json']
    for p in proofs:assert json.loads((NEW/p).read_text())['passed'],p
    total=sum(json.loads((NEW/p).read_text()).get('technical_updates',0) for p in proofs);assert total<=200
    put(NEW/'research/preflight.json',dict(passed=True,technical_updates=total,quality_states_reused=False,proofs={p:sha(NEW/p) for p in proofs},limitations='Memory probe uses synthetic cold-point cloning to65536,not a guarantee of later point count; CUDA rasterizer retains native local-piecewise behavior. Exact state restore and guarded fresh-process asset playback tested.'))
    review=dict(status='reference_only_pretraining_review',reviewed=['N06 and N11:views0,8,16 x first/middle/last plus fixed bad-window center'],observations={'N06':'Expanded capsules include distal lower legs and shoes, with surrounding background and some overlap near crossing legs; no true semantic/visibility GT.','N11':'Capsules include upper/forearms and wrists; side-view arm/head or torso overlaps exist.32 reference-area/overlap rows are unknown; true occlusion is unverified.'},not_reviewed='Every frame has deterministic masks and flags; no claim of manual frame-by-frame visibility verification',prediction_images_seen=False)
    put(NEW/'data/expanded_roi/visual_review.json',review)
    plan=json.loads((NEW/'PLAN_DRAFT.json').read_text());plan['schema']='evq_v1';plan['state']='frozen_before_training';plan['frozen_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    plan['training']['structure']='Native structure-before-optimizer ordering and thresholds retained.Each of2 mean-loss screen grads multiplied2 and accumulated with denominator+1 perview. Native Parameter replacement can skip Gaussian updates on topology steps;actual optimizer calls and per-group updates both reported.'
    plan['training']['paired_regularizer_RNG']='Shared cold warmup and initial RNG. Different topology creates different subsequent native stochastic perturbation draw counts; no claim of bitwise matched gradients.'
    plan['model']['residual_bounds']='Componentwise tanh bounds before gate; world-position each axis<=.03L; DC RGB each channel<=.1.'
    plan['evaluation']['expanded_ROI_rules']=json.loads((NEW/'data/expanded_roi/rules.json').read_text())
    plan['evaluation']['expanded_ROI_sha256']={str(p.relative_to(NEW)):sha(p) for p in (NEW/'data/expanded_roi').rglob('*') if p.is_file()}
    plan['evaluation']['expanded_aggregation']='Per-person and per-side pooled valid reference pixels/alpha mass over allH times; primary left side and fixed historical bad-window separately. Ratios with zero baseline undefined unless both exactly zero, then no demonstrated improvement. No cross-person averaging for gates; missing required support remains unknown.'
    plan['evaluation']['expanded_RGB']='MSE whole fixed proxy ROI including fixed background;do not substitute RMSE. E total squared RGB sum divided3*ROI_pixels; D positive(refalpha-predalpha)sum/reference_alpha_mass.'
    plan['evaluation']['local_LPIPS']='Per-frame valid reference:roi>=256px,alpha mass>=64,other-core fraction<=.25,foreground pixels>=128;fixed perview-side reference union crop at least64x64;mean valid frames,report NA denominator.'
    plan['implementation_sha256']={str(NEW.joinpath('scripts',n).relative_to(NEW)):sha(NEW/'scripts'/n) for n in ['common.py','evq.py','render_evq.py','runtime.py','train.py','assets.py','env.sh','run.sh']}
    plan['resource'].update(measured_65536_points_N11_peak_GiB=json.loads((NEW/'research/capacity_probe.json').read_text())['peak_reserved_bytes']/2**30,minimum_gpu_free_GiB=12,one_project_process_per_gpu=True,estimated_branch_reservation_GiB=6,estimated_warmup_reservation_GiB=1)
    put(NEW/'PLAN.json',plan);put(NEW/'research/freeze.json',dict(at=plan['frozen_at'],plan_sha256=sha(NEW/'PLAN.json'),formal_runs_at_freeze=0,user_supplied_numeric_rules=True,source_correction_sha256=sha(NEW/'research/user_clarification.json')))
    print('Frozen',sha(NEW/'PLAN.json'),flush=True)
if __name__=='__main__':main()
