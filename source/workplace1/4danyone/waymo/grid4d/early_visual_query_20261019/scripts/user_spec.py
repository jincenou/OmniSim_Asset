"""Authoritative user clarification received before formal training, 2026-10-19."""
import json
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
p=NEW/'PLAN_DRAFT.json';d=json.loads(p.read_text())
d['training']['new_regularizers'].update(provenance='User supplied exact coefficients and normalization on 2026-10-19; preregistered engineering choices, not original-paper defaults',position_color_before_stage_gate=True,color_units='RGB; SH DC multiplied by C0',once_per_dual_view_update=True,offset_increment_only=True)
d['gates']={
 'provenance':'Explicit user correction on 2026-10-19, before any formal training',
 'effectiveness':{'people':['N06','N11'],'comparisons':['R3_vs_R0','R3_vs_R2'],
 'appearance':{'expanded_RGB_MSE_ratio_max':.95,'expanded_soft_alpha_undercoverage_ratio_max':1.},
 'completeness':{'expanded_soft_alpha_undercoverage_ratio_max':.90,'expanded_RGB_MSE_ratio_max':1.},
 'separate_person_and_path':True},
 'R1':'Required capacity control, no extra percentage threshold. If R1 matches or exceeds R3, do not attribute gain to visual query.',
 'global_protection':{'comparison':'R3_vs_R0','people':['N04','N06','N11'],'foreground_psnr_drop_max':.2,'ssim_drop_max':.001,'crop_lpips_increase_max':.002,'each_core_coverage_drop_max':.01,'each_temporal_rmse_ratio_max':1.05,'outside_alpha_ratio_max':1.05},
 'expanded_excess_alpha_local_LPIPS':'Report on fixed masks and denominator; no invented user-specified threshold.',
 'unknown_not_pass':True,'no_threshold_sweep':True,'visual_veto':['new skin-color islands','geometry flicker','identity color destruction'],
 'A':'Quality reference separately; not a matched-budget causal control'}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
(NEW/'research/user_clarification.json').write_text(json.dumps({'new_regularizers':d['training']['new_regularizers'],'gates':d['gates'],'ROI_construction':'User authorizes engineer to freeze extension/width/min area/local LPIPS validity before viewing new model results.'},ensure_ascii=False,indent=2)+'\n')
