"""Freeze the newly user-authorized eight-identity expansion, keeping old plans intact."""
import json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
V3=Path(__file__).resolve().parents[1];ROOT=V3.parent;NEW=ROOT/'optimization_cross8_20261018'
sys.path.insert(0,str(ROOT/'scripts'))
from freeze_config import make
from grid_common import sha,save
assert not (NEW/'PLAN.json').exists()
for name in ['scripts','configs','source','research','runs','results','cache/raw','comparison','logs','tmp']:(NEW/name).mkdir(parents=True,exist_ok=True)
copied=['quality_common.py','train_prefix.py','train_fused.py','train_continuation.py','validate_quality.py','evaluate_quality.py','evaluate_parts.py','publish_quality.py']
for name in copied:shutil.copy2(V3/'scripts'/name,NEW/'scripts'/name)
shutil.copy2(V3/'source/train_fused.py',NEW/'source/train_fused.py')
(NEW/'REQUEST.txt').write_text('用户：你可以多尝试几个对象，之前不是有一共8个对象，现在都可以训练出来看看效果。\n继续使用已授权八张GPU，可与其他任务共享。复用已完成A与融合结果；补齐同八个有效输入的完整A和固定融合配方，不按人调整配方。\n')
labels=['N01','N02','N03','N04','N06','N07','N08','N11'];new_A=['N02','N03','N06','N08'];new_fused=['N02','N03','N04','N06','N08','N11'];configs={};fused={}
for label in sorted(set(new_A+new_fused)):
    config=make(label);config['saving_iterations']=[33000]
    config['cross8_note']='Same frozen recipe; only terminal asset saved to keep output compact. Rolling complete recovery still saved every1000 native iterations.'
    if label in new_A:
        p=NEW/'configs'/f'{label}_33k.json';save(p,config);configs[label]=dict(path=str(p),sha256=sha(p),source=config['provenance'])
    p=NEW/'configs'/f'{label}_fused.json';save(p,config);fused[label]=dict(path=str(p),sha256=sha(p),source=config['provenance'])
prior=json.loads((V3/'PLAN.json').read_text());cross=prior['cross_person'];cross.update(labels=new_A,configs=configs,save_alpha_extra=[15000],save_A_extra=[5000])
reuse={}
for label in ['N01','N04','N07','N11']:
    p=(V3/'runs'/f'{label}_A') if label in ['N01','N07'] else ROOT/'optimization_boundary_20261018/runs/primary'/f'{label}_A'
    m=(V3/'results'/f'{label}_A_53000') if label in ['N01','N07'] else ROOT/'optimization_boundary_20261018/results/primary'/f'{label}_A'
    raw=(V3/'cache/raw'/f'{label}_A_53000') if label in ['N01','N07'] else ROOT/'optimization_boundary_20261018/cache/raw/primary'/f'{label}_A'
    reuse[label+'_A']=dict(run=str(p),results=str(m),raw=str(raw),complete_sha256=sha(p/'complete.json'),snapshot_sha256=sha(p/'snapshots/53000.json'),metrics_sha256=sha(m/'metrics.json'))
for label in ['N01','N07']:
    p=V3/'runs'/f'{label}_fused';m=V3/'results'/f'{label}_fused_33000';raw=V3/'cache/raw'/f'{label}_fused_33000'
    reuse[label+'_fused']=dict(run=str(p),results=str(m),raw=str(raw),complete_sha256=sha(p/'complete.json'),snapshot_sha256=sha(p/'snapshots/33000.json'),metrics_sha256=sha(m/'metrics.json'))
plan=dict(schema='cross8_frozen_A_fusion_v1',created_at=datetime.now(timezone.utc).isoformat(),authorization='Latest explicit user expansion to all eight existing subjects overrides earlier bounded-round identity cap; previous PLAN files remain unchanged.',request_sha256=sha(NEW/'REQUEST.txt'),parent_plan_sha256=sha(V3/'PLAN.json'),parent_fusion_sha256=sha(V3/'FUSION_PLAN.json'),labels=labels,main_labels=labels[:-1],reserve_labels=['N11'],failed_frontend_not_replaced=['N05','N09','N10'],new_A_labels=new_A,new_fused_labels=new_fused,cross_person=cross,reuse=reuse,evaluation=prior['evaluation'],fusion_quality_gates=json.loads((V3/'FUSION_PLAN.json').read_text())['success'],seed=42,scope='Frozen A52999 and fused32999 pair per legal identity, no color/boundary/backend/seed expansion. Existing61/121 frame inputs retained.',disk=dict(reserve_GiB=35,minimum_launch_free_GiB=90,estimated_new_persistent_GiB=35,rolling_checkpoint_transient_GiB=16,policy='Do not delete historical assets. Pause admission below reserve plus per-job transient headroom.'),runtime=dict(environment=str(Path('/mnt/ojc/miniconda3/envs/grid4d')),tmux='fdgrid_cross8',gpus=list(range(8)),one_owned_job_per_gpu=True,sharing_other_jobs=True,minimum_gpu_free_MiB=16000),implementation={name:sha(NEW/'scripts'/name) for name in copied})
plan['evaluation']['cross_person_cohort']=labels[:-1];plan['evaluation']['N11']='Reserve separate; main7 and optional descriptive all8 denominators explicit'
save(NEW/'PLAN.json',plan);(NEW/'plan_hash.txt').write_text(sha(NEW/'PLAN.json')+'\n')
fp=json.loads((V3/'FUSION_PLAN.json').read_text());fp.update(schema='cross8_fusion_replication',parent_plan_sha256=sha(NEW/'PLAN.json'),labels=new_fused,configs=fused,basis='User explicitly requested all eight subjects after initial two-person fusion result.',scope='Six additional identities, same frozen alpha ramp and LR. Reuse N01/N07 without retraining.',no_extensions='No extra seeds, alpha schedules, color fitting or other backends in this expansion.')
save(NEW/'FUSION_PLAN.json',fp);(NEW/'fusion_plan_hash.txt').write_text(sha(NEW/'FUSION_PLAN.json')+'\n')
env=(V3/'scripts/env.sh').read_text().replace(str(V3),str(NEW));(NEW/'scripts/env.sh').write_text(env)
save(NEW/'STATUS.json',dict(state='plan_frozen',new_training_branches=10,reused_terminal_assets=6,target_terminal_pairs=8))
print('Frozen cross8 expansion:',NEW,flush=True)
