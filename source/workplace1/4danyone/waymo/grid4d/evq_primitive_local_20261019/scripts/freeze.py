import json,hashlib,datetime,shutil,difflib
from pathlib import Path
N=Path(__file__).resolve().parents[1];O=N.parent/'early_visual_query_20261019'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (N/'PLAN.json').exists()
reports={}
for label in ['N06','N11']:
 p=N/'research'/f'preflight_{label}';d=json.loads((p/'report.json').read_text());r=json.loads((p/'asset/replay_validation.json').read_text());assert d['passed'] and r['passed'];reports[label]=dict(report=str(p/'report.json'),sha=sha(p/'report.json'),replay=str(p/'asset/replay_validation.json'),replay_sha=sha(p/'asset/replay_validation.json'),technical_updates=d['technical_updates'])
assert sum(r['technical_updates'] for r in reports.values())<=200
cost=json.loads((N/'research/chunk_cost_probe.json').read_text());p=json.loads((N/'PLAN_DRAFT.json').read_text());p['model']['local_chunk']=cost['selected'];p['state']='frozen_before_training';p['frozen_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p['preflight']=reports;p['resource']['estimated_total_new_GiB']=35;p['resource']['free_GiB_at_freeze']=shutil.disk_usage(N).free/2**30
# Immutable training/runtime set; evaluation/report scripts may be added without changing optimizer behavior.
names=['common.py','evq.py','runtime.py','render_evq.py','update.py','train.py','assets.py','export.py'];p['implementation_sha256']={f'scripts/{n}':sha(N/'scripts'/n) for n in names}
p['source_integrity']={str(x):sha(x) for x in [O/'PLAN.json',O/'FINDINGS.md',N.parent/'evq_functional_diagnosis_20261019/FINDINGS.md',N.parent/'evq_functional_diagnosis_20261019/decision.json']}
for label in p['labels']:
 x=O/'runs'/f'{label}_warmup/recovery_2999.pt';p['inputs'][label]['warmup']=str(x);p['inputs'][label]['warmup_sha256']=sha(x)
 for name in ['schedule.json','anchors.npz','base_config.json','cameras.json']:p['inputs'][label][name+'_actual_sha256']=sha(N/'data'/label/name)
# Copied expanded-ROI hashes were relative to old experiment and remain exact.
for name,h in p['evaluation']['expanded_ROI_sha256'].items():assert sha(N/name)==h
(N/'PLAN.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n');(N/'research/preflight.json').write_text(json.dumps(dict(passed=True,reports=reports,total_technical_updates=sum(r['technical_updates'] for r in reports.values()),chunk_cost=cost,old_global_rasterizer_fd='previous failure retained;no threshold change',initial_footprint_check_failure='N06 before updates:invalid out-of-frustum float conversion;rechecked valid token pixel mapping in double,invalid token null mask separately tested',no_quality_based_selection=True),ensure_ascii=False,indent=2)+'\n')
patch=[]
for f in sorted((N/'scripts').glob('*.py')):
 a=(O/'scripts'/f.name).read_text().splitlines(True) if (O/'scripts'/f.name).exists() else [];b=f.read_text().splitlines(True);patch.extend(difflib.unified_diff(a,b,fromfile=f'a/scripts/{f.name}',tofile=f'b/scripts/{f.name}'))
(N/'research/source.patch').write_text(''.join(patch));print('FROZEN chunk',cost['selected'],'technical updates',sum(r['technical_updates'] for r in reports.values()))
