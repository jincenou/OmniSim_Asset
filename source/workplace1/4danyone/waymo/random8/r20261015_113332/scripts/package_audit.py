"""Non-destructive resource/configuration and exact parameter-budget accounting."""
from pathlib import Path
import json,hashlib,datetime,os,asyncio
import yaml
R=Path(__file__).resolve().parents[1]
class MetadataLoader(yaml.SafeLoader):pass
MetadataLoader.add_multi_constructor('tag:yaml.org,2002:python/object:',lambda loader,tag,node:loader.construct_mapping(node,deep=True))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');t.replace(p)
def audit(path):
 data=json.loads(path.read_text());label,branch=data['label'],data['branch'];base=R/'samples'/label;ft=branch.startswith('ftgs');model=base/branch if ft else R/'four_dg/runs'/f'{label}_grid{branch[-2:]}';experiment=json.loads((model/'experiment.json').read_text());completion=json.loads((model/('training_complete.json' if ft else 'complete.json')).read_text());asset=json.loads((path.parent/'asset_manifest.json').read_text());snapshot=data.get('snapshot');nominal=data['nominal_total_iterations'];updates=nominal if ft else (32999 if snapshot is None else 3000+snapshot-1)
 files={};names=[asset.get('dynamic_ply'),asset.get('checkpoint')] if ft else list(asset['dynamic_files'])
 for name in names:
  if name:
   p=Path(name);files[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
 if ft:
  historical=Path(experiment['historical_config']);old=yaml.load(historical.read_text(),Loader=MetadataLoader);actual=experiment['full_config'];diff={k:{'historical':v,'actual':actual.get(k)} for k,v in old.items() if v!=actual.get(k)};training_params=64*completion['final_gaussians'];network_params=0
 else:
  old=json.loads((R.parents[1]/'round2/alternatives/4dgaussians/runs/m02_61_recipe30k_v1/experiment.json').read_text());diff={section:{k:{'historical':v,'actual':experiment[section].get(k)} for k,v in old[section].items() if v!=experiment[section].get(k)} for section in ['dataset','hidden','pipeline','optimization']};network_params=completion['network_parameters'];training_params=None
 result={'label':label,'branch':branch,'snapshot':snapshot,'nominal_total_iterations':nominal,'actual_optimizer_updates_represented_by_evaluated_parameters':updates,'checkpoint_actual_updates':nominal if snapshot is not None and not ft else updates,'snapshot_note':'HUST nativepointcloud is saved before the current optimizerstep; fine12k evaluation therefore represents14999actualupdates. The lossless optimizer checkpoint saved afterward represents15000updates. Do not label the native snapshot as exact same-update comparison.' if snapshot is not None and not ft else 'Final HUST skips lastfine optimizerstep per unchanged official predicate;FTGS savedparams are poststep.','files':files,'dynamic_package_bytes':sum(v['bytes'] for v in files.values()),'final_training_resources':{k:completion[k] for k in ['elapsed_seconds','peak_allocated_gib','initial_gaussians','final_gaussians','network_parameters','device'] if k in completion},'training_resource_scope':'Resources measure the fullfinal recipe,not partialbudget snapshot runtime; allocated excludes CUDA context and driver overhead.','deformation_network_parameters':network_params,'FTGS_analytic_Gaussian_parameters':training_params,'config_diff_from_actual_historical':diff,'resolved_configuration':str(model/'experiment.json'),'source_provenance':str(R/'source/random8_changes.json'),'input_manifest_sha256':sha(R/'four_dg/data'/label/'fdanyone.json'),'metric_reference':'generated-heldout reconstruction;not realWaymo novelview GT','visual_review':'pending','production_approved':False,'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};write(path.parent/'resource_and_budget_audit.json',result)
def main():
 success=[];fail=[]
 for path in (R/'samples').glob('*/evaluations/*/results.json'):
  out=path.parent/'resource_and_budget_audit.json'
  if out.exists():success.append(str(out));continue
  try:audit(path);success.append(str(out))
  except Exception as e:fail.append({'path':str(path),'error':f'{type(e).__name__}:{e}'})
 write(R/'research/package_audit_status.json',{'success':success,'failures':fail,'scope':'Configuration/resource/accounting audit;not a substitute for actual render replay or fulltask completion'})
async def watch():
 while True:
  main();s=json.loads((R/'STATUS.json').read_text())
  if s['state'].startswith('all_registered_tasks_terminal'):return
  await asyncio.sleep(120)
if __name__=='__main__':
 import sys
 asyncio.run(watch()) if '--watch' in sys.argv else main()
