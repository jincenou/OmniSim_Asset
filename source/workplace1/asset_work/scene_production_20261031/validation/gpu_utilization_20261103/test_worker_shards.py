import ast,asyncio,json,tempfile
from pathlib import Path
R=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d));return p
async def exercise():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);write(root/'vlm_process.json',{'replica_gpus':[4,7]})
  rt=ast.parse((R/'runtime.py').read_text());helper=next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=='reconstruction_gpus')
  ns=dict(ROOT=root,Path=Path,read_json=read);exec(compile(ast.Module(body=[helper],type_ignores=[]),'roles','exec'),ns)
  workers=ns['reconstruction_gpus'](list(range(8)));assert workers==[0,1,2,3,5,6]
  batches=[]
  class Pool:
   def __init__(self,gpus):self.gpus=gpus
   async def acquire(self,stage):return self.gpus[0]
   def release(self,gpu):pass
  async def launch(stage,indices,gpu,path,accept):
   batches.append((stage,list(indices)))
   for i in indices:
    accept(dict(item_index=i,stage=stage,status='succeeded',raw_path='stub',raw_sha256='stub',raw_up_axis='z',gaussian_count=1))
  tree=ast.parse(Path('/mnt/ojc/DriveLab/asset_extraction/generation.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='generate')
  ns2=dict(Path=Path,asyncio=asyncio,read_json=read,write_json=write,GpuPool=Pool,_launch=launch,checked_outputs=lambda r:None,is_memory_failure=lambda r:False)
  exec(compile(ast.Module(body=[node],type_ignores=[]),'dispatch','exec'),ns2)
  jobs=dict(signature='fixture',config={},assets=[dict(item_index=i,obj_id=str(i),route='H',seed=i*1000) for i in range(24)])
  result=await ns2['generate'](jobs,root/'generation',workers)
  for stage in ['flux','H']:
   shards=[ids for s,ids in batches if s==stage];assert len(shards)==6 and all(len(ids)==4 for ids in shards)
   assert sorted(i for ids in shards for i in ids)==list(range(24))
  assert all(r['status']=='generated' and r['seed']==i*1000 for i,r in enumerate(result['assets']))
  assert read(root/'generation/jobs.json')==jobs
  write(root/'vlm_process.json',{'replica_gpus':list(range(8))})
  try:ns['reconstruction_gpus'](list(range(8)))
  except ValueError:pass
  else:raise AssertionError('Must reject empty worker pool')
asyncio.run(exercise());print('PASS: 24 identities scheduled once per stage; 6 workers x 4 jobs; seeds and frozen job bundle unchanged')
