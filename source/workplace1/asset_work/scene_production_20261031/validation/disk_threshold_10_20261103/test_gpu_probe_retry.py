"""Verify timeout retries cannot grant a GPU from missing/stale memory evidence."""
import ast,asyncio,fcntl,tempfile,time,types
from pathlib import Path
root=Path('/mnt/ojc/workplace1/asset_work/scene_production_20261031')
tree=ast.parse((root/'runtime.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedGpuPool')
async def main():
 with tempfile.TemporaryDirectory() as td:
  events=[];calls=0
  async def probe():
   nonlocal calls
   calls+=1;events.append('probe')
   if calls==1:raise asyncio.TimeoutError()
   return {0:50*1024}
  async def sleep(seconds):events.append(('retry_delay',seconds))
  ns=dict(Path=Path,LIBRARY=Path(td),fcntl=fcntl,time=time,generation=types.SimpleNamespace(gpu_free_mib=probe),asyncio=types.SimpleNamespace(Lock=asyncio.Lock,TimeoutError=asyncio.TimeoutError,sleep=sleep))
  exec(compile(ast.Module(body=[node],type_ignores=[]),'pool','exec'),ns)
  pool=ns['SharedGpuPool']([0]);gpu=await pool.acquire('render')
  assert events==['probe',('retry_delay',5),'probe'] and gpu==0 and list(pool.held)==[0]
  pool.release(gpu);assert not pool.held
  async def broken():raise RuntimeError('not a transient timeout')
  ns['generation'].gpu_free_mib=broken
  try:await pool.acquire('render')
  except RuntimeError as exc:assert str(exc)=='not a transient timeout'
  else:raise AssertionError('Unexpected errors must propagate')
asyncio.run(main());print('PASS: transient timeout retried; fresh memory required; lease released; other errors propagate')
