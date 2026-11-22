import asyncio,fcntl,json,os,shutil,sys,time
from pathlib import Path
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent
UUIDS=['GPU-0db9aa59-0293-a3ea-06a0-395913880fbd','GPU-4e635188-f5de-33f9-6ecb-53555a0a7c9d','GPU-e3be373d-e235-e8d2-76bf-3559bea96b58','GPU-16387ea4-8329-06c9-0c83-c18c0e71310c','GPU-f9cfc0e8-a5ac-7168-2700-4b52fee1e21d','GPU-85bb9093-960d-9196-9cf8-cdecf8fd0765','GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d','GPU-32040e1a-df6d-0776-5420-fae47bc11248']
def save(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
async def acquire(gpu):
    while True:
        p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        stdout,_=await p.communicate();assert p.returncode==0
        memory={r.split(',')[0].strip():int(r.split(',')[1]) for r in stdout.decode().splitlines()}
        if memory[UUIDS[gpu]]>=16000 and shutil.disk_usage(NEW).free>=40*2**30:
            lock=(ROOT/'cache/gpu_locks'/f'gpu{gpu}.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return lock
            except BlockingIOError:lock.close()
        await asyncio.sleep(20)
async def command(state,path,job,gpu,name,script,args):
    assert shutil.disk_usage(NEW).free>=40*2**30
    stage=dict(name=name,state='starting',started_at=time.time());job.setdefault('stages',[]).append(stage);save(path,state)
    logpath=NEW/'logs'/f'{name}.log';env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=UUIDS[gpu]
    with logpath.open('x') as log:
        proc=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts'/script),*args,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
        stage.update(state='running',pid=proc.pid,log=str(logpath));save(path,state);code=await proc.wait()
    stage.update(state='complete' if code==0 else 'failed',exit_code=code,ended_at=time.time());save(path,state)
    if code:raise RuntimeError(f'{name} exited{code}: {logpath}')
