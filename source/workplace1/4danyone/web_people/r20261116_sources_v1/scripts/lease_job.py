"""Isolated stages using both maintained and historical GPU lease namespaces."""
from pathlib import Path
import argparse, fcntl, json, os, subprocess, time
ROOT=Path(__file__).resolve().parents[1]
LOCK_ROOTS=[Path('/tmp/drivelab-gpu-locks'),Path('/mnt/ojc/workplace1/4danyone/waymo/gpu_leases_20261019')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--min-free-mib',type=int,default=12000)
    p.add_argument('--gpus',default='4,5,6,7,0,1,2,3')
    p.add_argument('--env',default='4danyone');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.name.replace('_','').isalnum() and a.command
    from runtime_env import runtime
    python,env=runtime(a.env)
    for d in LOCK_ROOTS:d.mkdir(exist_ok=True,parents=True)
    state=dict(state='waiting_gpu',controller_pid=os.getpid(),started=time.time(),command=a.command)
    status=ROOT/'logs'/f'{a.name}_process.json'
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status)
    save();handles=[]
    with (ROOT/'logs'/f'{a.name}.lock').open('a+') as joblock:
        fcntl.flock(joblock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not handles:
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            rows={int(s[0]):(s[1],int(s[2])) for line in out.splitlines() if (s:=[x.strip() for x in line.split(',')])}
            for gpu in map(int,a.gpus.split(',')):
                uuid,free=rows[gpu]
                if free<a.min_free_mib:continue
                acquired=[]
                try:
                    for directory in LOCK_ROOTS:
                        h=(directory/f'gpu{gpu}.lock').open('a+')
                        try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:h.close();raise
                        acquired.append(h)
                except BlockingIOError:
                    for h in acquired:h.close()
                    continue
                handles=acquired;state.update(physical_gpu=gpu,gpu_uuid=uuid);break
            if not handles:time.sleep(10)
        try:
            env.update(CUDA_VISIBLE_DEVICES=state['gpu_uuid'],FD_EXPECTED_GPU_UUID=state['gpu_uuid'],HUMAN_GPU_UUID=state['gpu_uuid'])
            command=[python if s=='PYTHON' else s for s in a.command]
            with (ROOT/'logs'/f'{a.name}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(state='running',pid=child.pid,launched=time.time());save()
                code=child.wait()
            state.update(state='completed' if code==0 else 'failed',exit_code=code,finished=time.time());save()
            raise SystemExit(code)
        finally:
            for h in handles:h.close()

if __name__=='__main__':main()
