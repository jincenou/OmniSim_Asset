"""After the current SplatAD batch finishes, run the staged web vehicle H wave."""
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT.parent / 'scene_production_20261031'
VEHICLES = ROOT
SPLATAD = Path('/mnt/ojc/workplace1/vehicle_model_comparison/state/splatad_resume_current_batch.json')
PYTHON = '/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python'
STATUS = ROOT / 'handoff_status.json'

def write(value):
    tmp = STATUS.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(STATUS)

def all_gpus_free():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                       capture_output=True,text=True,check=True)
    return not p.stdout.strip()

def launch(name, command, extra_env=None):
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    handle = (logs / (name+'.log')).open('ab',buffering=0)
    env = dict(os.environ, PYTHONUNBUFFERED='1', **(extra_env or {}))
    process = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                               env=env,start_new_session=True)
    return process,handle

def stop(processes):
    for process, _ in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline=time.time()+180
    for process, handle in processes:
        try:process.wait(timeout=max(1,deadline-time.time()))
        except subprocess.TimeoutExpired:process.kill();process.wait()
        handle.close()

def main():
    state={'status':'waiting_for_splatad_current_batch','started_at':time.time(),
           'splatad_status_file':str(SPLATAD),'staging_status_file':str(ROOT/'staging_status.json')}
    write(state)
    while not all_gpus_free():
        state.update(status='waiting_for_gpu_release');write(state);time.sleep(5)
    state.update(status='starting_qwen',handoff_at=time.time())
    write(state)
    processes=[]
    try:
        for gpu,port in [(4,8083),(7,8084)]:
            process,handle=launch(f'qwen_gpu{gpu}',[PYTHON,str(PRODUCTION/'vlm_cluster.py'),
                                    '--replica',str(gpu),str(port)],
                                  {'ASSET_VLM_MEMORY_UTILIZATION':'0.65'})
            processes.append((process,handle))
        process,handle=launch('qwen_dispatcher',[PYTHON,str(PRODUCTION/'vlm_cluster.py'),
                                                   '--port','8082'])
        processes.append((process,handle))
        deadline=time.time()+600
        while time.time()<deadline:
            if any(p.poll() is not None for p,_ in processes):
                raise RuntimeError('Qwen service exited before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8082/v1/models',timeout=5) as response:
                    data=json.loads(response.read())
                if data.get('data'):
                    backend=PRODUCTION/'vlm_cluster/dispatcher_status.json'
                    if backend.exists():
                        status=json.loads(backend.read_text())
                        if sum(int(v.get('healthy',False)) for v in status['backends'].values())==2:break
            except Exception:pass
            time.sleep(5)
        else:raise TimeoutError('Two Qwen replicas did not become healthy in 10 minutes')
        state.update(status='running_rider_H',qwen_replicas=2,qwen_gpus=[4,7],
                     compute_gpus=[0,1,2,3,5,6],qwen_ready_at=time.time())
        write(state)
        while not (ROOT/'READY').exists():
            time.sleep(5)
        worker,work_log=launch('riders_H',[PYTHON,'-u',str(VEHICLES/'run.py')])
        state['vehicle_worker_pid']=worker.pid
        write(state)
        code=worker.wait()
        work_log.close()
        state.update(status='rider_wave_completed' if code==0 else 'rider_wave_failed',
                     vehicle_exit_code=code,vehicle_finished_at=time.time())
        write(state)
    except BaseException as exc:
        state.update(status='handoff_failed',error=repr(exc),failed_at=time.time())
        write(state)
        raise
    finally:
        stop(processes)

if __name__=='__main__':main()
