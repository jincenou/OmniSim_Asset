"""Local OpenAI-compatible async dispatcher and GPU-lease-aware Qwen replicas."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'vlm_cluster'
MODEL='Qwen/Qwen3-VL-8B-Instruct'
PORTS=(8083,8084)
MEMORY_UTILIZATION=float(os.environ.get('ASSET_VLM_MEMORY_UTILIZATION','0.90'))
if not 0.1 <= MEMORY_UTILIZATION <= 0.90:
    raise ValueError('Invalid ASSET_VLM_MEMORY_UTILIZATION')


def write(name,data):
    STATE.mkdir(exist_ok=True)
    target=STATE/name
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(target)


async def replica(gpu,port):
    name=f'replica_gpu{gpu}.json'
    lock=Path('/mnt/ojc/workplace1/asset_library/.gpu_locks')/f'gpu{gpu}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        while True:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                write(name,dict(status='waiting_gpu_lease',manager_pid=os.getpid(),gpu=gpu,port=port,updated=time.time()))
                await asyncio.sleep(5)
        while True:
            proc=await asyncio.create_subprocess_exec('nvidia-smi','-i',str(gpu),
                '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
            data,_=await proc.communicate()
            if proc.returncode:raise RuntimeError('GPU memory query failed')
            free,total=map(float,data.decode().strip().split(','))
            if free>=(MEMORY_UTILIZATION+.02)*total:break
            write(name,dict(status='waiting_free_memory',manager_pid=os.getpid(),gpu=gpu,port=port,free_MiB=free,updated=time.time()))
            await asyncio.sleep(5)
        env=dict(os.environ,PYTHON=sys.executable,MODEL_PATH='/mnt/ojc/models/qwen3_vl/Qwen3-VL-8B-Instruct',
            VLM_GPU=str(gpu),VLM_PORT=str(port),VLM_CUDA_HOME='/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (STATE/f'gpu{gpu}_port{port}.log').open('ab') as log:
            child=await asyncio.create_subprocess_exec('bash','/mnt/ojc/DriveLab/scripts/start_vlm.sh',
                '--gpu-memory-utilization',str(MEMORY_UTILIZATION),env=env,
                stdout=log,stderr=asyncio.subprocess.STDOUT)
            write(name,dict(status='service_process_started',manager_pid=os.getpid(),service_pid=child.pid,
                gpu=gpu,port=port,started=time.time(),memory_utilization=MEMORY_UTILIZATION))
            loop=asyncio.get_running_loop()
            def stop():
                if child.returncode is None:child.send_signal(signal.SIGTERM)
            loop.add_signal_handler(signal.SIGTERM,stop)
            loop.add_signal_handler(signal.SIGINT,stop)
            code=await child.wait()
            write(name,dict(status='service_exited',manager_pid=os.getpid(),service_pid=child.pid,
                gpu=gpu,port=port,exit_code=code,updated=time.time()))
            if code:raise RuntimeError(f'Replica exited with code {code}')


def app():
    from fastapi import FastAPI,Request
    from fastapi.responses import JSONResponse,StreamingResponse
    from openai import AsyncOpenAI,APIStatusError
    clients={p:AsyncOpenAI(base_url=f'http://127.0.0.1:{p}/v1',api_key='EMPTY',timeout=180,max_retries=0) for p in PORTS}
    state={p:dict(healthy=False,active=0,success=0,failed=0,started=0,total_seconds=0.) for p in PORTS}
    lock=asyncio.Lock()

    async def probe(port):
        try:
            models=await asyncio.wait_for(clients[port].models.list(),5)
            if MODEL not in [m.id for m in models.data]:raise ValueError('Served model mismatch')
            state[port].update(healthy=True,last_health=time.time(),health_error=None)
        except Exception as exc:
            state[port].update(healthy=False,last_health=time.time(),health_error=f'{type(exc).__name__}: {exc}')

    async def watch():
        while True:
            await asyncio.gather(*(probe(p) for p in PORTS))
            write('dispatcher_status.json',dict(pid=os.getpid(),updated=time.time(),backends=state,
                policy='least active healthy replica; unchanged OpenAI request body; no fabricated response'))
            await asyncio.sleep(5)

    @asynccontextmanager
    async def lifespan(server):
        await asyncio.gather(*(probe(p) for p in PORTS))
        task=asyncio.create_task(watch())
        try:yield
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            await asyncio.gather(*(c.close() for c in clients.values()))

    server=FastAPI(lifespan=lifespan)

    @server.get('/health')
    async def health():
        return JSONResponse(dict(backends=state),status_code=200 if any(v['healthy'] for v in state.values()) else 503)

    @server.get('/v1/models')
    async def models():
        if not any(v['healthy'] for v in state.values()):return JSONResponse({'error':{'message':'No healthy Qwen replica'}},status_code=503)
        return {'object':'list','data':[{'id':MODEL,'object':'model','created':0,'owned_by':'local'}]}

    @server.post('/v1/chat/completions')
    async def completion(request:Request):
        body=await request.json()
        if body.get('model')!=MODEL or not isinstance(body.get('messages'),list):
            return JSONResponse({'error':{'message':'Expected configured model and messages array'}},status_code=400)
        async with lock:
            available=[p for p in PORTS if state[p]['healthy']]
            if not available:return JSONResponse({'error':{'message':'No healthy Qwen replica'}},status_code=503)
            port=min(available,key=lambda p:(state[p]['active'],state[p]['started'],p))
            state[port]['active']+=1;state[port]['started']+=1
        started=time.monotonic()
        stream=bool(body.pop('stream',False));model=body.pop('model');messages=body.pop('messages')
        async def finish(success):
            async with lock:
                state[port]['active']-=1
                state[port]['success' if success else 'failed']+=1
                state[port]['total_seconds']+=time.monotonic()-started
        try:
            result=await clients[port].chat.completions.create(model=model,messages=messages,
                stream=stream,extra_body=body)
        except BaseException as exc:
            await finish(False)
            if isinstance(exc,asyncio.CancelledError):raise
            if isinstance(exc,APIStatusError):
                return JSONResponse({'error':exc.body},status_code=exc.status_code)
            return JSONResponse({'error':{'message':f'Qwen replica {port}: {type(exc).__name__}: {exc}'}},status_code=502)
        if not stream:
            await finish(True)
            return JSONResponse(result.model_dump(mode='json'),headers={'X-Qwen-Backend':str(port)})
        async def events():
            success=False
            try:
                async for chunk in result:
                    yield 'data: '+json.dumps(chunk.model_dump(mode='json'))+'\n\n'
                yield 'data: [DONE]\n\n';success=True
            finally:
                await result.close();await finish(success)
        return StreamingResponse(events(),media_type='text/event-stream',headers={'X-Qwen-Backend':str(port)})
    return server


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replica',nargs=2,type=int,metavar=('GPU','PORT'))
    parser.add_argument('--port',type=int,default=8082);parser.add_argument('--wait-bind',action='store_true');args=parser.parse_args()
    if args.replica:asyncio.run(replica(*args.replica))
    else:
        import uvicorn
        if args.wait_bind:
            import socket
            async def serve_when_available():
                sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                while True:
                    try:sock.bind(('127.0.0.1',args.port));break
                    except OSError as exc:
                        if exc.errno!=98:raise
                        await asyncio.sleep(.05)
                sock.listen(1024);sock.setblocking(False)
                server=uvicorn.Server(uvicorn.Config(app(),host='127.0.0.1',port=args.port,timeout_graceful_shutdown=180))
                await server.serve(sockets=[sock])
            asyncio.run(serve_when_available())
        else:uvicorn.run(app(),host='127.0.0.1',port=args.port,timeout_graceful_shutdown=180)
