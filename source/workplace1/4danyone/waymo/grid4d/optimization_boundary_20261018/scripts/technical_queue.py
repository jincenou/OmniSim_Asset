import asyncio
import fcntl
import json
import os
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]


async def one(label,uuid):
    with (NEW/'logs'/f'technical_{label}.log').open('a') as log:
        process=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/technical_check.py'),'--label',label,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=asyncio.subprocess.STDOUT)
        code=await process.wait()
    return dict(label=label,pid=process.pid,exit_code=code)


async def main():
    with (NEW/'logs/technical.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=await asyncio.gather(one('N04','GPU-cd4b2342-fba4-2b42-beb9-60b698debc2d'),one('N11','GPU-32040e1a-df6d-0776-5420-fae47bc11248'))
        (NEW/'research/technical_processes.json').write_text(json.dumps(results,indent=2)+'\n')
        assert all(r['exit_code']==0 for r in results),results
        reports=[json.loads((NEW/'research'/f'{label}_technical.json').read_text()) for label in ['N04','N11']]
        (NEW/'gradient_checks.json').write_text(json.dumps(dict(state='passed',checks=reports),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':asyncio.run(main())
