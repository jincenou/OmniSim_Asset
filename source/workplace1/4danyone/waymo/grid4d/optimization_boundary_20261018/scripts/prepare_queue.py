import asyncio
import fcntl
import json
import sys
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]


async def one(label):
    target=NEW/'data'/label/'bands/complete.json'
    if target.exists():return dict(label=label,state='existing_complete')
    with (NEW/'logs'/f'prepare_{label}.log').open('a') as log:
        child=await asyncio.create_subprocess_exec(sys.executable,str(NEW/'scripts/prepare_bands.py'),'--label',label,stdout=log,stderr=asyncio.subprocess.STDOUT)
        code=await child.wait()
    return dict(label=label,pid=child.pid,exit_code=code,state='passed' if code==0 else 'failed')


async def main():
    with (NEW/'logs/prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=await asyncio.gather(one('N04'),one('N11'))
        (NEW/'research/preparation.json').write_text(json.dumps(results,indent=2)+'\n')
        assert all(r['state'] in ['passed','existing_complete'] for r in results),results


if __name__=='__main__':asyncio.run(main())
