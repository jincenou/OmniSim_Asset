"""Start decode/tracking as each prescribed sequence is CRC-complete."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=dict(state='waiting_or_preparing_sources',pid=os.getpid(),started=time.time(),jobs={})
CPU=asyncio.Semaphore(2)


def save():
    path=ROOT/'sources/preparation_status.json';tmp=path.with_suffix('.tmp')
    STATE['updated']=time.time();tmp.write_text(json.dumps(STATE,indent=2));tmp.replace(path)


async def run(source,preferred):
    all_entries=json.loads((ROOT/'sources/mot_selected_zip_entries.json').read_text())
    expected={row['path'] for row in all_entries if row['sequence']==source}
    STATE['jobs'][source]=dict(state='waiting_crc_complete');save()
    while True:
        ready=set()
        for path in (ROOT/'sources').glob('mot_range_*_receipt.json'):
            ready.update(row['path'] for row in json.loads(path.read_text())['files'] if row['sequence']==source)
        if ready==expected:break
        download=json.loads((ROOT/'sources/mot_ranges_status.json').read_text())
        if download['state']=='failed':raise RuntimeError(f'{source}: range download failed')
        pid=download['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError(f'{source}: downloader exited before required files complete')
        await asyncio.sleep(20)
    async with CPU:
        with (ROOT/'logs'/f'inspect_{source}.log').open('x') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scripts/inspect_mot.py'),source,
                stdout=log,stderr=log,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            STATE['jobs'][source].update(state='decoding',pid=proc.pid);save()
            code=await proc.wait();assert code==0,f'{source}: decode failed {code}'
    path=json.loads((ROOT/'sources'/f'{source}_decode.json').read_text())['path']
    with (ROOT/'logs'/f'track_{source}_controller.log').open('x') as log:
        proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scripts/run_leased.py'),
            '--name','track_'+source.replace('-','_'),'--preferred-gpu',str(preferred),
            sys.executable,'-u',str(ROOT/'scripts/track_video.py'),source,path,stdout=log,stderr=log,cwd=ROOT)
        STATE['jobs'][source].update(state='tracking_or_waiting_gpu',pid=proc.pid);save()
        code=await proc.wait();assert code==0,f'{source}: tracking failed {code}'
    STATE['jobs'][source].update(state='tracked_candidate_selection_pending',finished=time.time());save()


async def main():
    save()
    sequences=['MOT16-09','MOT16-02','MOT16-10']
    results=await asyncio.gather(*(run(s,g) for s,g in zip(sequences,[5,6,7],strict=True)),return_exceptions=True)
    failures={s:repr(r) for s,r in zip(sequences,results,strict=True) if isinstance(r,BaseException)}
    STATE.update(state='failed' if failures else 'all_sources_tracked_selection_pending',failures=failures,finished=time.time());save()
    if failures:raise RuntimeError(failures)


if __name__=='__main__':asyncio.run(main())
