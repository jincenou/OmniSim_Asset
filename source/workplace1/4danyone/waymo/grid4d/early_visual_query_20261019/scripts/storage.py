"""Lossless sparse checkpoint storage; required stage recoveries are never deleted."""
import asyncio,hashlib,json,os
from pathlib import Path
NEW=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
async def main():
    target=NEW/'research/storage_ledger.json';ledger=json.loads(target.read_text()) if target.exists() else {};done=set(ledger)
    while True:
        for snapshot in sorted((NEW/'runs').glob('*/snapshot_*.json')):
            step=int(snapshot.stem.split('_')[1]);p=snapshot.parent/f'recovery_{step}.pt';key=str(p.relative_to(NEW))
            if key in done:continue
            data=json.loads(snapshot.read_text());before=p.stat();h=sha(p);assert h==data['recovery_sha256']
            proc=await asyncio.create_subprocess_exec('fallocate','--dig-holes',str(p),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE);_,err=await proc.communicate();assert proc.returncode==0,err.decode();after=p.stat();assert sha(p)==h
            ledger[key]=dict(operation='punch_zero_blocks_only',sha256=h,exact_file_bytes_unchanged=True,logical_bytes=after.st_size,allocated_before=before.st_blocks*512,allocated_after=after.st_blocks*512);done.add(key);save(target,ledger)
        for complete in (NEW/'runs').glob('*/training.complete.json'):
            run=complete.parent;step=json.loads(complete.read_text())['step'];terminal=run/f'recovery_{step}.pt';snap=json.loads((run/f'snapshot_{step}.json').read_text());temp=run/'recovery_latest.pt';key=str(temp.relative_to(NEW))
            if temp.exists() and key not in done:
                assert sha(terminal)==snap['recovery_sha256'];s=temp.stat();ledger[key]=dict(operation='remove_superseded_rolling_recovery',sha256=sha(temp),bytes=s.st_size,required_terminal_retained=str(terminal),required_terminal_sha256=snap['recovery_sha256']);temp.unlink();done.add(key);save(target,ledger)
        if (NEW/'training_queue.complete.json').exists():return
        await asyncio.sleep(30)
if __name__=='__main__':asyncio.run(main())
