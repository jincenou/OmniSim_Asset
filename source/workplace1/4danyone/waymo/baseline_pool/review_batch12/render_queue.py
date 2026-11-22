import asyncio,json,hashlib,time,os
from pathlib import Path
R=Path(__file__).resolve().parent; P=R.parent
PY='/mnt/ojc/miniconda3/envs/4danyone/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def main():
    rows=json.loads((R/'selection.json').read_text())['selected'];sem=asyncio.Semaphore(2)
    async def one(row):
        async with sem:
            dest=R/row['review_id'];manifest=dest/'manifest.json'
            if not manifest.exists():
                if dest.exists():raise RuntimeError(f'Partial output requires inspection: {dest}')
                with (R/(row['review_id']+'.log')).open('x') as log:
                    child=await asyncio.create_subprocess_exec(PY,str(P/'render_review_contacts.py'),row['package'],str(dest),stdout=log,stderr=log)
                    await child.communicate()
                if child.returncode:raise RuntimeError((row['review_id'],child.returncode))
            data=json.loads(manifest.read_text())
            for evidence in data['evidence']:
                assert sha(Path(evidence['source']))==evidence['source_sha256']
                assert sha(Path(evidence['sheet']))==evidence['sheet_sha256']
            print(row['review_id'],'complete',flush=True)
            return row['review_id']
    complete=await asyncio.gather(*map(one,rows));(R/'render_complete.json').write_text(json.dumps(dict(at=time.time(),complete=complete,pid=os.getpid()))+'\n')
if __name__=='__main__':asyncio.run(main())
