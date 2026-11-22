import asyncio, json, os, fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[1]
async def main():
    with (R/'single_time_after.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=json.loads((R/'single_time_queue.json').read_text())
            if state['state']=='single_time_diagnostics_complete_review_pending':break
            if state['state']=='completed_with_failures':raise RuntimeError('Single-time diagnostic failed; inspect original logs')
            os.kill(state['pid'],0)
            await asyncio.sleep(30)
        process=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/4danyone-ftgs/bin/python',str(R/'scripts/single_time_report.py'))
        code=await process.wait()
        if code:raise RuntimeError('Sampling audit or report failed')
if __name__=='__main__':asyncio.run(main())
