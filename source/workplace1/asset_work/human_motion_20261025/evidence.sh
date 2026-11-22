#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd /mnt/ojc/workplace1/asset_work/human_motion_20261025/code
exec /mnt/ojc/miniconda3/envs/4danyone/bin/python -u - <<'PYCODE'
import asyncio,sys,time
import motion_experiment as m
async def main():
    sem=asyncio.Semaphore(2)
    async def one(item):
        async with sem:
            identity=item['id']
            code="import motion_experiment as m; from placement import foot_correspondence_probe; import sys; p=m.ASSETS/sys.argv[1]; before=m.inventory(p); foot_correspondence_probe(sys.argv[1]); assert m.inventory(p)==before"
            with (m.LAB/'logs'/f'{identity}_foot_regions.log').open('w') as log:
                p=await asyncio.create_subprocess_exec(sys.executable,'-c',code,identity,stdout=log,stderr=asyncio.subprocess.STDOUT)
                result=await p.wait()
            return dict(identity=identity,exit_code=result)
    results=await asyncio.gather(*map(one,m.frozen_plan()['cohort']))
    m.save(m.LAB/'reports/foot_regions_population.json',dict(updated_at=time.time(),results=results,scope='Read-only conditional geometric diagnostic; not semantic foot tracking approval'))
asyncio.run(main())
PYCODE
