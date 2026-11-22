"""Wait for the existing queue, then verify deliveries without launching inference."""
import asyncio
import json
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]

async def main():
    while (R / 'reserve_pending.flag').exists():
        proc = await asyncio.create_subprocess_exec('tmux', '-L', 'fdscene20', 'has-session', '-t', 'reserve')
        assert await proc.wait() == 0, 'Queue owner missing; inspect rather than restart automatically'
        await asyncio.sleep(30)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in ('compact_failures.py', 'audit_delivery.py'):
        proc = await asyncio.create_subprocess_exec(sys.executable, str(R / 'scripts' / name), env=env)
        assert await proc.wait() == 0, name
    audit = json.loads((R / 'research/delivery_audit.json').read_text())
    print(json.dumps({'state': 'technical_audit_done', 'people': audit['completed_people'],
                      'remaining': 'Review last generated contact and finalize report; no new training scheduled'}), flush=True)

if __name__ == '__main__':
    asyncio.run(main())
