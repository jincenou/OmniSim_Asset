"""Finish required read-only mechanism/recovery evidence after immutable assets."""
import asyncio
import fcntl
import json
import os
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]


async def main():
    labels = json.loads((NEW/'PLAN.json').read_text())['labels']
    failed = {}
    while True:
        for label in labels:
            if label in failed or (NEW/'research'/f'{label}_terminal_mechanism.json').exists():
                continue
            if not (NEW/'assets'/f'{label}_R3/replay_validation.json').exists():
                continue
            p = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--query-gpu=index,uuid,memory.free',
                '--format=csv,noheader,nounits', stdout=asyncio.subprocess.PIPE)
            out, _ = await p.communicate()
            assert p.returncode == 0
            gpus = [line.split(',') for line in out.decode().splitlines()]
            for index, uuid, free in sorted(gpus, key=lambda x: -float(x[2])):
                if float(free) < 8192:
                    continue
                lease = (NEW/'locks'/f'gpu{int(index)}.lock').open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = uuid.strip()
                with (NEW/'logs'/f'{label}_terminal_mechanism.log').open('x') as log:
                    process = await asyncio.create_subprocess_exec(
                        'bash', str(NEW/'scripts/run.sh'), 'audit_mechanism',
                        '--label', label, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
                    print('mechanism', label, index.strip(), process.pid, flush=True)
                    code = await process.wait()
                lease.close()
                if code:
                    failed[label] = code
                break
        done = [l for l in labels if (NEW/'research'/f'{l}_terminal_mechanism.json').exists()]
        state = dict(pid=os.getpid(), completed=done, failed=failed,
                     pending=[l for l in labels if l not in done and l not in failed])
        tmp = NEW/'AUDIT_STATUS.writing'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(NEW/'AUDIT_STATUS.json')
        if len(done) + len(failed) == len(labels) and (NEW/'training_queue.complete.json').exists():
            with (NEW/'logs/final_training_audit.log').open('x') as log:
                process = await asyncio.create_subprocess_exec(
                    'bash', str(NEW/'scripts/run.sh'), 'audit_completed',
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                code = await process.wait()
            assert code == 0, 'See final_training_audit.log'
            (NEW/'audit_queue.complete.json').write_text(json.dumps(
                dict(mechanism_failed=failed, training_audit_process_exit=code,
                     goal_complete=False, visual_review_pending=True))+'\n')
            return
        await asyncio.sleep(30)


if __name__ == '__main__':
    asyncio.run(main())
