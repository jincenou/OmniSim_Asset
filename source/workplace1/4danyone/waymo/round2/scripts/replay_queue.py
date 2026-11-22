"""Run the walking replay on a leased authorized GPU, preserving its exit code."""
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from resource_lease import lease, wait_memory

ROOT = Path(__file__).resolve().parents[1]


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    path = ROOT / 'replay_run.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


async def main():
    state = {'state': 'waiting_for_gpu7_lease', 'controller_pid': os.getpid()}
    save(state)
    async with lease([7]) as descriptors:
        await wait_memory([7], 12000)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_HOME='/mnt/ojc/miniconda3/envs/asset_harvester_2026',
                   TORCH_EXTENSIONS_DIR=str(ROOT.parent / 'reconstruction/deps/torch_extensions'),
                   TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='4', OMP_NUM_THREADS='4')
        env['PATH'] = env['CUDA_HOME'] + '/bin:' + env['PATH']
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        command = [sys.executable, str(ROOT / 'scripts/replay_walking.py')]
        logfile = ROOT / 'logs/replay_render.log'
        with logfile.open('x') as log:
            process = await asyncio.create_subprocess_exec(*command, env=env, stdout=log,
                stderr=asyncio.subprocess.STDOUT, pass_fds=descriptors)
            state.update(state='rendering', pid=process.pid, command=command, log=str(logfile), gpu=7)
            save(state)
            result = await process.wait()
        state.update(state='rendered_and_decoded_visual_review_pending' if result == 0 else 'failed', exit_code=result)
        save(state)
        return result


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
