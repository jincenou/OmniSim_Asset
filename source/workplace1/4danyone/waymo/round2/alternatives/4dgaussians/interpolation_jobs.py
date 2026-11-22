"""Two async interpolation diagnostics share the same resource controller."""
import asyncio
from pathlib import Path
import temporal_recipe_jobs as controller
from resource_lease import lease, wait_memory

A = Path(__file__).resolve().parent


async def one(label, gpu):
    async with lease([gpu]) as handles:
        await wait_memory([gpu], 10000)
        import os
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            TORCH_HOME=str(A / 'cache/torch'), MPLCONFIGDIR=str(A / 'cache/matplotlib'))
        await controller.command('interpolation_' + label, label, gpu,
            [A / 'temporal_interpolation.py', '--label', label], handles, environment)


async def main():
    controller.STATE_PATH = A / 'interpolation_queue.json'
    assert not controller.STATE_PATH.exists(), 'Do not restart running or completed interpolation'
    controller.STATE['state'] = 'two_interpolation_diagnostics_running'
    controller.write()
    results = await asyncio.gather(one('m02_short61_24', 6), one('r07_short61_24', 7), return_exceptions=True)
    await controller.gate_results(results, 'two_temporal_interpolation_diagnostics_complete')


if __name__ == '__main__':
    asyncio.run(main())
