"""Reuse validated environments; put new caches and temporary files in this run."""
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ENVS=Path('/mnt/ojc/miniconda3/envs')
GRID=Path('/mnt/ojc/workplace1/4danyone/waymo/grid4d')


def runtime(name):
    for folder in ['tmp','cache/cuda','cache/matplotlib','cache/numba','cache/extensions','cache/torch']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(ROOT/'tmp'),
        CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(ROOT/'cache/numba'),TORCH_EXTENSIONS_DIR=str(ROOT/'cache/extensions'),
        TORCH_HOME=str(ROOT/'cache/torch'))
    env.pop('PYTHONPATH',None)
    env['PATH']=str(ENVS/name/'bin')+':'+str(ENVS/'4dgaussians/bin')+':'+env.get('PATH','')
    if name=='grid4d':
        private_build=ROOT/'cache/extensions/hashencoder'
        if not private_build.exists():
            shutil.copytree(GRID/'cache/extensions/hashencoder',private_build,
                            ignore=shutil.ignore_patterns('lock'))
        env.update(CUDA_HOME=str(ENVS/'4dgaussians'),CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',
            TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',GRID4D_HASH_BUILD=str(private_build))
    return str(ENVS/name/'bin/python'),env
