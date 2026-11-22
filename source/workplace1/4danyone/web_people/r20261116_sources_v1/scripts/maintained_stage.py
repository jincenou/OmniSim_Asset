"""Invoke the isolated maintained implementation with explicit batch paths."""
from pathlib import Path
import os,sys,runpy
ROOT=Path(__file__).resolve().parents[1]
module=sys.argv[1]
assert module in ('frontend','dataset','reconstruction','evaluation','assets','batch','prepare')
os.environ.update(HUMAN_RUN_ROOT=str(ROOT),HUMAN_RUNTIME_CONFIG='/mnt/ojc/DriveLab/human/config/runtime.local.json',
 HUMAN_GVHMR_ROOT='/mnt/ojc/workplace1/4danyone/repo/third_party/GVHMR',
 HUMAN_ASSET_ROOT='/dev/shm/ojc_webhuman_r20261116_sources_v1/assets')
sys.path.insert(0,str(ROOT/'code'))
sys.argv=[str(ROOT/'code'/f'{module}.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
