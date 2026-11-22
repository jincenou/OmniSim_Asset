"""Dispatch one dependency-checked stage in its existing dedicated environment."""
import argparse
import json
import os
from pathlib import Path
import sys
from runtime_env import ROOT,runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('id');p.add_argument('--output');args=p.parse_args()
    from prepare_person import selected
    selected(args.id)
    if args.stage in ('accessory_mask','accessory_generate'):assert args.id in ('W02','W04')
    d=ROOT/'backend';scripts=d/'scripts'
    ordinary={
        'prepare':('4danyone',ROOT/'scripts/prepare_person.py',['prepare',args.id]),
        'segment':('4danyone-sam2',ROOT/'scripts/prepare_person.py',['segment',args.id]),
        'motion':('4danyone',ROOT/'scripts/prepare_person.py',['motion',args.id]),
        'accessory_mask':('4danyone-sam2',ROOT/'scripts/accessory_trial.py',['--person',args.id]),
        'accessory_generate':('4danyone',ROOT/'scripts/accessory_generation.py',['generate','--person',args.id]),
        'conditioning':('4danyone',ROOT/'scripts/run_frontend.py',['conditioning',args.id]),
        'generate':('4danyone',ROOT/'scripts/run_frontend.py',['generate',args.id]),
        'validate_generation':('4danyone',ROOT/'scripts/finish_generation.py',[args.id]),
        'generated_preview':('4danyone',ROOT/'scripts/publish_previews.py',['generated',args.id]),
        'comparison':('4danyone',ROOT/'scripts/publish_previews.py',['comparison',args.id]),
        'masks':('4danyone-sam2',scripts/'masks.py',[]),
        'hull':('4danyone-ftgs',scripts/'export_sequence.py',[]),
        'manifest':('4danyone-ftgs',scripts/'export_4dgs.py',['--label',args.id,'--data',str(ROOT/'rgba'/args.id)]),
        'geometry':('4danyone-ftgs',scripts/'audit_geometry.py',[args.id]),
        'configure':('grid4d',scripts/'configure_grid.py',[args.id]),
        'native':('grid4d',scripts/'train_prefix.py',['--config',str(d/'configs'/f'{args.id}_33k.json'),'--run',str(d/'runs'/f'{args.id}_33k')]),
        'alpha':('grid4d',scripts/'train_continuation.py',['--label',args.id,'--stage','alpha']),
        'A':('grid4d',scripts/'train_continuation.py',['--label',args.id,'--stage','A']),
        'package':('4danyone',ROOT/'scripts/package_asset.py',[args.id]),
        'replay':('grid4d',scripts/'replay_asset.py',[args.id]+(['--output',args.output] if args.output else [])),
    }
    name,script,extra=ordinary[args.stage];assert script.is_file()
    python,env=runtime(name)
    env.update(FDANYONE_ASSET_NAME=args.id,FDANYONE_ASSET_SOURCE=str(ROOT/'frontend'/args.id/'outputs/turbo'),
        FDANYONE_DATA=str(ROOT/'rgba'/args.id),FDANYONE_FRAMES='121')
    os.execve(python,[python,'-u',str(script),*extra],env)


if __name__=='__main__':main()
