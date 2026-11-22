"""Build a small, explicitly sampled RGBA bridge from validated official outputs."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent/'repo'
LABEL = os.environ.get('FDANYONE_BRIDGE_LABEL', 'official_sample')
assert LABEL in ('official_sample','waymo_sample')
SOURCE = Path(os.environ.get('FDANYONE_BRIDGE_SOURCE', str(ROOT.parent/'outputs/turbo')))
sys.path.insert(0,str(REPO))


def worker(rank):
    import av
    import numpy as np
    from PIL import Image
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.assets import resolve_foreground_model
    source=SOURCE
    destination=ROOT/'bridge'/LABEL
    fps=json.loads((source/'metadata.json').read_text())['output']['fps']
    from fractions import Fraction
    views=list(range(rank*8,(rank+1)*8))
    picked=[]; provenance=[]
    for view in views:
        with av.open(str(source/'videos'/f'{view:02d}.mp4')) as video:
            video.streams.video[0].thread_count=1
            for idx,frame in enumerate(video.decode(video=0)):
                if idx in (0,60,120):
                    picked.append(frame.to_ndarray(format='rgb24'))
                    provenance.append((view,idx))
    masks=predict_foreground_masks(tuple(picked),resolve_foreground_model(REPO/'models'),'cuda:0')
    records=[]
    for rgb,mask,(view,idx) in zip(picked,masks,provenance,strict=True):
        im=Image.fromarray(rgb).resize((176,320),Image.Resampling.LANCZOS)
        alpha=Image.fromarray(mask).resize((176,320),Image.Resampling.BILINEAR)
        im.putalpha(alpha)
        folder=destination/'images'/f'cam{view:02d}';folder.mkdir(parents=True,exist_ok=True)
        im.save(folder/f'{idx//60:06d}.png')
        records.append({'view':view,'source_frame':idx,'bridge_frame':idx//60,'seconds':float(idx/Fraction(fps)),'alpha_foreground_fraction':float((np.array(alpha)>127).mean())})
    (ROOT/'bridge'/f'{LABEL}_masks_rank{rank}.json').write_text(json.dumps({'physical_gpu':rank+5,'records':records},indent=2)+'\n')
    print(f'masks rank {rank} complete: {len(records)} frames',flush=True)


async def launch():
    async def run(rank):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank+5)
        with (ROOT/'logs'/f'{LABEL}-masks-{rank}.log').open('w') as log:
            p=await asyncio.create_subprocess_exec(sys.executable,__file__,'--rank',str(rank),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            code=await p.wait()
        if code:raise RuntimeError(f'Mask worker {rank} exited {code}')
    await asyncio.gather(*map(run,range(3)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int);args=parser.parse_args()
    if args.rank is None:asyncio.run(launch())
    else:worker(args.rank)
