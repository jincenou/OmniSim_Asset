"""Export every generated time slot with independently tracked view-specific alpha."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
WAYMO=ROOT.parent
LABEL=os.environ.get('FDANYONE_ASSET_NAME','phone_full')
assert LABEL.replace('_','').isalnum()
SOURCE=Path(os.environ.get('FDANYONE_ASSET_SOURCE',str(WAYMO/'outputs/turbo')))
DEST=ROOT/'data'/LABEL
GPU_IDS=[int(x) for x in os.environ.get('FDANYONE_ASSET_GPU_IDS','4,5').split(',')]
assert GPU_IDS and set(GPU_IDS)<=set(range(4,8))
VIEWS=len(json.loads((SOURCE/'cameras.json').read_text())['cameras'])
PREFIX='full-mask' if LABEL=='phone_full' else f'{LABEL}-mask'


def worker(rank):
    import av
    import numpy as np
    from PIL import Image
    import torch
    sys.path[:0]=[str(WAYMO/'segmentation'),str(WAYMO.parent/'repo')]
    from sam2.build_sam import build_sam2_video_predictor
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.assets import resolve_foreground_model
    torch.cuda.init();assert torch.cuda.device_count()==1
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(WAYMO/'segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
    records=[]
    for view in list(range(VIEWS))[rank::len(GPU_IDS)]:
        folder=DEST/'images'/f'cam{view:02d}';folder.mkdir(parents=True,exist_ok=False)
        temp=DEST/'sam2_input'/f'cam{view:02d}';temp.mkdir(parents=True,exist_ok=False)
        frames=[]
        with av.open(str(SOURCE/'videos'/f'{view:02d}.mp4')) as con:
            con.streams.video[0].thread_count=1
            for i,f in enumerate(con.decode(video=0)):
                assert float(f.pts*f.time_base)==i/25
                im=f.to_image().resize((352,640),Image.Resampling.LANCZOS);frames.append(np.array(im));im.save(temp/f'{i:05d}.jpg',quality=95,subsampling=0)
        assert len(frames)==121
        keyframes=[0,30,60,90,120]
        reference=predict_foreground_masks(tuple(frames[i] for i in keyframes),resolve_foreground_model(WAYMO.parent/'repo/models'),'cuda:0')
        masks=np.empty((121,640,352),np.uint8);seen=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(video_path=str(temp),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=reference[0]>127)
            for idx,ids,logits in predictor.propagate_in_video(state):
                assert ids==[1];masks[idx]=logits[0,0].sigmoid().float().cpu().numpy().__mul__(255).round().astype(np.uint8);seen.append(idx)
        assert seen==list(range(121))
        checks=[]
        for idx,ref in zip(keyframes,reference,strict=True):
            a=masks[idx]>127;b=ref>127;checks.append(float((a&b).sum()/max((a|b).sum(),1)))
        if min(checks)<.65 or np.median(checks)<.8:raise RuntimeError(f'View {view}: propagation mismatch {checks}')
        for i,(rgb,alpha) in enumerate(zip(frames,masks,strict=True)):
            assert .03<float((alpha>127).mean())<.65
            im=Image.fromarray(rgb);im.putalpha(Image.fromarray(alpha));im.save(folder/f'{i:06d}.png')
        records.append({'view':view,'frames':121,'keyframe_sam2_birefnet_iou':checks,'physical_gpu':GPU_IDS[rank],'cuda_uuid':str(torch.cuda.get_device_properties(0).uuid)})
        (ROOT/'research'/f'{PREFIX}-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(records[-1]),flush=True)
        del state,frames,masks,reference;torch.cuda.empty_cache()


async def launch():
    async def run(rank):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(GPU_IDS[rank])
        with (ROOT/'logs'/f'{PREFIX}-rank{rank}.log').open('w') as log:
            p=await asyncio.create_subprocess_exec(sys.executable,__file__,'--rank',str(rank),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
            if await p.wait():raise RuntimeError(f'Full mask worker {rank} failed')
    results=await asyncio.gather(*map(run,range(len(GPU_IDS))),return_exceptions=True)
    failures=[str(result) for result in results if isinstance(result,BaseException)]
    if failures:raise RuntimeError('; '.join(failures))
    (ROOT/'research'/f'{PREFIX}-complete.json').write_text(json.dumps({'source':str(SOURCE),'views':VIEWS,'frames':121,'dimensions':[352,640],'fps':25,'alpha':'SAM2 propagation seeded by per-view BiRefNet; checked against BiRefNet every 30 frames','scope':'generated observations, not Waymo novel-view ground truth'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int);a=p.parse_args()
    if a.rank is None:asyncio.run(launch())
    else:worker(a.rank)
