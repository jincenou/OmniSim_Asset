"""Frozen source crop -> official VAE round trip. Does not run the DiT."""
import argparse,json,os,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont,ImageOps
from scipy.ndimage import binary_erosion
import torch

ROOT=Path(__file__).resolve().parent;W=ROOT.parent;RUN=W/'scene20/r20261019'
parser=argparse.ArgumentParser();parser.add_argument('case');args=parser.parse_args()
OLD=RUN/'scenes'/args.case;OUT=ROOT/'appearance'/OLD.name
frames=json.loads((OLD/'generation.json').read_text())['output']['frames_per_video']
REPO=W/'round2/moving61_runtime' if frames==61 else W.parent/'repo'
sys.path[:0]=[str(REPO),str(RUN/'scripts')]
from interpolate import device_check
from fdanyone.foreground import predict_foreground_masks
from fdanyone.assets import resolve_foreground_model,resolve_base_assets
from fdanyone.config import CROP,INFERENCE
from fdanyone.geometry.crop import mask_bounds,crop_from_bounds
from fdanyone.skeleton.pipeline import _video_tensor
from fdanyone.model.vae import VaeExecutor

def main():
 torch.set_num_threads(4);begin=time.time();device=device_check();OUT.mkdir(parents=True,exist_ok=False)
 with av.open(str(OLD/'input/model_input.mp4')) as video:
  video.streams.video[0].thread_count=1;rgb=tuple(f.to_ndarray(format='rgb24') for f in video.decode(video=0))
 assert len(rgb)==frames
 masks=predict_foreground_masks(rgb,resolve_foreground_model(REPO/'models'),'cuda:0')
 h,w=rgb[0].shape[:2]
 bounds=mask_bounds(masks,CROP.mask_threshold)
 if bounds is None:raise ValueError('Cannot audit absent foreground')
 crop=crop_from_bounds(bounds=bounds,image_height=h,image_width=w,output_height=INFERENCE.height,output_width=INFERENCE.width,margins=CROP.margins,allow_upscale=CROP.allow_upscale)
 source=_video_tensor(OLD/'input/model_input.mp4',frames,crop=crop)
 source_rgb=np.rint((source[0].permute(1,2,3,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
 del source_rgb # Stats use exact floating source tensor below, not an extra quantized input.
 vae=VaeExecutor.load(resolve_base_assets(REPO/'models').vae,('cuda:0',))
 latent=vae.encode(source)
 decoded=vae._decode_and_publish(latent,lambda index,images:images)[0];vae.close()
 original=(source[0].permute(1,2,3,0).numpy()+1)*127.5
 sam=np.load(OLD/'input/target_masks.npz')['masks']
 rows=[];panels=[];font=ImageFont.load_default(size=18)
 selected=np.rint(np.linspace(0,frames-1,5)).astype(int).tolist()
 for t,(a,b,mask) in enumerate(zip(original,decoded,sam,strict=True)):
  mask=Image.fromarray(mask).crop((crop.left,crop.top,crop.left+crop.width,crop.top+crop.height)).resize((INFERENCE.width,INFERENCE.height),Image.Resampling.NEAREST)
  core=binary_erosion(np.asarray(mask)>242,iterations=5)
  if not core.any():raise ValueError(f'No reliable source foreground at {t}')
  x=a[core].astype(float);y=b[core].astype(float);luma=np.array([.2126,.7152,.0722])
  rows.append(dict(slot=t,core_pixels=int(core.sum()),source_mean_luma_rgb8=float((x@luma).mean()),vae_mean_luma_rgb8=float((y@luma).mean()),
   signed_luma_change_rgb8=float(((y-x)@luma).mean()),rgb_mae=float(np.abs(y-x).mean()),source_rgb_mean=x.mean(0).tolist(),vae_rgb_mean=y.mean(0).tolist()))
  if t in selected:
   pair=Image.new('RGB',(528,510),'#20242b');draw=ImageDraw.Draw(pair)
   for j,(label,im) in enumerate([('VAE input',Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8))),('VAE reconstruction',Image.fromarray(b))]):
    draw.text((j*264+3,3),f'{OLD.name} t={t/25:.2f}s {label}',font=font,fill='white');im=ImageOps.contain(im,(264,480));pair.paste(im,(j*264+(264-im.width)//2,30+(480-im.height)//2))
   panels.append(pair)
 sheet=Image.new('RGB',(528*len(panels),510))
 for i,panel in enumerate(panels):sheet.paste(panel,(528*i,0))
 sheet.save(OUT/'vae_comparison.jpg',quality=94)
 report=dict(state='roundtrip_measured_not_quality_pass',device=device,source=str(OLD),source_crop_recomputed=asdict(crop),frames=frames,
  mean_signed_luma_change_rgb8=float(np.mean([r['signed_luma_change_rgb8'] for r in rows])),
  mean_foreground_rgb_mae_rgb8=float(np.mean([r['rgb_mae'] for r in rows])),
  seconds=time.time()-begin,metrics=rows,
  scope='Same source pixels and frozen official crop/mask/VAE implementation; foreground is original SAM alpha>242 eroded5 pixels after nearest resize. Gamma-encoded luma proxy, not linear radiance. Not a comparison of different generated target views.',
  limitation='Original source crop metadata was not retained in compact delivery; recomputed via pinned current model/code, not proven byte-identical to historical crop. No DiT rerun and no exposure correction.')
 (OUT/'vae_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()
