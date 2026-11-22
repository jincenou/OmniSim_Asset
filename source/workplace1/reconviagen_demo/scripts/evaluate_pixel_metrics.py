"""Matched-source diagnostics; protocol frozen before any scores are computed."""
from pathlib import Path
import os,json,hashlib,csv,time,math
ROOT=Path(__file__).resolve().parents[1]
os.environ['TORCH_HOME']=str(ROOT/'cache/pixel_metrics_torch')
import numpy as np
from PIL import Image
import torch,lpips,skimage
from skimage.metrics import structural_similarity
from evaluate_asset_metrics import foreground

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(im):return torch.from_numpy(np.array(im,dtype=np.float32)/127.5-1).permute(2,0,1)[None].cuda()
def native_square(rgb_path,mask_path):
 rgb=np.array(Image.open(rgb_path).convert('RGB'));mask=np.array(Image.open(mask_path).convert('L'))>127
 h,w=mask.shape;side=max(h,w);x=(side-w)//2;y=(side-h)//2
 canvas=Image.new('RGB',(side,side),'white');canvas.paste(Image.fromarray(np.where(mask[...,None],rgb,255).astype('uint8')),(x,y))
 m=Image.new('L',(side,side),0);m.paste(Image.fromarray(mask.astype('uint8')*255),(x,y))
 return canvas.resize((512,512),Image.Resampling.BICUBIC),np.array(m.resize((512,512),Image.Resampling.NEAREST))>127

def compute(a,am,b,bm,net):
 x=np.asarray(a,dtype=np.float64)/255;y=np.asarray(b,dtype=np.float64)/255
 union=am|bm;inter=am&bm;err=(x-y)**2;mse=err.mean();mseu=err[union].mean()
 ssim,smap=structural_similarity(x,y,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11,full=True)
 with torch.inference_mode():lp=float(net(tensor(a),tensor(b)).item())
 return dict(PSNR_dB=float(-10*np.log10(mse)) if mse>0 else None,SSIM=float(ssim),LPIPS_alex=lp,PSNR_union_dB=float(-10*np.log10(mseu)) if mseu>0 else None,SSIM_union=float(smap.mean(2)[union].mean()),silhouette_IoU=float(inter.sum()/union.sum()),foreground_union_fraction=float(union.mean()))

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 backbone=Path('/mnt/ojc/workplace1/4danyone/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
 target=Path(os.environ['TORCH_HOME'])/'hub/checkpoints'/backbone.name;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.symlink_to(backbone)
 linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
 protocol=dict(version='matched_source_pixel_v1',frozen_before_scores=True,population='Same ten frozen identities and all six routes, seed42; ten representative extras separately',source='Same fixed undistorted training source foreground as ED-R, not heldout novel-view ground truth',primary='normalized: same foreground() as ED-R; independently center foreground tight bbox, square with 5% padding each side, white background, RGB bicubic512, mask nearest512; no rotation/warp/color fitting',secondary='native: identical original cropped camera canvas for source/render, zero per-image centering or scale fitting; common square padding and512 resize; preserves projected size/position mismatch',PSNR='RGB sRGB values in [0,1]; 10log10(1/MSE), whole square; report per identity dB then arithmetic identity mean',SSIM='skimage RGB channel average, data_range1, gaussian_weights=True sigma1.5 win_size11 use_sample_covariance=False; standard interior average',LPIPS='Official lpips AlexNet, learned v0.1 calibration; RGB[-1,1], float32, eval mode; standard full-square spatial average',foreground_sensitivity='PSNR_union: MSE only on union of source/render masks. SSIM_union: spatial SSIM map RGB mean pooled on union. These are explicitly nonstandard supplementary metrics. No intersection-only masking.',silhouette='IoU of the binary masks under the indicated alignment',aggregation='Identity unit, all ten common-valid IDs, no failure exclusions or best-view search; bootstrap10000 seed20261023',limitations=['Full-square scores include white background; union sensitivities accompany the main table','Matched camera is approximate and ignores row-wise rolling shutter','Training-source reconstruction diagnostic, not heldout quality or unseen backside ground truth','New exploratory metrics, not claimed as original rigid benchmark protocol'],weights=dict(backbone=str(backbone),backbone_sha256=sha(backbone),linear=str(linear),linear_sha256=sha(linear)),versions=dict(torch=torch.__version__,skimage=skimage.__version__),references=['https://github.com/richzhang/PerceptualSimilarity','https://scikit-image.org/docs/stable/api/skimage.metrics.html'])
 (ROOT/'docs/random10_pixel_protocol.json').write_text(json.dumps(protocol,indent=2))
 net=lpips.LPIPS(net='alex',version='0.1',lpips=True,pnet_rand=False,pnet_tune=False,verbose=False).cuda().eval()
 # Meaningful checks: identity must yield zero perceptual distance and SSIM one.
 test=Image.fromarray(np.random.default_rng(42).integers(0,256,(512,512,3),dtype=np.uint8));mask=np.ones((512,512),bool)
 check=compute(test,mask,test,mask,net);assert abs(check['LPIPS_alex'])<1e-7 and abs(check['SSIM']-1)<1e-10 and check['PSNR_dB'] is None
 tasks=json.loads((ROOT/'inputs/random10_experiment_matrix.json').read_text())+json.loads((ROOT/'inputs/random10_additional_matrix.json').read_text());rows=[]
 for t in tasks:
  case=t['case_id'];v=t['variant'];seed=t['seed'];cam_path=ROOT/'inputs'/case/'evaluation/camera.json';cam=json.loads(cam_path.read_text());out=ROOT/'results'/case/f'{v}_seed{seed}'/'evaluation'
  audit=json.loads((out/'render_audit.json').read_text());assert audit['valid_export'] and audit['valid_16_views'] and audit['source_camera_sha256']==sha(cam_path)
  rendered=out/'matched_source.png';render_mask=out/'matched_source_alpha.png';assert Image.open(rendered).size==Image.open(cam['source_rgb']).size
  results={}
  for mode in ['normalized','native']:
   func=foreground if mode=='normalized' else native_square
   a,am,*_=func(cam['source_rgb'],cam['source_mask']);b,bm,*_=func(rendered,render_mask)
   score=compute(a,am,b,bm,net);results[mode]=score
   rows.append(dict(case_id=case,variant=v,seed=seed,alignment=mode,**score))
  (out/'pixel_metrics.json').write_text(json.dumps(dict(case_id=case,variant=v,seed=seed,source_id=cam['source_id'],source_camera_sha256=sha(cam_path),source_rgb_sha256=sha(cam['source_rgb']),source_mask_sha256=sha(cam['source_mask']),render_sha256=sha(rendered),render_mask_sha256=sha(render_mask),protocol_sha256=sha(ROOT/'docs/random10_pixel_protocol.json'),scores=results),indent=2))
  print(case,v,seed,results['normalized'],flush=True)
 ids={c['case_id'] for c in json.loads((ROOT/'inputs/random10_cases.json').read_text())};variants=['B0','R','F','H','B0_real1','B0_real4'];metrics=list(check);summary=[];paired=[]
 for mode in ['normalized','native']:
  for v in variants:
   selected=[r for r in rows if r['alignment']==mode and r['variant']==v and r['seed']==42];assert {r['case_id'] for r in selected}==ids
   s=dict(variant=v,alignment=mode,n=10)
   for k in metrics:
    x=np.array([r[k] for r in selected]);assert np.isfinite(x).all();rng=np.random.default_rng(20261023);boot=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    s[k]=dict(mean=float(x.mean()),sample_std=float(x.std(ddof=1)),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
   summary.append(s)
  for target,ref in [('H','B0'),('B0_real4','B0'),('H','B0_real4'),('H','R')]:
   s=dict(alignment=mode,target=target,reference=ref,n=10)
   for k in metrics:
    by={(r['case_id'],r['variant']):r[k] for r in rows if r['alignment']==mode and r['seed']==42}
    x=np.array([by[c,target]-by[c,ref] for c in sorted(ids)]);rng=np.random.default_rng(20261023);boot=x[rng.integers(0,10,(10000,10))].mean(1);s[k]=dict(mean_delta=float(x.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
   paired.append(s)
 result=dict(protocol='random10_pixel_protocol.json',rows=rows,summary=summary,paired=paired,checks=check,completed_unix=time.time())
 (ROOT/'docs/random10_pixel_metrics.json').write_text(json.dumps(result,indent=2))
 with (ROOT/'docs/random10_pixel_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('COMPLETE',len(rows),'scores',flush=True)
if __name__=='__main__':main()
