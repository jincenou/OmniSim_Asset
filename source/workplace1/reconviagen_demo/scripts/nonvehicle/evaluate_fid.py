"""Small-sample foreground orbit FID; distribution level, never per-asset FID."""
from pathlib import Path
import os,sys,json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parents[2];N=R/'nonvehicle10';D=N/'docs'
os.environ['TORCH_HOME']=str(R/'cache/fid')
sys.path[:0]=[str(R/'tools/fid_deps'),str(R/'scripts')]
from evaluate_asset_metrics import foreground
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
weight=R/'cache/fid/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(weight).startswith('6726825d'),'Official checkpoint hash mismatch or incomplete download'
cases=json.loads((N/'inputs/cases.json').read_text());variants=['B0','R','H','B0_real1'];groups={'source':[]}|{v:[] for v in variants}
for c in cases:
 cid=c['case_id'];manifest=json.loads((N/'inputs'/cid/'segmented_manifest.json').read_text());src=next(x for x in manifest['images'] if x['id']==manifest['flux_reference_id'])
 groups['source'].append((cid,src['path'],src['mask_path']))
 for v in variants:
  o=N/'results'/cid/f'{v}_seed42'/'orbit'
  groups[v].extend((cid,str(o/f'view_{i:02}.png'),str(o/f'alpha_{i:02}.png')) for i in range(16))
protocol=dict(name='FID-2048 foreground orbit pilot',version=1,implementation='pytorch-fid==0.3.0',implementation_url='https://github.com/mseitzer/pytorch-fid',checkpoint=str(weight),checkpoint_sha256=sha(weight),features='FID TensorFlow-compatible Inception v3 pool3, 2048 unnormalized activations; float32; no TF32',reference='One fixed real_00 foreground per identity: 10 images',generated='All 16 shared orbit views per identity: 160 images per route',preprocessing='Same foreground helper as I16: alpha>127, tight square crop, 5% padding per side, white background, PIL bicubic to512; float RGB[0,1]; pytorch-fid bilinear299 align_corners=False and normalization to[-1,1]',covariance='Unbiased ddof=1; float64 mean/covariance algebra; exact low-rank SVD of centered feature cross-product; no epsilon regularization',formula='||mu_x-mu_y||^2 + tr(C_x)+tr(C_y) - 2||A B^T||_* where A=(X-mu_x)/sqrt(n_x-1), B=(Y-mu_y)/sqrt(n_y-1)',aggregation='One distribution-level value per route. Not averaged per asset or per view. No per-category FID with n=1/2.',lower_is_better=True,limitations=['Only10 reference identities; rank-deficient covariance, strong finite-sample bias','160 orbit images are correlated, not160 independent objects','Source view and full orbit distribution differ; foreground only, not scene insertion FID','Exploratory comparison within this fixed cohort only; not publication-grade population estimate or semantic/geometry pass criterion'],case_ids=[c['case_id'] for c in cases],input_sha256={k:[dict(case_id=c,rgb=p,mask=m,rgb_sha256=sha(p),mask_sha256=sha(m)) for c,p,m in rows] for k,rows in groups.items()})
# Persist protocol and exact inputs BEFORE reading scores.
(D/'fid_protocol.json').write_text(json.dumps(protocol,indent=2))
def fid(x,y):
 x=x.astype(np.float64);y=y.astype(np.float64);a=(x-x.mean(0))/np.sqrt(len(x)-1);b=(y-y.mean(0))/np.sqrt(len(y)-1)
 return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)+np.sum(b*b)-2*np.linalg.svd(a@b.T,compute_uv=False).sum())
rng=np.random.default_rng(42);x=rng.normal(size=(41,16));y=rng.normal(size=(53,16))
ref=calculate_frechet_distance(x.mean(0),np.cov(x,rowvar=False),y.mean(0),np.cov(y,rowvar=False));err=abs(fid(x,y)-ref);assert err<1e-8,err
assert abs(fid(x,x))<1e-8
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval();feats={};cache=N/'evaluation/fid';cache.mkdir(parents=True,exist_ok=True)
for name,rows in groups.items():
 out=[]
 for i in range(0,len(rows),16):
  ims=[np.array(foreground(p,m)[0],dtype=np.float32)/255 for _,p,m in rows[i:i+16]]
  batch=torch.from_numpy(np.stack(ims)).permute(0,3,1,2).cuda()
  with torch.inference_mode():f=model(batch)[0].flatten(1).cpu().numpy()
  out.append(f)
 feats[name]=np.concatenate(out);assert feats[name].shape==(len(rows),2048) and np.isfinite(feats[name]).all()
 np.save(cache/f'{name}_features.npy',feats[name]);print('ENCODED',name,len(rows),flush=True)
results=[dict(variant=v,FID_2048=fid(feats['source'],feats[v]),N_reference=10,N_rendered=160,N_identities=10) for v in variants]
# Cross-check one actual, singular 2048-d covariance through upstream implementation.
x=feats['source'].astype('float64');y=feats['B0'].astype('float64')
upstream=calculate_frechet_distance(x.mean(0),np.cov(x,rowvar=False),y.mean(0),np.cov(y,rowvar=False));delta=abs(upstream-results[0]['FID_2048']);assert delta<1e-3,delta
out=dict(protocol='fid_protocol.json',results=results,numerical_audit=dict(random_full_rank_max_error=err,actual_B0_upstream_scipy_value=upstream,actual_B0_absolute_delta=delta,self_fid=fid(x,x)),feature_files={k:dict(path=str(cache/f'{k}_features.npy'),sha256=sha(cache/f'{k}_features.npy')) for k in feats},torch_version=torch.__version__)
(D/'fid_metrics.json').write_text(json.dumps(out,indent=2))
with (D/'fid_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
print(json.dumps(results),flush=True)
