"""Evaluate frozen H-count renders with existing metric definitions, isolated outputs."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'h_count_ablation'
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_asset_metrics import foreground

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tasks():return json.loads((WORK/'registry.json').read_text())['tasks']
def check(t):
    out=ROOT/'results'/t['case_id']/f"{t['variant']}_seed42"/'evaluation'
    cam_path=ROOT/'inputs'/t['case_id']/'evaluation/camera.json'
    audit=json.loads((out/'render_audit.json').read_text())
    assert audit['valid_export'] and audit['valid_16_views'] and audit['matched_foreground_pixels']>0
    assert audit['source_camera_sha256']==sha(cam_path)
    return out,json.loads(cam_path.read_text())

def pixels(case_id):
    os.environ['TORCH_HOME']=str(ROOT/'cache/pixel_metrics_torch')
    import torch,lpips
    from evaluate_pixel_metrics import compute,native_square
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=lpips.LPIPS(net='alex',version='0.1',lpips=True,pnet_rand=False,pnet_tune=False,verbose=False).cuda().eval()
    for t in tasks():
        if t['case_id']!=case_id or t['reused']:continue
        out,cam=check(t);results={}
        for mode,func in [('normalized',foreground),('native',native_square)]:
            a,am,*_=func(cam['source_rgb'],cam['source_mask'])
            b,bm,*_=func(out/'matched_source.png',out/'matched_source_alpha.png')
            results[mode]=compute(a,am,b,bm,net)
        record=dict(case_id=case_id,variant=t['variant'],seed=42,scores=results,
            protocol_sha256=sha(ROOT/'docs/random10_pixel_protocol.json'),
            registry_sha256=sha(WORK/'registry.json'),render_sha256=sha(out/'matched_source.png'))
        (out/'pixel_metrics.json').write_text(json.dumps(record,indent=2))
        print(case_id,t['variant'],results['normalized'],flush=True)

def fid(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64)
    a=(x-x.mean(0))/np.sqrt(len(x)-1);b=(y-y.mean(0))/np.sqrt(len(y)-1)
    return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)+np.sum(b*b)-2*np.linalg.svd(a@b.T,compute_uv=False).sum())

def evaluate_fid():
    os.environ['TORCH_HOME']=str(ROOT/'cache/fid')
    sys.path.insert(0,str(ROOT/'tools/fid_deps'))
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weight=ROOT/'cache/fid/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
    assert sha(weight)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
    groups={'source':[],**{str(n):[] for n in range(1,5)}}
    for t in tasks():
        out,cam=check(t);n=t['real_count'];cid=t['case_id']
        if n==1:groups['source'].append((cid,cam['source_rgb'],cam['source_mask']))
        groups[str(n)].extend((cid,str(out/f'orbit_{i:02}.png'),str(out/f'orbit_{i:02}_alpha.png')) for i in range(16))
    base=json.loads((ROOT/'docs/random10_fid_protocol.json').read_text())
    protocol={k:base[k] for k in ['implementation','features','preprocessing','covariance','formula','limitations']}
    protocol.update(registry_sha256=sha(WORK/'registry.json'),checkpoint_sha256=sha(weight),
        groups={k:[dict(case_id=c,rgb=p,mask=m,rgb_sha256=sha(p),mask_sha256=sha(m)) for c,p,m in v] for k,v in groups.items()})
    (WORK/'fid_protocol.json').write_text(json.dumps(protocol,indent=2))
    rng=np.random.default_rng(42);x=rng.normal(size=(41,16));y=rng.normal(size=(53,16))
    error=abs(fid(x,y)-calculate_frechet_distance(x.mean(0),np.cov(x,rowvar=False),y.mean(0),np.cov(y,rowvar=False)))
    assert error<1e-8 and abs(fid(x,x))<1e-8
    model=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    cache=WORK/'fid_features';cache.mkdir(exist_ok=True);features={}
    for name,rows in groups.items():
        chunks=[]
        for i in range(0,len(rows),16):
            batch=np.stack([np.array(foreground(p,m)[0],dtype=np.float32)/255 for _,p,m in rows[i:i+16]])
            with torch.inference_mode():f=model(torch.from_numpy(batch).permute(0,3,1,2).cuda())[0].flatten(1).cpu().numpy()
            chunks.append(f)
        features[name]=np.concatenate(chunks);assert np.isfinite(features[name]).all()
        np.save(cache/f'{name}.npy',features[name]);print('FID encoded',name,len(rows),flush=True)
    scores=[dict(real_count=n,FID=fid(features['source'],features[str(n)]),N_source=10,N_render=160) for n in range(1,5)]
    old=json.loads((ROOT/'docs/random10_fid_metrics.json').read_text())
    old_h=next(r['FID_2048'] for r in old['results'] if r['variant']=='H')
    assert abs(scores[-1]['FID']-old_h)<1e-3,(scores[-1],old_h)
    (WORK/'fid_metrics.json').write_text(json.dumps(dict(results=scores,numerical_check=error,H4_reproduction_delta=scores[-1]['FID']-old_h),indent=2))
    print(json.dumps(scores),flush=True)

def summarize():
    rows=[];fids={r['real_count']:r['FID'] for r in json.loads((WORK/'fid_metrics.json').read_text())['results']}
    for t in tasks():
        out,cam=check(t)
        clip=json.loads((out/'clip_metrics.json').read_text());dino=json.loads((out/'dino_metrics.json').read_text())
        pixel=json.loads((out/'pixel_metrics.json').read_text())['scores']['normalized']
        info=json.loads((out.parent/'inference.json').read_text())
        rows.append(dict(case_id=t['case_id'],real_count=t['real_count'],variant=t['variant'],
            CLIP_I16=clip['I16'],DINO_I16=dino['I16'],ED_R=dino['ED_R'],
            PSNR=pixel['PSNR_dB'],SSIM=pixel['SSIM'],LPIPS=pixel['LPIPS_alex'],
            silhouette_IoU=pixel['silhouette_IoU'],PSNR_union=pixel['PSNR_union_dB'],
            inference_seconds=info['inference_seconds'],peak_GiB=info['peak_cuda_allocated_bytes']/1024**3))
    summary=[];paired=[];keys=list(rows[0])[3:]
    for n in range(1,5):
        selected=[r for r in rows if r['real_count']==n];assert len(selected)==10
        summary.append(dict(real_count=n,N=10,AOR=100.,FID=fids[n],**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
        if n<4:
            by={(r['case_id'],r['real_count']):r for r in rows}
            record=dict(real_count=n,reference_count=4)
            for key in keys:
                delta=np.array([r[key]-by[(r['case_id'],4)][key] for r in selected])
                rng=np.random.default_rng(42);boot=delta[rng.integers(0,10,(10000,10))].mean(1)
                record[key]=dict(mean_delta=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
            paired.append(record)
    for name,data in [('per_asset.csv',rows),('summary.csv',summary)]:
        with (WORK/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (WORK/'metrics.json').write_text(json.dumps(dict(rows=rows,summary=summary,paired_vs_H4=paired),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pixels','fid','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.stage=='pixels':pixels(a.case)
    elif a.stage=='fid':evaluate_fid()
    else:summarize()
