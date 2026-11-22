"""Preregistered fixed expanded proxies; MSE (not RMSE), deficit AND excess."""
from common import *
from prepare_diagnostics import remap
import argparse,csv,cv2,lpips
from PIL import Image

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--key',required=True);ap.add_argument('--raw');a=ap.parse_args();assert a.label in ['N06','N11'];verify();cv2.setNumThreads(2)
    root=NEW/'data/expanded_roi';manifest_roi=json.loads((root/a.label/'manifest.json').read_text());plan=json.loads((NEW/'PLAN.json').read_text())
    for p,h in plan['evaluation']['expanded_ROI_sha256'].items():assert sha(NEW/p)==h
    rawroot=Path(a.raw) if a.raw else NEW/'cache/raw'/a.key;assert (rawroot/'complete.json').exists();out=NEW/'results'/a.key;out.mkdir(parents=True,exist_ok=True);assert not (out/'expanded.json').exists();rows=[];perceptual=lpips.LPIPS(net='alex').cuda().eval()
    for v in [0,8,16]:
        with np.load(rawroot/f'view{v:02d}.npz') as z:rgb,alpha=z['rgb'],z['alpha']
        with np.load(root/a.label/f'view{v:02d}.npz') as z:masks,known,boxes=z['masks'],z['known'],z['boxes']
        with np.load(ROOT/'audit/roi'/a.label/f'flow{v:02d}.npz') as z:flow,valid=z['backward'],z['valid']
        previous=previous_target=None
        for f in range(len(rgb)):
            rgba=np.asarray(Image.open(BASE/'samples'/a.label/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'),np.float32)/255;ra=rgba[...,3];target=rgba[...,:3]*ra[...,None]+.5*(1-ra[...,None]);pred=np.clip(rgb[f]+.5*(1-alpha[f,...,None]),0,1);error=np.square(pred-target).mean(-1)
            te=None if f==0 else np.square((pred-remap(previous,flow[f-1]))-(target-remap(previous_target,flow[f-1]))).mean(-1)
            for side in [0,1]:
                m=masks[f,side];mass=float(ra[m].sum());area=int(m.sum());fgarea=int((m&(ra>.5)).sum());ok=bool(known[f,side]);temporal_mask=None if f==0 else m&valid[f-1]&(remap(masks[f-1,side].astype(np.uint8),flow[f-1],True)>0)
                if f and not known[f-1,side]:temporal_mask[:]=False
                row=dict(label=a.label,method=a.key,view=v,frame=f,side=side,known=ok,roi_pixels=area,reference_alpha_mass=mass,reference_foreground_pixels=fgarea,RGB_MSE=None,RGB_error_sum=None,deficit_sum=None,positive_excess_sum=None,soft_undercoverage=None,positive_excess=None,local_lpips=None,temporal_mse=None,temporal_valid_pixels=0,temporal_valid_fraction=None)
                if ok:
                    deficit=float(np.maximum(ra-alpha[f],0)[m].sum());excess=float(np.maximum(alpha[f]-ra,0)[m].sum());err=float(error[m].sum());row.update(RGB_MSE=err/area,RGB_error_sum=err,deficit_sum=deficit,positive_excess_sum=excess,soft_undercoverage=deficit/mass,positive_excess=excess/mass)
                    if fgarea>=128:
                        x0,y0,x1,y1=boxes[side];pa=torch.tensor(pred[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None];pb=torch.tensor(target[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None];row['local_lpips']=float(perceptual(pa*2-1,pb*2-1))
                    if f:
                        count=int(temporal_mask.sum());row.update(temporal_valid_pixels=count,temporal_valid_fraction=count/area,temporal_mse=float(te[temporal_mask].mean()) if count else None)
                rows.append(row)
            previous,previous_target=pred,target
    def aggregate(selection):
        valid=[r for r in selection if r['known']];n=len(valid);lp=[r['local_lpips'] for r in valid if r['local_lpips'] is not None];tp=[r for r in valid if r['temporal_mse'] is not None];area=sum(r['roi_pixels'] for r in valid);mass=sum(r['reference_alpha_mass'] for r in valid);tpx=sum(r['temporal_valid_pixels'] for r in tp)
        return dict(total_rows=len(selection),known_rows=n,unknown_rows=len(selection)-n,roi_pixels=area,reference_alpha_mass=mass,RGB_MSE=sum(r['RGB_error_sum'] for r in valid)/area if area else None,soft_undercoverage=sum(r['deficit_sum'] for r in valid)/mass if mass else None,positive_excess=sum(r['positive_excess_sum'] for r in valid)/mass if mass else None,local_LPIPS=float(np.mean(lp)) if lp else None,local_LPIPS_valid_rows=len(lp),temporal_RMSE=float(np.sqrt(sum(r['temporal_mse']*r['temporal_valid_pixels'] for r in tp)/tpx)) if tpx else None,temporal_valid_pixels=tpx,temporal_valid_fraction=tpx/area if area else None)
    bad=plan['evaluation']['fixed_bad_windows'][a.label];summary={f'side{s}':aggregate([r for r in rows if r['side']==s]) for s in [0,1]};summary['fixed_bad_window_left']=aggregate([r for r in rows if r['side']==0 and r['view']==bad['view'] and bad['frames'][0]<=r['frame']<=bad['frames'][1]])
    with (out/'expanded.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'expanded.json',dict(state='measured_proxy_unknown_preserved',label=a.label,method=a.key,summary=summary,roi_rules_sha256=sha(root/'rules.json'),all_methods_same_masks_denominators=True,full_reference_semantic_or_occlusion_GT=False,local_lpips='unmasked fixed reference ROI union crop,not a perceptual metric restricted pixelwise to anatomical part'))
    print(a.key,'expanded complete',flush=True)
if __name__=='__main__':main()
