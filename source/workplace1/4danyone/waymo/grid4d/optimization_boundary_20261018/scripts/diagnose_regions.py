"""Historical continue/alpha/combined partitions, train anchors and fixed local LPIPS."""
import argparse
import csv
import json
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import lpips
from boundary_common import *
from region_metrics import rows_for_frame
from appearance_query import panel,crop_box,PARTS


def write_csv(path,rows):
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args();begin=time.monotonic();device=verify();cv2.setNumThreads(2)
    label=args.label;spec=plan();out=NEW/'results/regions'/label;out.mkdir(parents=True,exist_ok=False)
    anchors=[a for a in json.loads((NEW/'data/anchors.json').read_text()) if a['label']==label];train=json.loads(Path(spec['identities'][label]['data']['training']).read_text());paths={(r['view'],r['frame']):r['path'] for r in train}
    original_manifest=json.loads(Path(json.loads((Path(spec['identities'][label]['start_run'])/'config.json').read_text())['provenance']['manifest']).read_text());source=Path(original_manifest['source_data'])
    perceptual=lpips.LPIPS(net='alex').cuda().eval();regions_rows=[];external_rows=[];local_rows=[];image_evidence=[]
    selections=json.loads((ROOT/'gallery'/label/'selection.json').read_text())['parts']
    def local_measure(rgb,alpha,rgba,core,part,method,view,frame,domain,anchor_id=None):
        mask=core==PARTS.index(part)+1;area=int(mask.sum());box=crop_box(mask);ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1)
        row=dict(label=label,method=method,domain=domain,anchor=anchor_id,view=view,frame=frame,part=part,reference_core_pixels=area,crop_xyxy=json.dumps(box),crop_lpips=None,coverage=None,mse=None,red_bias=None,green_bias=None,blue_bias=None,reference_foreground_pixels=int((rgba[...,3]>=.5).sum()),unknown_foreground_pixels=int(((core==0)&(rgba[...,3]>=.5)).sum()))
        if area:
            row.update(coverage=float(alpha[mask].mean()),mse=float(np.square(pred[mask]-ref[mask]).mean()),**dict(zip(['red_bias','green_bias','blue_bias'],(pred[mask]-ref[mask]).mean(0).astype(float).tolist())))
            x0,y0,x1,y1=box
            if min(x1-x0,y1-y0)>=32:
                a=torch.tensor(pred[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None];b=torch.tensor(ref[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None]
                row['crop_lpips']=float(perceptual(a*2-1,b*2-1))
        local_rows.append(row)
    for method in ['continue','alpha','combined']:
        run=ROOT/'optimization/runs'/f'{label}_{method}';model,deform,pipe,config=load_asset(run,48000);data=json.loads((run/'asset_camera_data.json').read_text())
        for anchor in anchors:
            view,frame=anchor['view'],anchor['frame'];rgba=np.asarray(Image.open(paths[view,frame]),np.float32)/255;core=np.load(ROOT/'optimization/data'/label/f'view{view:02d}.npy',mmap_mode='r')[frame]
            raw,a=raw_render(model,deform,pipe,camera(data,view,frame));rgb=raw.permute(1,2,0).cpu().numpy();alpha=a[0].cpu().numpy()
            rows,external=rows_for_frame(rgb,alpha,rgba,core,label,method,view,frame,'training_anchor');regions_rows.extend(rows);external_rows.append(dict(label=label,method=method,domain='training_anchor',view=view,frame=frame,**external))
            local_measure(rgb,alpha,rgba,core,anchor['part'],method,view,frame,'training_anchor',anchor['id'])
            mask=core==PARTS.index(anchor['part'])+1;box=crop_box(mask)
            name=f"{anchor['id']}_{method}";path=out/'partition_images'/f'{name}.png';ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])
            panel(path,[ref,np.clip(rgb,0,1),np.clip(rgb+.5*(1-alpha[...,None]),0,1),np.clip(rgb+1-alpha[...,None],0,1),alpha,rgba[...,3]],['Reference gray','Pred black','Pred gray','Pred white','Pred alpha','Ref soft alpha'],box)
            image_evidence.append(str(path.relative_to(NEW)))
            if method=='alpha':
                partition=regions(rgba[...,3]);colors=np.array([[25,25,25],[60,160,80],[240,170,20],[210,70,180]],np.uint8);paint=colors[partition]
                target=out/'partition_images'/f"{anchor['id']}_regions.png";Image.fromarray(paint).save(target)
        del model,deform
        for view in [0,8,16]:
            with np.load(ROOT/'optimization/raw'/f'{label}_{method}'/f'view{view:02d}.npz') as z:rgbs,alphas=z['rgb'],z['alpha']
            core=np.load(ROOT/'audit/roi'/label/f'view{view:02d}.npz')['labels'];sampled=np.rint(np.linspace(0,len(alphas)-1,13)).astype(int).tolist()
            for frame in range(len(alphas)):
                rgba=np.asarray(Image.open(source/'images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                rows,external=rows_for_frame(rgbs[frame],alphas[frame],rgba,core[frame],label,method,view,frame,'heldout_development');regions_rows.extend(rows);external_rows.append(dict(label=label,method=method,domain='heldout_development',view=view,frame=frame,**external))
                if frame in sampled:
                    for part in PARTS:local_measure(rgbs[frame],alphas[frame],rgba,core[frame],part,method,view,frame,'heldout_development')
    # Apply the same crop rule to new arms; these are diagnostics, not additive LPIPS maps.
    for method in ['A','U','B']:
        directory=NEW/'cache/raw/primary'/f'{label}_{method}';assert (directory/'complete.json').exists()
        for view in [0,8,16]:
            with np.load(directory/f'view{view:02d}.npz') as z:rgbs,alphas=z['rgb'],z['alpha']
            core=np.load(ROOT/'audit/roi'/label/f'view{view:02d}.npz')['labels']
            for frame in np.rint(np.linspace(0,len(alphas)-1,13)).astype(int):
                rgba=np.asarray(Image.open(source/'images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                for part in PARTS:local_measure(rgbs[frame],alphas[frame],rgba,core[frame],part,method,view,int(frame),'heldout_development')
    write_csv(out/'boundary_diagnosis.csv',regions_rows);write_csv(out/'external_alpha.csv',external_rows);write_csv(out/'local_crops.csv',local_rows)
    save(out/'complete.json',dict(state='measured_visual_mask_review_pending',label=label,device=device,seconds=time.monotonic()-begin,images=image_evidence,rows=len(regions_rows),outside_alpha_definition='mean predicted accumulated alpha over frozen reference alpha<=0.5; uint8 labels mean no exact0.5; background is not assumed reference alpha==0',positive_excess='mean max(pred_alpha-ref_soft_alpha,0) on same region',lpips='Independent fixed reference-core bounding crops with8px padding; min32x32. Not additive regional contributions.',unknown='Core0 inside reference foreground is unknown, not absent body; no hand/foot/face GT claims',mask_policy='Labels unchanged. Potential neighboring people/poles/ground contamination requires visual review; hats/bags are not automatically removed'))
    print(label,'historical partitions and same-rule local crops measured',flush=True)


if __name__=='__main__':main()
