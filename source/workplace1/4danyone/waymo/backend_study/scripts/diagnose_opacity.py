"""Read-only checkpoint diagnosis: alpha errors under different backgrounds."""
import gc,json
import numpy as np
import torch
from PIL import Image
from common import ROOT,manifest,device,save,stamp
from evaluate_case import CEM,scalar

@torch.inference_mode()
def main():
    target=ROOT/'research/opacity_diagnostic.json'
    assert not target.exists(),'Keep this completed diagnostic immutable'
    gpu=device();records=[]
    for label in ['N04','N08','N07','N02']:
        data,_=manifest(label);frames=[0,(data['frame_count']-1)//2,data['frame_count']-1]
        for stage in [1,2]:
            model=CEM(label,stage)
            for view in [0,8,16,1,9,17]:
                for frame in frames:
                    path=__import__('pathlib').Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
                    im=np.asarray(Image.open(path),np.float32)/255.;mask=im[...,3]>.5
                    rgb,alpha=model.raw(view,frame)
                    bad=(~mask)&(alpha>.5)
                    outside=~mask;count=int(bad.sum());means={}
                    for background in [0.,.5,1.]:
                        reference=im[...,:3]*im[...,3:]+(1-im[...,3:])*background
                        prediction=np.clip(rgb+(1-alpha[...,None])*background,0,1)
                        score=scalar(prediction,reference,alpha,mask)
                        score['outside_mse']=float(((prediction-reference)**2)[outside].mean())
                        means[str(background)]=score
                    color=np.clip(rgb[bad]/alpha[bad,None],0,1)
                    records.append(dict(label=label,stage=stage,view=view,frame=frame,
                        split='heldout' if view in [0,8,16] else 'train',
                        false_opaque_pixels=count,outside_pixels=int(outside.sum()),
                        false_opaque_fraction=count/int(outside.sum()),
                        false_opaque_color_mean_rgb=color.mean(0).tolist() if count else None,
                        false_opaque_near_white_fraction=float((color.min(-1)>.9).mean()) if count else None,
                        by_background=means))
            del model;gc.collect();torch.cuda.empty_cache()
    save(target,dict(state='complete',updated_at=stamp(),device=gpu,records=records,
        protocol='Read-only final checkpoints; same premultiplied RGB/alpha recomposited on black/gray/white. No altered training or postprocessing. Train views1,9,17 vs heldout0,8,16 at first/mid/last.',
        interpretation_limit='This diagnoses opacity/color ambiguity; it does not establish the causal gain of adding a mask loss or changing the training background.'))
    print(json.dumps({'state':'complete','records':len(records),'path':str(target)}))

if __name__=='__main__':main()
