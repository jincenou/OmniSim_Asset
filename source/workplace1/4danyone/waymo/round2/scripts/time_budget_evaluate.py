"""Same pixel metrics and backgrounds as the fixed-time diagnostic."""
import numpy as np
import torch
from PIL import Image, ImageDraw
from train_control import save
from evaluate_asset import Renderer, scalar_metrics, sha

@torch.inference_mode()
def evaluate(paths, data, frame, out):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    models=[Renderer(p,data) for p in paths]
    if not np.array_equal(models[0].parser.transform,models[1].parser.transform):
        raise ValueError('Camera coordinate transforms changed')
    rows=[]
    for view in (0,8,16,1,9,17):
        source=data/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
        rgba=np.asarray(Image.open(source).convert('RGBA'),dtype=np.float32)/255
        panels=[]
        for bg in (0.,.5,1.):
            target=rgba[...,:3]*rgba[...,3:]+bg*(1-rgba[...,3:])
            panels.append(target)
            for tag,model in zip(('dense50k','time_support'),models,strict=True):
                rgb,alpha=model.render(view,frame)
                pred=np.clip(rgb+bg*(1-alpha[...,None]),0,1)
                tensor=lambda x:torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1)[None].cuda()
                lp=float(metric(tensor(pred),tensor(target)));metric.reset()
                values=scalar_metrics(pred,target,alpha,rgba[...,3]>.5)
                if not all(np.isfinite(v) for v in (*values.values(),lp)):
                    raise ValueError('Nonfinite metric')
                rows.append(dict(view=view,frame=frame,background=bg,model=tag,lpips=lp,
                    split='validation_generated' if view in (0,8,16) else 'training_diagnostic',
                    reference_sha256=sha(source),**values))
                panels.append(pred)
        image=Image.fromarray(np.rint(np.concatenate(panels,axis=1)*255).astype(np.uint8))
        ImageDraw.Draw(image).text((5,5),f'v{view} frame{frame}; target / dense50k / time-support fit; black,gray,white',fill='red')
        image.save(out/f'view{view:02d}.jpg',quality=95)
    save(out/'metrics.json',{'state':'time_support_diagnostic_complete_visual_review_pending',
        'source_checkpoint':str(paths[0]),'source_sha256':sha(paths[0]),
        'checkpoint':str(paths[1]),'checkpoint_sha256':sha(paths[1]),'frame':frame,
        'samples':rows,'no_point_filtering':True,
        'scope':'Equal-extra-budget time-support diagnostic from dense50k. Selected-frame evaluation does not certify cross-time coherence or real Waymo accuracy.'})
