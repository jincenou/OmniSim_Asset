"""Matched raw metrics and complete four-panel videos; never publish partial method panels."""
from argparse import ArgumentParser
from contextlib import ExitStack
from types import SimpleNamespace
import ast,importlib.util,json,math
import numpy as np
import torch
import cv2
import lpips
from PIL import Image,ImageDraw
from common import ROOT,BASE,REPO,LABELS,manifest,camera,device,save,sha,stamp
from scene import init_model
from gaussian_renderer import render

KEYS=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']
def byte(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def scalar(pred,target,alpha,mask):
    e=(pred-target)**2;s=alpha>.5
    return {'full_psnr':-10*math.log10(max(float(e.mean()),1e-12)),
        'foreground_psnr':-10*math.log10(max(float(e[mask].mean()),1e-12)),
        'silhouette_iou':float((s&mask).sum()/max((s|mask).sum(),1)),
        'outside_alpha_mean':float(alpha[~mask].mean())}
def native_ssim():
    p=BASE/'source/4dgaussians/utils/loss_utils.py';tree=ast.parse(p.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['gaussian','create_window','ssim','_ssim']]
    scope={'torch':torch,'F':torch.nn.functional,'Variable':torch.autograd.Variable,'exp':math.exp}
    exec(compile(tree,str(p),'exec'),scope);return scope['ssim']
def writer_class():
    p=BASE/'backend/src/video_io.py';spec=importlib.util.spec_from_file_location('fixed_video_io',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.MP4Writer
class CEM:
    def __init__(self,label,stage,root=None):
        self.root=ROOT/'runs'/label/f'stage{stage}' if root is None else root
        self.config=json.loads((self.root/'config.json').read_text())
        complete=json.loads((self.root/'complete.json').read_text());assert complete['asset_sha256']==sha(self.root/'asset.pt')
        ds=SimpleNamespace(**self.config['dataset']);opt=SimpleNamespace(**self.config['optimization'])
        self.pipe=SimpleNamespace(**self.config['pipeline']);self.model=init_model(ds.model,ds)
        self.model.restore(torch.load(self.root/'asset.pt',weights_only=False)[0],opt)
        self.data,_=manifest(label);self.replay=[]
    def raw(self,view,frame):
        r=render(camera(self.data,view,frame),self.model,self.pipe,torch.zeros(3,device='cuda'),near=.01,far=100.)
        return r['render'].permute(1,2,0).cpu().numpy(),r['acc'][0].cpu().numpy()
    def validate(self):
        reference=np.load(self.root/'replay_reference.npz')
        frames=sorted({int(key.rsplit('_',1)[1]) for key in reference.files if key.startswith('rgb_')})
        assert len(frames)==3
        for frame in frames:
            for view in [0,8,16]:
                r,a=self.raw(view,frame)
                error=max(float(np.max(np.abs(r.transpose(2,0,1)-reference[f'rgb_{view}_{frame}']))),float(np.max(np.abs(a-reference[f'alpha_{view}_{frame}'][0]))))
                assert error<1e-5,error;self.replay.append({'view':view,'frame':frame,'max_error':error})

@torch.inference_mode()
def main():
    cli=ArgumentParser();cli.add_argument('--label',required=True,choices=LABELS);arg=cli.parse_args()
    gpu=device();data,mpath=manifest(arg.label);F=data['frame_count'];final=ROOT/'results'/arg.label
    assert not final.exists()
    out=ROOT/'tmp'/f'publish_{arg.label}';out.mkdir(parents=True,exist_ok=False)
    caches=ROOT/'tmp'/f'{arg.label}_hust_raw';assert (caches/'complete.json').exists()
    baseline=json.loads((BASE/'samples'/arg.label/'evaluations/4dgs_grid61/results.json').read_text())
    prior={(r['view'],r['frame']):r for r in baseline['samples']};crops={int(k):v for k,v in baseline['crops'].items()}
    models={'cem_stage1':CEM(arg.label,1),'cem_stage2':CEM(arg.label,2)}
    for model in models.values():
        model.validate()
        assert {r['frame'] for r in model.replay}=={0,(F-1)//2,F-1}
    ssim=native_ssim();perceptual=lpips.LPIPS(net='alex').cuda().eval();Writer=writer_class()
    rows={k:[] for k in ['hust_grid61',*models]};dense={k:[] for k in rows};temporal={k:[] for k in rows}
    sampled=baseline['sampled_frames'];baseline_max_error={k:0. for k in KEYS}
    for view in [0,8,16]:
        values=np.load(caches/f'view{view:02d}.npz');h_rgb=values['rgb'];h_alpha=values['alpha'];previous={}
        with Writer(str(out/f'view{view:02d}.mp4'),25,1408,672) as video:
            for frame in range(F):
                path=__import__('pathlib').Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
                im=np.asarray(Image.open(path),np.float32)/255.;mask=im[...,3]>.5
                target=im[...,:3]*im[...,3:]+(1-im[...,3:])*.5
                raw={'hust_grid61':(h_rgb[frame],h_alpha[frame]),**{k:m.raw(view,frame) for k,m in models.items()}}
                panels=[target]
                for name,(rgb,alpha) in raw.items():
                    assert rgb.shape==(640,352,3) and alpha.shape==(640,352)
                    assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                    pred=np.clip(rgb+(1-alpha[...,None])*.5,0,1);panels.append(pred)
                    score=scalar(pred,target,alpha,mask);score.update(view=view,frame=frame)
                    dense[name].append(score.copy())
                    if previous:
                        union=mask|previous['mask'];residual=(pred-previous[name])-(target-previous['target'])
                        temporal[name].append({'view':view,'frame':frame,'full_residual_mse':float(np.mean(residual**2)),
                            'fg_union_residual_mse':float(np.mean(residual[union]**2)),
                            'fg_union_residual_mae':float(np.mean(np.abs(residual[union])))})
                    if frame in sampled:
                        x0,y0,x1,y1=crops[view]
                        a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                        score.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),
                            crop_xyxy=crops[view],input_sha256=sha(path))
                        assert score['input_sha256']==prior[(view,frame)]['input_sha256']
                        rows[name].append(score)
                        if name=='hust_grid61':
                            for key in KEYS:
                                error=abs(score[key]-prior[(view,frame)][key]);baseline_max_error[key]=max(baseline_max_error[key],error)
                                assert error<1e-4,(view,frame,key,error)
                previous={'target':target,'mask':mask,**{name:panel for name,panel in zip(raw,panels[1:])}}
                canvas=Image.new('RGB',(1408,672),(24,24,24));canvas.paste(Image.fromarray(byte(np.concatenate(panels,1))),(0,32))
                draw=ImageDraw.Draw(canvas)
                for j,title in enumerate(['Generated reference','HUST grid61 (33k)','CEM stage1 (27k)','CEM complete (35k)']):draw.text((j*352+8,9),title,fill='white')
                video.append_data(np.asarray(canvas))
                if frame in [0,(F-1)//2,F-1]:canvas.save(out/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
        print(json.dumps({'label':arg.label,'view':view,'frames_evaluated':F}),flush=True)
    decoded={}
    for p in out.glob('*.mp4'):
        cap=cv2.VideoCapture(str(p));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(672,1408,3);count+=1
        cap.release();assert count==F;decoded[p.name]={'frames':count,'sha256':sha(p)}
    result={'label':arg.label,'state':'evaluated_visual_review_pending','updated_at':stamp(),'device':gpu,
        'mean':{name:{key:float(np.mean([r[key] for r in rr])) for key in KEYS} for name,rr in rows.items()},
        'temporal_mean':{name:{key:float(np.mean([r[key] for r in rr])) for key in ['full_residual_mse','fg_union_residual_mse','fg_union_residual_mae']} for name,rr in temporal.items()},
        'samples':rows,'dense':dense,'temporal':temporal,'baseline_metric_max_absolute_error':baseline_max_error,
        'videos':decoded,'replay':{k:m.replay for k,m in models.items()},'production_approved':False,
        'protocol':baseline['protocol'],'temporal_protocol':'Consecutive-frame RGB residual versus generated reference. Range [0,1] RGB, union of reference foreground masks for each pair. No optical-flow ground truth; not real-motion accuracy.',
        'input_manifest_sha256':sha(mpath),'budget_note':'CEM27k+8k vs HUST3k+30k; different sampling/recipes, not equal compute ablation.'}
    save(out/'results.json',result)
    save(out/'assets.json',{k:{'checkpoint':str(m.root/'asset.pt'),'sha256':sha(m.root/'asset.pt'),
        'config':str(m.root/'config.json'),'runtime':str(REPO),'adapter':str(ROOT/'scripts/common.py'),
        'format':'Native grouped Gaussian checkpoint with per-group trajectory, hierarchy, temporal opacity and SH',
        'coordinate_mapping':data['source_to_normalized'],'time':'frame index f, seconds f/25',
        'independent_replay':m.replay} for k,m in models.items()})
    final.parent.mkdir(parents=True,exist_ok=True)
    out.rename(final)
    print(json.dumps({'state':result['state'],'label':arg.label,'mean':result['mean']}),flush=True)

if __name__=='__main__':main()
