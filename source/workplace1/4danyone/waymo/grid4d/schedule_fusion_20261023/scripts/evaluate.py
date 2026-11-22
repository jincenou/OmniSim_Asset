"""Stream fixed global/part/time evaluation; publish comparisons only when complete."""
import argparse,csv,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch,lpips,imageio_ffmpeg
from PIL import Image,ImageDraw
STUDY=Path(__file__).resolve().parents[1];GRID=STUDY.parent
sys.path.insert(0,str(GRID/'scripts'))
from grid_common import load_asset,raw_render,camera,manifest,save,sha,verify
from evaluate_grid import frozen_functions
from prepare_diagnostics import PARTS,remap


class Video:
    def __init__(self,path,width,height):
        assert not path.exists();self.path=path
        self.p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r','25','-i','-','-c:v','libx264','-threads','1','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    def append(self,image):self.p.stdin.write(np.ascontiguousarray(image).tobytes())
    def close(self):
        self.p.stdin.close();error=self.p.stderr.read();assert self.p.wait()==0,error.decode()


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--methods',nargs='+',default=['F0','F1','F2']);args=ap.parse_args()
    device=verify();begin=time.monotonic();label=args.label;cv2.setNumThreads(2)
    maps=json.loads((GRID/'optimization_cross8_20261018/ASSETS.json').read_text())['assets']
    baseline=next(a for a in maps if a['label']==label and a['method']=='A')
    descriptors={'A':(Path(baseline['asset']),53000)}
    for name in args.methods:
        run=STUDY/'runs'/f'{label}_{name}';assert (run/'complete.json').is_file()
        descriptors[name]=(run,33000 if not name.endswith('38') else 38000)
    suffix='' if args.methods==['F0','F1','F2'] else '_extension'
    out=STUDY/'results'/f'{label}{suffix}';out.mkdir(parents=True,exist_ok=False)
    media=STUDY/'comparison'/f'{label}{suffix}';media.mkdir(parents=True,exist_ok=False)
    models={};data=None;replays={}
    # Deny reference image reads while replaying sparse exported probes.
    denied=GRID.parent/'random8/r20261015_113332';scope={'replay':False}
    def guard(event,arguments):
        if scope['replay'] and event=='open' and isinstance(arguments[0],(str,bytes)):
            if Path(os.fsdecode(arguments[0])).resolve().is_relative_to(denied.resolve()):raise PermissionError('No original input reads during replay')
    sys.addaudithook(guard)
    for name,(run,it) in descriptors.items():
        models[name]=load_asset(run,it)[:3]
        current=json.loads((run/'asset_camera_data.json').read_text())
        if data is None:data=current
        else:assert all(current[k]==data[k] for k in ['cameras','width','height','frame_count','time_denominator'])
        if name=='A':continue
        scope['replay']=True
        with np.load(run/f'snapshots/{it}_reference.npz',allow_pickle=False) as refs:
            errors=[]
            for key in refs.files:
                v,f=key.split('_');rgb,alpha=raw_render(*models[name],camera(current,int(v[1:]),float(f[1:])))
                error=float(np.abs(torch.cat([rgb,alpha]).cpu().numpy()-refs[key]).max());assert error<=1e-5
                errors.append(error)
        scope['replay']=False;replays[name]={'probes':len(errors),'max_error':max(errors),'source_image_reads':0,'includes_fractional12_25':True}
    full,_=manifest(label);source=Path(full['source_data']);F=data['frame_count'];H,W=data['height'],data['width']
    scalar,crop,(ssim,ssim_sha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval()
    sampled=np.rint(np.linspace(0,F-1,13)).astype(int).tolist();metric_rows={n:[] for n in models};part_rows={n:[] for n in models};contacts=[]
    for view in [0,8,16]:
        masks=[np.asarray(Image.open(source/'images'/f'cam{view:02d}'/f'{f:06d}.png'))[...,3]>127 for f in sampled]
        box=crop(masks);x0,y0,x1,y1=box
        with np.load(GRID/'audit/roi'/label/f'view{view:02d}.npz') as z:labels=z['labels']
        with np.load(GRID/'audit/roi'/label/f'flow{view:02d}.npz') as z:flows,valid=z['backward'],z['valid']
        assert len(labels)==F
        previous={};previous_target=None
        writer=Video(media/f'view{view:02d}.mp4',W*(len(models)+1),H+30)
        for frame in range(F):
            path=source/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            rgba=np.asarray(Image.open(path),np.float32)/255;target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);fg=rgba[...,3]>.5
            tiles=[np.rint(target*255).astype(np.uint8)]
            trusted_by_part={}
            if frame:
                flow=flows[frame-1];target_motion=target-remap(previous_target,flow);previous_labels=remap(labels[frame-1],flow,True)
                trusted_by_part={p:(labels[frame]==p+1)&valid[frame-1]&(previous_labels==p+1) for p in range(6)}
            for name,(model,deform,pipe) in models.items():
                rgb,alpha=raw_render(model,deform,pipe,camera(data,view,frame));rgb=rgb.permute(1,2,0).cpu().numpy();alpha=alpha[0].cpu().numpy()
                assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);tiles.append(np.rint(pred*255).astype(np.uint8))
                error=np.square(pred-target).mean(-1)
                if frame:temporal_error=np.square((pred-remap(previous[name],flow))-target_motion).mean(-1)
                for p,part in enumerate(PARTS):
                    mask=labels[frame]==p+1;area=int(mask.sum());trusted=trusted_by_part.get(p)
                    pixels=int(trusted.sum()) if frame and area else 0
                    part_rows[name].append({'view':view,'frame':frame,'part':part,'reference_core_pixels':area,
                        'coverage':float(alpha[mask].mean()) if area else None,'mse':float(error[mask].mean()) if area else None,
                        'temporal_mse':float(temporal_error[trusted].mean()) if pixels else None,
                        'temporal_valid_pixels':pixels,'temporal_valid_fraction':pixels/area if frame and area else None})
                if frame in sampled:
                    a=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];b=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                    metric_rows[name].append({'view':view,'frame':frame,**scalar(pred,target,alpha,fg),
                        'ssim':float(ssim(a,b)),'crop_lpips':float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),
                        'crop_xyxy':box,'input_sha256':sha(path)})
                previous[name]=pred
            previous_target=target
            sheet=Image.new('RGB',(W*(len(models)+1),H+30),'#333333');draw=ImageDraw.Draw(sheet)
            for column,(name,image) in enumerate(zip(['Generated reference',*models],tiles)):
                sheet.paste(Image.fromarray(image),(column*W,30));draw.text((column*W+5,7),f'{name} | {label} v{view} f{frame}',fill='white')
            writer.append(np.asarray(sheet))
            if frame in [0,(F-1)//2,F-1]:
                sheet.save(media/f'v{view:02d}_f{frame:03d}.jpg',quality=93)
                if view==0:contacts.append(sheet.copy())
        writer.close()
    # Compare fixed denominators, crops and global A numbers against frozen historical reports.
    historical=json.loads((Path(baseline['results'])/'metrics.json').read_text())
    signature=lambda rs:sorted((r['view'],r['frame'],r['crop_xyxy'],r['input_sha256']) for r in rs)
    for name in models:assert signature(metric_rows[name])==signature(historical['samples'])
    summary={};fields=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']
    for name in models:
        means={k:float(np.mean([r[k] for r in metric_rows[name]])) for k in fields}
        if name=='A':assert all(abs(means[k]-historical['mean'][k])<1e-6 for k in fields),(label,means,historical['mean'])
        parts={}
        for part in PARTS:
            selected=[r for r in part_rows[name] if r['part']==part and r['coverage'] is not None];temporal=[r for r in selected if r['temporal_mse'] is not None]
            parts[part]={'coverage':float(np.mean([r['coverage'] for r in selected])) if selected else None,
                'compensated_rmse':math.sqrt(float(np.mean([r['temporal_mse'] for r in temporal]))) if temporal else None,
                'valid_slots':len(selected),'temporal_slots':len(temporal)}
        summary[name]={'mean':means,'parts':parts,'samples':metric_rows[name]}
        write_csv(out/f'{name}_parts.csv',part_rows[name])
    anchor=[(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in part_rows['A']]
    for name in models:assert [(r['view'],r['frame'],r['part'],r['reference_core_pixels'],r['temporal_valid_pixels']) for r in part_rows[name]]==anchor
    historical_parts=list(csv.DictReader((Path(baseline['results'])/'parts.csv').open()))
    assert [(int(r['view']),int(r['frame']),r['part'],int(r['reference_core_pixels']),int(r['temporal_valid_pixels'])) for r in historical_parts]==anchor
    for part in PARTS:
        old=[r for r in historical_parts if r['part']==part and r['coverage']!=''];oldtime=[r for r in old if r['temporal_mse']!='']
        if old:assert abs(summary['A']['parts'][part]['coverage']-float(np.mean([float(r['coverage']) for r in old])))<1e-6
        if oldtime:assert abs(summary['A']['parts'][part]['compensated_rmse']-math.sqrt(float(np.mean([float(r['temporal_mse']) for r in oldtime]))))<1e-6
    save(out/'metrics.json',{'state':'fixed_terminal_metrics_visual_review_pending','label':label,'methods':summary,'replays':replays,
        'denominator_checks':'identical39 global reference hashes/crops and all part/time denominators','device':device,'seconds':time.monotonic()-begin,
        'scope':'generated-reference H development evaluation, not real Waymo novel-view truth','ssim_source_sha256':ssim_sha})
    contact=Image.new('RGB',(contacts[0].width,contacts[0].height*len(contacts)))
    for i,image in enumerate(contacts):contact.paste(image,(0,i*image.height))
    contact.save(media/'contact.jpg',quality=93)
    print(label,'evaluation complete',json.dumps({n:v['mean'] for n,v in summary.items()}),flush=True)


if __name__=='__main__':main()
