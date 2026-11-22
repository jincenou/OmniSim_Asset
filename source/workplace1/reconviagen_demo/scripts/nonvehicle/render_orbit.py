"""Render original-loader Gaussians under shared metric cameras with gsplat."""
from pathlib import Path
import sys,os,argparse,json,math
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/mnt/wmj/workspace4/pipeline')
from curate.core import load_rigid

def main():
    p=argparse.ArgumentParser();p.add_argument('--ply',required=True);p.add_argument('--output',required=True)
    p.add_argument('--length',type=float,default=1.);p.add_argument('--size',type=int,default=512)
    p.add_argument('--distance',type=float,default=5.);p.add_argument('--focal',type=float,default=700.)
    p.add_argument('--video-frames',type=int,default=96)
    a=p.parse_args();from gsplat import rasterization
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    gs=load_rigid(Path(a.ply));xyz=gs.xyz[0].copy()
    extent=np.quantile(xyz[:,0],.995)-np.quantile(xyz[:,0],.005)
    if extent<=0:raise ValueError('Invalid length')
    factor=1.0
    tensors=[torch.as_tensor(v,device='cuda',dtype=torch.float32) for v in
             [xyz*factor,gs.rot[0],gs.scale[0]*factor,gs.opacity,gs.color]]
    means,quats,scales,opacity,color=tensors;quats=quats/quats.norm(dim=-1,keepdim=True)
    if a.video_frames<16 or a.video_frames%16:raise ValueError('video-frames must be a positive multiple of 16')
    lo,hi=np.quantile(xyz,[.005,.995],axis=0);center=(lo+hi)/2;up=np.array([0,0,1.]);tiles=[];video_frames=[]
    for i in range(a.video_frames):
        az=2*math.pi*i/a.video_frames;el=math.radians(12)
        eye=center+a.distance*np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
        forward=(center-eye)/np.linalg.norm(center-eye);right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
        r=np.stack([right,down,forward]);v=np.eye(4);v[:3,:3]=r;v[:3,3]=-r@eye
        k=np.array([[a.focal,0,a.size/2],[0,a.focal,a.size/2],[0,0,1.]])
        with torch.no_grad():
            rgb,alpha,meta=rasterization(means,quats,scales,opacity,color,
                torch.tensor(v[None],device='cuda',dtype=torch.float32),torch.tensor(k[None],device='cuda',dtype=torch.float32),
                a.size,a.size,backgrounds=torch.full((1,3),.5,device='cuda'),packed=False)
        frame=(rgb[0,:,:,:3].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        amask=(alpha[0,:,:,0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        video_frames.append(frame)
        if i%(a.video_frames//16)==0:
            j=i//(a.video_frames//16)
            imageio.imwrite(out/f'view_{j:02}.png',frame);imageio.imwrite(out/f'alpha_{j:02}.png',amask)
            tiles.append(frame)
    sheet=Image.new('RGB',(a.size*4,(a.size+30)*4),'#ccc');draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(tiles):
        x=i%4*a.size;y=i//4*(a.size+30);sheet.paste(Image.fromarray(frame),(x,y+30));draw.text((x+5,y+5),f'{22.5*i:.1f} deg',fill='black')
    sheet.save(out/'turntable.jpg',quality=92)
    imageio.mimwrite(out/'turntable.mp4',video_frames,fps=16,codec='libx264',quality=8)
    (out/'render_config.json').write_text(json.dumps(dict(ply=str(Path(a.ply).resolve()),units='arbitrary, max robust dimension 2',metric_scale_known=False,normalization_scale=factor,
        distance_m=a.distance,focal_px=a.focal,size=a.size,elevation_deg=12,center_m=center.tolist(),background=.5,
        azimuths_deg=[22.5*i for i in range(16)],video_frames=a.video_frames,video_fps=16,loader='curate.core.load_rigid',kernel='gsplat.rasterization',
        scope='standalone orbit; not scene insertion'),indent=2))

if __name__=='__main__':main()
