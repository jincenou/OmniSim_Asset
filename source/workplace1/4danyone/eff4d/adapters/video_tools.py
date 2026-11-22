import subprocess,json,math
from pathlib import Path
import numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageOps
from data import output_path,sha
class Video:
 def __init__(self,path,width,height,fps=25):
  self.path=output_path(path);self.width=width;self.height=height;self.frames=0;self.fps=fps
  self.proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
 def add(self,arr):
  assert arr.shape==(self.height,self.width,3);self.proc.stdin.write(np.ascontiguousarray(arr,dtype='uint8').tobytes());self.frames+=1
 def close(self):
  self.proc.stdin.close();code=self.proc.wait();assert code==0,(self.path,code)
  decoded=imageio_ffmpeg.read_frames(str(self.path),pix_fmt='rgb24',input_params=['-threads','1']);meta=next(decoded);count=sum(1 for _ in decoded)
  assert count==self.frames and abs(meta['fps']-self.fps)<1e-3 and meta['size']==(self.width,self.height)
  return {'path':str(self.path),'frames':self.frames,'fps':self.fps,'sha256':sha(self.path),'full_decode':True}

def uint8(x):return np.round(np.clip(x,0,1)*255).astype('uint8')
def panel(image,text):
 im=Image.fromarray(image) if isinstance(image,np.ndarray) else image;canvas=Image.new('RGB',(352,680),(28,28,28));im=ImageOps.contain(im,(352,640));canvas.paste(im,((352-im.width)//2,40+(640-im.height)//2));ImageDraw.Draw(canvas).text((5,4),text,fill='white');return np.array(canvas)
def camera_orbit(cameras,num=120):
 from scipy.spatial.transform import Rotation,Slerp
 cs=sorted(cameras,key=lambda c:c['yaw']);angles=np.array([c['yaw'] for c in cs]+[360.]);poses=np.array([c['c2w'] for c in cs]+[cs[0]['c2w']]);slerp=Slerp(angles,Rotation.from_matrix(poses[:,:3,:3]));out=[]
 for angle in np.linspace(0,360,num,endpoint=False):
  c=dict(cs[0]);pose=np.eye(4);pose[:3,:3]=slerp(angle).as_matrix();pose[:3,3]=[np.interp(angle,angles,poses[:,k,3]) for k in range(3)];c['c2w']=pose.tolist();out.append(c)
 return out

def crop_bounds(masks):
 ys,xs=np.where(np.logical_or.reduce(masks));assert len(xs)>0;h,w=masks[0].shape;pad=max(4,round(.05*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)));return [max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(w,int(xs.max())+pad+1),min(h,int(ys.max())+pad+1)]

