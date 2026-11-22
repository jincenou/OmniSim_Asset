"""Explicit data adapter for the author CEM implementation; old experiments are read-only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import lru_cache
import ctypes,hashlib,json,os,random,sys,uuid
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'random8/r20261015_113332'
REPO=ROOT/'sources/cem4dgs'
LABELS=['N01','N02','N03','N04','N06','N07','N08','N11']
sys.path.insert(0,str(REPO))

def stamp(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)

def manifest(label):
    assert label in LABELS
    path=BASE/'four_dg/data'/label/'fdanyone.json'
    data=json.loads(path.read_text())
    assert data['heldout_views']==[0,8,16] and len(data['train_views'])==21
    assert not set(data['heldout_views'])&set(data['train_views'])
    assert data['time_denominator']==data['frame_count']-1
    assert (data['width'],data['height'])==(352,640)
    return data,path

@lru_cache(maxsize=1536)
def image_bytes(path):
    with Image.open(path) as im:
        assert im.mode=='RGBA' and im.size==(352,640)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)

def read_image(path,resolution,im_scale):
    assert tuple(resolution)==(352,640) and im_scale==1
    rgba=image_bytes(str(path)).float()/255
    return rgba[:3]*rgba[3:]+1-rgba[3:]

def camera(data,view,frame,pose=None):
    from scene.cameras import Cameravideo
    from utils.graphics_utils import focal2fov
    record=next(c for c in data['cameras'] if c['view']==view)
    K=np.asarray(record['K'],np.float64)
    w2c=np.linalg.inv(np.asarray(record['camtoworld'] if pose is None else pose,np.float64))
    path=Path(data['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
    cam=Cameravideo(colmap_id=view,R=w2c[:3,:3].T,T=w2c[:3,3],
        FoVx=focal2fov(K[0,0],352),FoVy=focal2fov(K[1,1],640),
        gt_alpha_mask=None,image=None,image_name=f'cam{view:02d}/{frame:06d}',
        image_path=str(path),uid=view*data['frame_count']+frame,data_device='cpu',
        near=.01,far=100.,timestamp=float(frame),resolution=(352,640))
    # Author CUDA rasterizer's ndc2Pix uses ((ndc+1)*size-1)/2.
    cam.projection_matrix[2,0]=(2*K[0,2]+1)/352-1
    cam.projection_matrix[2,1]=(2*K[1,2]+1)/640-1
    cam.full_proj_transform=cam.world_view_transform@cam.projection_matrix
    cam.source_view=view;cam.source_frame=frame
    return cam

from scene import Scene as NativeScene
class WaymoScene(NativeScene):
    def __init__(self,args,gaussians,**kwargs):
        self.data,self.manifest_path=manifest(args.label)
        self.gaussians=gaussians;self.model_path=args.model_path
        self.cameras_extent=self.data['scene_radius'];self.sample_len=0;self.last_sample_len=0
        self.min_timestamp=0;self.start_timestamp=0;self.end_timestamp=self.data['frame_count']
        self.lazy_loader=True;self.loaded_iter=None;self.cam_num=21
        self.train_cameras={1.0:[camera(self.data,v,f) for f in range(self.data['frame_count']) for v in self.data['train_views']]}
        # Native training evaluation is disabled. Heldout is only constructed by independent evaluator.
        self.test_cameras={1.0:[]};self.test_samplelist=[]
        with np.load(self.manifest_path.parent/self.data['canonical_npz']) as z:
            from utils.graphics_utils import BasicPointCloud
            cloud=BasicPointCloud(points=z['points'].copy(),colors=z['colors'].copy(),normals=np.zeros_like(z['points']),times=np.zeros((len(z['points']),1),np.float32))
        assert len(cloud.points)==self.data['initial_points'] and np.isfinite(cloud.points).all()
        gaussians.create_from_pcd(cloud,self.cameras_extent)
        save(Path(self.model_path)/'input.json',{'manifest':str(self.manifest_path),'sha256':sha(self.manifest_path),
             'initialization_sha256':sha(self.manifest_path.parent/self.data['canonical_npz']),
             'train_views':self.data['train_views'],'heldout_views':self.data['heldout_views'],
             'frame_count':self.data['frame_count'],'training_background':1.0})

    def getTrainCameras(self,scale=1.,shuffle=True,return_as='generator',return_path=False,get_img=True,**kwargs):
        cams=self.getTCams(scale,'train')
        if shuffle: random.shuffle(cams)
        assert all(c.colmap_id not in [0,8,16] for c in cams)
        paths=[(c.image_path,c.resolution,c.im_scale) for c in cams]
        if return_path: return cams,paths
        imgs=map(lambda x: read_image(*x),paths) if get_img else None
        if return_as=='list' and imgs is not None: imgs=list(imgs)
        return cams,imgs

def device():
    expected=os.environ['CUDA_VISIBLE_DEVICES']
    assert expected.startswith('GPU-') and ',' not in expected
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    # CUDA-visible logical device zero must be the UUID explicitly selected by launcher.
    torch.empty(1,device='cuda')
    driver=ctypes.CDLL('libcuda.so.1');ordinal=ctypes.c_int()
    driver.cuCtxGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype=ctypes.c_int
    assert driver.cuCtxGetDevice(ctypes.byref(ordinal))==0
    value=(ctypes.c_ubyte*16)()
    driver.cuDeviceGetUuid.argtypes=[ctypes.c_void_p,ctypes.c_int]
    driver.cuDeviceGetUuid.restype=ctypes.c_int
    assert driver.cuDeviceGetUuid(ctypes.byref(value),ordinal.value)==0
    actual='GPU-'+str(uuid.UUID(bytes=bytes(value)))
    assert actual==expected,(actual,expected)
    return {'uuid':actual,'name':torch.cuda.get_device_name(0),'pid':os.getpid()}
