"""Unfiltered walking demos and external nonlinear translation trajectories."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation,Slerp
import torch
from evaluate_asset61 import Renderer,save,sha
from freetime_ops import positions_at_time,temporal_opacity
from gsplat import rasterization
from video_io import MP4Writer

ROUND=Path(__file__).resolve().parents[1];W=ROUND.parent


@torch.inference_mode()
def render(renderer,cameras,Ks,frame,translation):
    s=renderer.s;t=frame/60
    means=positions_at_time(s['means'],s['velocities'],s['times'],t,renderer.state['use_velocity'])
    # Evaluate in a local origin: translating geometry by d is exactly equivalent
    # to translating cameras by -d. Subtract in float64 before the float32 CUDA
    # projection, avoiding cancellation when both cameras and points move far away.
    local_cameras=np.asarray(cameras,dtype=np.float64).copy()
    local_cameras[:,:3,3]-=np.asarray(translation,dtype=np.float64)
    opacity=(s['opacities'].sigmoid()*temporal_opacity(s['times'],s['durations'],t)).clamp_min(1e-4)
    rgb,alpha,_=rasterization(means=means,quats=s['quats'],scales=s['scales'].exp(),opacities=opacity,
        colors=torch.cat([s['sh0'],s['shN']],1),viewmats=torch.linalg.inv(torch.as_tensor(local_cameras,device='cuda',dtype=torch.float32)),
        Ks=torch.as_tensor(Ks,device='cuda',dtype=torch.float32),width=352,height=640,sh_degree=renderer.cfg['sh_degree'],
        packed=renderer.cfg['packed'],near_plane=renderer.cfg['near_plane'],far_plane=renderer.cfg['far_plane'],
        rasterize_mode='antialiased' if renderer.cfg['antialiased'] else 'classic')
    return rgb.cpu().numpy(),alpha.cpu().numpy()


def composite(rgb,alpha,background=1.):return np.clip(rgb+(1-alpha)*background,0,1)
def uint8(image):return np.round(image*255).astype(np.uint8)


def decoded(path):
    reader=imageio.get_reader(str(path),input_params=['-threads','1'])
    try:
        if reader.get_meta_data()['fps']!=25:raise ValueError('Source frame rate mismatch')
        for frame in reader:yield Image.fromarray(frame)
    finally:reader.close()


def pane(image,label):
    canvas=Image.new('RGB',(352,664),(25,25,25));im=ImageOps.contain(image,(352,640))
    canvas.paste(im,((352-im.width)//2,24+(640-im.height)//2));ImageDraw.Draw(canvas).text((5,5),label,fill='white');return np.asarray(canvas)

