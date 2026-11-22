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
from evaluate_asset import Renderer,save,sha
from freetime_ops import positions_at_time,temporal_opacity
from gsplat import rasterization
from video_io import MP4Writer

ROUND=Path(__file__).resolve().parents[1];W=ROUND.parent
ROOT=ROUND/'experiments/walk063_rootlocal24_v1';ASSET=ROUND/'reconstruction/outputs/walk063_rootlocal24'
DATA=ROUND/'reconstruction/data/walk063_rootlocal24';SOURCE=W/'experiments/pilot_063'


@torch.inference_mode()
def render(renderer,cameras,Ks,frame,translation):
    s=renderer.s;t=frame/120
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


def fit_fixed_cameras(cameras,Ks,center,points):
    # Frame the body trajectory, never delete low-opacity/remote Gaussian points.
    shifted_center=(points.reshape(-1,3).min(0)+points.reshape(-1,3).max(0))/2
    def pose(scale):
        result=cameras.copy();result[:,:3,3]=shifted_center+(cameras[:,:3,3]-center)*scale;return result
    def inside(cams):
        w=np.linalg.inv(cams);p=np.einsum('vij,tkj->vtki',w[:,:3,:3],points)+w[:,None,None,:3,3]
        q=np.einsum('vij,vtkj->vtki',Ks,p);uv=q[...,:2]/q[...,2:]
        return bool(((p[...,2]>0)&(uv[...,0]>=.06*352)&(uv[...,0]<=.94*352)&(uv[...,1]>=.06*640)&(uv[...,1]<=.94*640)).all())
    low=.5;high=1.
    while not inside(pose(high)):
        high*=1.25
        if high>20:raise RuntimeError('Trajectory cannot fit a reasonable fixed camera')
    for _ in range(24):
        mid=(low+high)/2
        if inside(pose(mid)):high=mid
        else:low=mid
    return pose(high*1.001),{'camera_radius_scale':high*1.001,'center':shifted_center.tolist(),'framing_source':'all121 projected motion keypoints for both trajectories; full unfiltered model is rendered'}


def decoded(path):
    reader=imageio.get_reader(str(path),input_params=['-threads','1'])
    try:
        if reader.get_meta_data()['fps']!=25:raise ValueError('Source frame rate mismatch')
        for frame in reader:yield Image.fromarray(frame)
    finally:reader.close()


def pane(image,label):
    canvas=Image.new('RGB',(352,664),(25,25,25));im=ImageOps.contain(image,(352,640))
    canvas.paste(im,((352-im.width)//2,24+(640-im.height)//2));ImageDraw.Draw(canvas).text((5,5),label,fill='white');return np.asarray(canvas)


def main():
    physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in (6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();assert str(torch.cuda.get_device_properties(0).uuid)==expected
    out=ASSET/'demo';out.mkdir(exist_ok=False)
    renderer=Renderer(ASSET/'ckpts/ckpt_4999.pt',DATA)
    A=np.asarray(renderer.parser.transform,dtype=np.float64);Ai=np.linalg.inv(A)
    motion=np.load(ASSET/'root_motion.npz');times=motion['frame_times']
    if not np.allclose(times,np.arange(121)/25,atol=1e-8):raise ValueError('Root timeline is not the generated video timeline')
    G=motion['local_to_canonical'];assert np.allclose(G[:,:3,:3],np.eye(3))
    original=G[:,:3,3];u=np.linspace(0,1,121)
    new=np.stack([1.5*np.sin(np.pi*u),np.zeros_like(u),2*u],-1)
    originals_asset=original@A[:3,:3].T;news_asset=new@A[:3,:3].T
    keypoints=np.load(ROOT/'preflight/conditioning/keypoints_3d.npy')
    local=keypoints@A[:3,:3].T+A[:3,3]
    points=np.concatenate([local+originals_asset[:,None],local+news_asset[:,None]],axis=0)
    views=[0,6,12];poses=renderer.parser.camtoworlds[views].astype(np.float64)
    Ks=np.stack([renderer.parser.Ks_dict[renderer.parser.camera_ids[index]] for index in views])
    rig=json.loads((ROOT/'preflight/conditioning/cameras.json').read_text());center=np.asarray(rig['ring_center'])@A[:3,:3].T+A[:3,3]
    replay_cameras,framing=fit_fixed_cameras(poses,Ks,center,points)
    full_poses=renderer.parser.camtoworlds;angles=np.arange(25);closed=np.concatenate([full_poses,full_poses[:1]])
    orbit_at=np.linspace(0,24,72,endpoint=False);orbit=np.repeat(np.eye(4)[None],72,axis=0)
    orbit[:,:3,:3]=Slerp(angles,Rotation.from_matrix(closed[:,:3,:3]))(orbit_at).as_matrix()
    orbit[:,:3,3]=np.stack([np.interp(orbit_at,angles,closed[:,axis,3]) for axis in range(3)],-1)
    mapping=json.loads((SOURCE/'inputs/frame_mapping.json').read_text())
    native=json.loads((SOURCE/'inputs_native/frame_mapping.json').read_text())
    by_frame={row['frame_index']:row for row in native['frames']};raw_cache={};raw_hashes={};provenance=[]
    input_frames=decoded(ROOT/'inputs/main.mp4');generated_frames=decoded(ROOT/'outputs/turbo/videos/00.mp4')
    writers={};checks=[];coordinate_errors=[];foreground_checks=[]
    with ExitStack() as stack:
        for name,width,height in [('fixed_views',1056,640),('backgrounds',1056,640),('replay_original',1056,640),('replay_new',1056,640),('four_panel',1408,664),('orbit_fixed_time',352,640)]:
            writer=MP4Writer(str(out/f'{name}.mp4'),25,width,height);stack.callback(writer.close);writers[name]=writer
        for frame in range(121):
            rgb,alpha=render(renderer,poses,Ks,frame,np.zeros(3));local_images=composite(rgb,alpha)
            writers['fixed_views'].append_data(uint8(np.concatenate(local_images,axis=1)))
            backgrounds=np.concatenate([composite(rgb[:1],alpha[:1],bg)[0] for bg in (0.,.5,1.)],axis=1)
            writers['backgrounds'].append_data(uint8(backgrounds))
            gr,ga=render(renderer,replay_cameras,Ks,frame,originals_asset[frame]);nr,na=render(renderer,replay_cameras,Ks,frame,news_asset[frame])
            minimum=min(int((value>.1).sum((1,2,3)).min()) for value in (alpha,ga,na))
            if minimum<100:raise ValueError(f'Empty or near-empty replay view at frame{frame}')
            foreground_checks.append(minimum)
            writers['replay_original'].append_data(uint8(np.concatenate(composite(gr,ga),axis=1)))
            writers['replay_new'].append_data(uint8(np.concatenate(composite(nr,na),axis=1)))
            slot=mapping['frames'][frame];source=by_frame[slot['left_source_frame']];path=Path(source['source_path'])
            if str(path) not in raw_cache:
                raw_hashes[str(path)]=sha(path)
                with Image.open(path) as im:
                    original_image=im.convert('RGB');crop=original_image.crop(source['crop_xyxy']);ImageDraw.Draw(crop).rectangle(source['bbox_input_xyxy'],outline='lime',width=2)
                    inset=ImageOps.contain(original_image,(132,88));crop.paste(inset,(0,0));raw_cache[str(path)]=crop
            raw=raw_cache[str(path)];input_image=next(input_frames);generated=next(generated_frames)
            panels=[pane(raw,'Waymo crop + box (native held frame)'),pane(input_image,'Actual input (RIFE + target mask)'),pane(generated,'4DAnyone generated view0'),pane(Image.fromarray(uint8(local_images[0])),'GS: same generated camera, t')]
            writers['four_panel'].append_data(np.concatenate(panels,axis=1))
            provenance.append({'frame':frame,'video_seconds':frame/25,'left_source_frame':source['frame_index'],'source_path':str(path),'source_timestamp_micros':source['timestamp_micros'],'interpolated_input':slot['interpolated']})
            if frame in (0,60,120):
                print(f'Rendered walking frame{frame}/120',flush=True)
                reference,reference_alpha=renderer.render(0,frame)
                assert max(np.abs(reference-rgb[0]).max(),np.abs(reference_alpha-alpha[0,...,0]).max())<1e-5
                shifted=poses.copy();shifted[:,:3,3]+=originals_asset[frame]
                rr,ra=render(renderer,shifted,Ks,frame,originals_asset[frame])
                error=max(float(np.abs(rr-rgb).max()),float(np.abs(ra-alpha).max()))
                if error>2e-4:raise ValueError(f'Joint camera/geometry translation changed rendering: {error}')
                checks.append({'frame':frame,'joint_translation_render_max_abs':error})
                means=positions_at_time(renderer.s['means'],renderer.s['velocities'],renderer.s['times'],frame/120,True).detach().cpu().numpy()
                recovered=(means+originals_asset[frame])@Ai[:3,:3].T+Ai[:3,3]
                expected_local=means@Ai[:3,:3].T+Ai[:3,3]+original[frame]
                coordinate_errors.append(float(np.max(np.abs(recovered-expected_local))))
                Image.fromarray(uint8(backgrounds)).save(out/f'backgrounds_frame{frame:03d}.jpg',quality=95)
        for camera in orbit:
            rgb,alpha=render(renderer,camera[None],Ks[:1],60,np.zeros(3));writers['orbit_fixed_time'].append_data(uint8(composite(rgb,alpha)[0]))
    # Decode all outputs; playback remains25fps and orbit time is explicitly fixed.
    videos={}
    for path in out.glob('*.mp4'):
        reader=imageio.get_reader(str(path),input_params=['-threads','1'])
        try:
            metadata=reader.get_meta_data();count=sum(1 for _ in reader)
            assert count==(72 if path.stem=='orbit_fixed_time' else 121) and metadata['fps']==25
            videos[path.name]={'frames':count,'fps':25,'size':metadata['size']}
        finally:reader.close()
    np.savez_compressed(out/'replay_cameras_and_trajectories.npz',fixed_local_cameras=poses,fixed_global_cameras=replay_cameras,K=Ks,
        local_to_asset=A,original_root_translation=original,synthetic_translation=new,original_asset_translation=originals_asset,
        synthetic_asset_translation=news_asset,orbit_camera_to_asset=orbit,frame_seconds=times,orbit_model_time=np.full(72,.5))
    report={'state':'rendered_and_decoded_visual_review_pending','physical_gpu':physical,'cuda_uuid':str(torch.cuda.get_device_properties(0).uuid),
        'videos':videos,'global_framing':framing,'translation_checks':checks,'coordinate_recomposition_max_abs':max(coordinate_errors),
        'minimum_foreground_pixels_across_local_and_replay_views':min(foreground_checks),'native_source_hashes':raw_hashes,
        'checkpoint_sha256':sha(ASSET/'ckpts/ckpt_4999.pt'),'input_video_sha256':sha(ROOT/'inputs/main.mp4'),
        'generated_view0_sha256':sha(ROOT/'outputs/turbo/videos/00.mp4'),
        'root_motion_sha256':sha(ASSET/'root_motion.npz'),'source_timeline':provenance,
        'transform_rule':'Global means_asset(t) += A_linear @ delta_local(t); implemented with equivalent camera translation -delta in float64 before CUDA float32 projection. Nonlinear root trajectory remains external to FTGS linear velocities.',
        'covariance_rotation_SH':'Both replay paths use identity external rotation: covariance, quaternions and SH coefficients unchanged. SH ray directions are evaluated from translated positions by rasterizer.',
        'new_path':'synthetic fixed-heading curve x=1.5sin(pi*u), y=0,z=2u in estimated local units; not measured motion or arbitrary pose retargeting',
        'display':'No Gaussian filtering. Source panel uses native left observation at each model time; input interpolation remains labeled. Only generated and GS panels share target camera.',
        'geometry_orbit':'72 changed camera poses at fixed model frame60; not an action video'}
    save(out/'replay_validation.json',report);print(json.dumps({'state':report['state'],'videos':videos}),flush=True)


if __name__=='__main__':main()
