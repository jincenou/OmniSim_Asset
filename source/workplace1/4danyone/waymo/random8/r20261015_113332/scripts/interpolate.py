"""RIFE interpolation with matching per-frame crop K and camera rotation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch.nn import functional as F
from m02_prepare import R,ROOT,save,sha

W=Path('/mnt/ojc/workplace1/4danyone/waymo');P=W.parent
sys.path[:0]=[str(W/'rife_model'),str(W/'rife'),str(W/'scripts'),str(P/'repo')]
from train_log.IFNet_HDv3 import IFNet
from prepare import encode


def device_check():
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (4,5,6,7):raise ValueError('Unauthorized physical device')
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual!=expected or torch.cuda.device_count()!=1:raise RuntimeError('Actual CUDA UUID mismatch')
    return {'physical_gpu':gpu,'actual_cuda_uuid':actual,'pid':os.getpid()}


def main():
    device=device_check();source=json.loads((ROOT/'native.json').read_text());records=source['frames']
    if len(records) not in (25,49) or sha(ROOT/'native_cameras.npz')!=source['camera_npz_sha256']:raise ValueError('Native source mismatch')
    output=ROOT/'inputs';output.mkdir(exist_ok=False);save(output/'device.json',device)
    rgb=[]
    for record in records:
        path=Path(record['path'])
        if sha(path)!=record['sha256']:raise ValueError('Native image changed')
        rgb.append(np.asarray(Image.open(path).convert('RGB')))
    model=IFNet().eval().cuda();weight_path=W/'rife_model/train_log/flownet.pkl'
    weights=torch.load(weight_path,map_location='cpu',weights_only=True)
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    training_only=[k for k in weights if k.startswith(('teacher.','caltime.'))]
    model.load_state_dict({k:v for k,v in weights.items() if k not in training_only},strict=True);del weights
    h,w=rgb[0].shape[:2]
    tensors=[F.pad(torch.from_numpy(x[:,:,::-1].copy()).permute(2,0,1).float().div(255).unsqueeze(0),(0,(-w)%64,0,(-h)%64)) for x in rgb]
    count=(len(records)-1)*5//2+1
    images=[None]*count;pending=[];mapping=[];times=[];affines=[];intrinsics=[]
    native=np.load(ROOT/'native_cameras.npz');native_time=native['image_center_timestamps_s']
    # Relative times avoid subtracting large Unix timestamps inside Slerp.
    relative=native_time-native_time[0]
    if not np.isfinite(relative).all() or not np.all((np.diff(relative)>=.08)&(np.diff(relative)<=.12)):raise ValueError('Illegal exposure-time continuity')
    nominal_span=(count-1)/25
    if nominal_span-relative[-1]>.001:raise ValueError('Exposure interval lacks coverage beyond frozen1ms timing tolerance')
    sampled_span=min(float(relative[-1]),nominal_span)
    # Never speed up source motion. A submillisecond-short interval is uniformly slowed,not padded.
    source_times=np.linspace(0,sampled_span,count)
    for slot in range(count):
        t=float(source_times[slot])
        right=int(np.searchsorted(relative,t,side='left'))
        if right<len(relative) and abs(relative[right]-t)<1e-8:left=right;alpha=0.
        else:left=right-1;alpha=float((t-relative[left])/(relative[right]-relative[left]))
        if not (0<=left<=right<len(relative) and 0<=alpha<=1):raise ValueError('Exposure interpolation outside source observations')
        times.append(t)
        A=(1-alpha)*native['undistorted_to_input'][left]+alpha*native['undistorted_to_input'][right]
        K=(1-alpha)*native['K_input'][left]+alpha*native['K_input'][right]
        affines.append(A);intrinsics.append(K)
        mapping.append({'slot':slot,'video_seconds':slot/25,'source_relative_exposure_seconds':float(t),
            'left_native_index':left,'right_native_index':right,'left_source_frame':records[left]['frame_index'],
            'right_source_frame':records[right]['frame_index'],'interpolation_fraction':alpha,'interpolated':bool(right!=left),
            'source_pair_image_center_timestamps_s':[float(native_time[left]),float(native_time[right])],
            'bbox_input_xyxy':((1-alpha)*np.asarray(records[left]['bbox_input_xyxy'])+alpha*np.asarray(records[right]['bbox_input_xyxy'])).tolist(),
            'undistorted_to_input':A.tolist(),'K_input':K.tolist(),
            'geometry_scope':'Interpolated camera/crop metadata describes model-estimated intermediate frames; it is not a new measured image'})
        if left==right:images[slot]=rgb[left]
        else:pending.append((slot,left,right,alpha))
    with torch.inference_mode():
        for start in range(0,len(pending),4):
            batch=pending[start:start+4]
            a=torch.cat([tensors[x[1]] for x in batch]).cuda();b=torch.cat([tensors[x[2]] for x in batch]).cuda()
            t=torch.tensor([x[3] for x in batch],device='cuda').reshape(-1,1,1,1)
            prediction=model(torch.cat((a,b),1),t,[16,8,4,2,1])[2][-1]
            predicted=prediction[:,:,:h,:w].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()[:,:,:,::-1]
            for description,im in zip(batch,predicted,strict=True):images[description[0]]=im.copy()
    if any(x is None for x in images):raise RuntimeError('Missing interpolated frame')
    encode(output/'main.mp4',[Image.fromarray(x) for x in images],25)
    from fractions import Fraction
    from fdanyone.video import decode_canonical_clip
    clip=decode_canonical_clip(output/'main.mp4',num_frames=count,start_time=0,fps=Fraction(25))
    if any(not np.array_equal(actual.rgb,expected) for actual,expected in zip(clip.frames,images,strict=True)):
        raise RuntimeError('Encoded interpolation differs from actual output pixels')
    clip.write_metadata(output/'canonical_clip.json')
    rotations=Slerp(relative,Rotation.from_matrix(native['R_world_to_cv_camera']))(times).as_matrix()
    np.savez_compressed(output/'cameras.npz',K_input=intrinsics,R_world_to_cv_camera=rotations,
        undistorted_to_input=affines,source_relative_exposure_seconds=times,video_seconds=np.arange(count)/25)
    save(output/'frame_mapping.json',{'preprocessing_protocol':'random8-frontend-v1.2','state':'interpolated_camera_aware_input_ready','frames':mapping,'native_source':str(ROOT/'native.json'),
        'native_source_sha256':sha(ROOT/'native.json'),'camera_npz_sha256':sha(output/'cameras.npz'),
        'video_sha256':sha(output/'main.mp4'),'native_frames':len(records),'direct_native_output_slots':sum(not row['interpolated'] for row in mapping),'interpolated_slots':sum(row['interpolated'] for row in mapping),
        'sample_span_s':(count-1)/25,'sampled_source_span_s':sampled_span,'source_speed_ratio':sampled_span/nominal_span,'speedup':False,'endpoint_padding':False,'real_exposure_span_s':float(relative[-1]),
        'max_time_regularization_s':float(np.abs(np.asarray(times)-np.arange(count)/25).max()),
        'RIFE_weights_sha256':sha(weight_path),'training_only_keys_excluded':training_only,
        'camera_resampling':'Output25fps; measured-exposure brackets and fractions; source-span=min(measured,nominal),no acceleration; sub1ms deficit uses uniform slowdown,not repeated endpoints; SO3 Slerp/cropK at sampled actualsource times',
        'scope':'Known-person motion diagnostic, unsegmented background; no generation or world-motion accuracy claim',
        'device':device,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    print(json.dumps({'state':'interpolated_input_verified','frames':count,'device':device}),flush=True)


if __name__=='__main__':main()
