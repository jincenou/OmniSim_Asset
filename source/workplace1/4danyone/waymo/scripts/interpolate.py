"""One explicit RIFE 4.25 branch: 49 real 10 Hz frames -> 121 slots at 25 Hz."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()
sys.path[:0]=[str(BASE/'rife_model'),str(BASE/'rife')]
from train_log.IFNet_HDv3 import IFNet
from prepare import encode


def main():
    assert torch.cuda.is_available()
    source=json.loads(Path(os.environ.get('FDANYONE_RIFE_SOURCE',str(ROOT/'inputs_C00/frame_mapping.json'))).read_text())
    records=source['frames'][:49]
    rgb=[np.array(Image.open(x['source_path']).convert('RGB').crop(tuple(x['crop_xyxy']))) for x in records]
    encode(ROOT/'inputs/raw_49frames_10fps.mp4',[Image.fromarray(x) for x in rgb],10)
    model=IFNet().eval().cuda()
    weights=torch.load(BASE/'rife_model/train_log/flownet.pkl',map_location='cpu',weights_only=True)
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    # The official archive includes the teacher/caltime training-only branches,
    # explicitly commented out in its inference IFNet. Account for these exact
    # prefixes, then require every inference parameter to match strictly.
    training_only=[k for k in weights if k.startswith(('teacher.','caltime.'))]
    weights={k:v for k,v in weights.items() if k not in training_only}
    model.load_state_dict(weights,strict=True)
    del weights
    h,w=rgb[0].shape[:2]
    tensors=[F.pad(torch.from_numpy(x[:,:,::-1].copy()).permute(2,0,1).float().div(255).unsqueeze(0),(0,(-w)%64,0,(-h)%64)) for x in rgb]
    output=[None]*121; mapping=[]; pending=[]
    for slot in range(121):
        left,rem=divmod(slot*2,5)
        right=left if rem==0 else left+1
        alpha=rem/5
        entry=dict(records[left]);entry.update({'input_frame':slot,'model_slot':slot,'video_seconds':slot/25,
            'left_source_frame':records[left]['frame_index'],'right_source_frame':records[right]['frame_index'],
            'source_pair_timestamp_micros':[records[left]['timestamp_micros'],records[right]['timestamp_micros']],
            'camera_pose_source':'left real source frame; not an interpolated pose estimate',
            'right_source_camera_to_world_file':records[right]['source_camera_to_world_file'],
            'interpolation_fraction':alpha,'interpolated':bool(rem),'enhanced':False,
            'bbox_annotation_source':'original' if rem==0 else 'linear interpolation of neighboring dataset boxes',
            'bbox_input_xyxy':((1-alpha)*np.array(records[left]['bbox_input_xyxy'])+alpha*np.array(records[right]['bbox_input_xyxy'])).tolist(),
            'relative_seconds':(1-alpha)*records[left]['relative_seconds']+alpha*records[right]['relative_seconds']})
        entry.pop('timestamp_micros',None)
        mapping.append(entry)
        if rem==0:output[slot]=rgb[left]
        else:pending.append((slot,left,right,alpha))
    with torch.inference_mode():
        for start in range(0,len(pending),4):
            group=pending[start:start+4]
            a=torch.cat([tensors[x[1]] for x in group]).cuda();b=torch.cat([tensors[x[2]] for x in group]).cuda()
            t=torch.tensor([x[3] for x in group],device='cuda').reshape(-1,1,1,1)
            prediction=model(torch.cat((a,b),1),t,[16,8,4,2,1])[2][-1]
            result=prediction[:,:,:h,:w].clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()[:,:,:,::-1]
            for desc,im in zip(group,result,strict=True):output[desc[0]]=im.copy()
    assert all(x is not None for x in output)
    encode(ROOT/'inputs/main.mp4',[Image.fromarray(x) for x in output],25)
    source.update({'candidate_id':source['candidate_id']+'_short_rife','frames':mapping,'branch':'RIFE4.25 arbitrary timestep; no image enhancement',
        'real_source_frames':49,'output_frames':121,'real_source_frame_range':[records[0]['frame_index'],records[-1]['frame_index']],'native_fps':10,'output_fps':25,
        'true_span_seconds':records[-1]['relative_seconds'],'encoded_sample_span_seconds':4.8,'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','5').split(',')[0])})
    (ROOT/'inputs/frame_mapping.json').write_text(json.dumps(source,indent=2)+'\n')
    provenance={'model':'official RIFE4.25','source_code_commit':'bbfd2ea90910789a860ea3e2b32a240cd577b75e',
        'archive_record':str(BASE/'research/rife-download.json'),'strict_inference_state_dict_load':True,'training_only_keys_excluded':training_only,'physical_gpu':source['physical_gpu'],'original_frames_used_directly':25,
        'interpolated_output_frames':96,'source_frames':49,'source_seconds':records[-1]['relative_seconds'],'output_sample_span_seconds':4.8,
        'max_time_regularization_seconds':max(abs(x['relative_seconds']-x['video_seconds']) for x in mapping),
        'source_color':'RGB images converted to BGR for official IFNet, then converted back','padding':[0,(-w)%64,0,(-h)%64],
        'limitations':'Inserted frames are model estimates, not new Waymo observations. No cross-gap/camera/identity interpolation.'}
    (ROOT/'research/interpolation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
