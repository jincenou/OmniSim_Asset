"""Independent61-frame24-view decoding and source/root-time checks."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import av
import numpy as np

ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT'])
OUT=ROOT/'outputs/turbo'


def main():
    files=sorted((OUT/'videos').glob('*.mp4'))
    if len(files)!=24:raise ValueError('Expected24 generated views')
    cameras=json.loads((OUT/'cameras.json').read_text())
    if len(cameras['cameras'])!=24 or cameras['world_frame']['name']!='root_local_human_world':
        raise ValueError('Camera contract mismatch')
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    if len(mapping['frames'])!=61:raise ValueError('Source mapping has wrong length')
    if not np.allclose([x['video_seconds'] for x in mapping['frames']],np.arange(61)/25,atol=1e-9):
        raise ValueError('Source playback changed')
    motion=np.load(OUT/'root_motion.npz')
    if motion['local_to_canonical'].shape!=(61,4,4):raise ValueError('Root motion length changed')
    rows=[]
    for path in files:
        with av.open(str(path)) as con:
            stream=con.streams.video[0];stream.thread_count=1
            if (stream.width,stream.height,stream.average_rate)!=(704,1280,Fraction(25)):
                raise ValueError('Unexpected video format')
            count=0
            for i,frame in enumerate(con.decode(video=0)):
                if frame.pts is None or frame.pts*stream.time_base!=Fraction(i,25):
                    raise ValueError('Unexpected exact presentation time')
                count+=1
        if count!=61:raise ValueError('Incomplete output')
        rows.append({'path':str(path),'frames':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    device=json.loads((ROOT/'generation_device.json').read_text())
    if device['physical_gpus']!=[4,5,6,7] or len(device['actual_cuda_uuids'])!=4:
        raise ValueError('Device record mismatch')
    result={'state':'short24_decoded_source_root_times_verified_visual_review_pending',
        'frames':61,'views':24,'fps':25,'sample_span_s':2.4,'source_observations':25,
        'videos':rows,'device':device,'no_padding_to121':True,
        'scope':'Video-generation acceptance only; no dynamic Gaussian reconstruction yet.'}
    (OUT/'preview/short24_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'state':result['state'],'decoded_frames':24*61}),flush=True)


if __name__=='__main__':main()
