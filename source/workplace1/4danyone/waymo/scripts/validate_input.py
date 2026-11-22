"""Check time mapping and lossless preservation of genuine observations."""
import json
import os
from pathlib import Path
from fractions import Fraction
import hashlib
import av
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()


def main():
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    composited='background_compositing' in mapping
    masks=np.load(ROOT/mapping['background_compositing']['mask_source'])['masks'] if composited else None
    crop=mapping['frames'][0]['crop_xyxy'];width,height=crop[2]-crop[0],crop[3]-crop[1]
    padding=mapping.get('input_padding_ltrb',[0,0,0,0])
    width+=padding[0]+padding[2];height+=padding[1]+padding[3]
    real=0;hashes=[];times=[]
    with av.open(str(ROOT/'inputs/main.mp4')) as video:
        stream=video.streams.video[0];stream.thread_count=1
        assert stream.average_rate==Fraction(25)
        for idx,frame in enumerate(video.decode(video=0)):
            record=mapping['frames'][idx]
            rgb=frame.to_ndarray(format='rgb24')
            assert rgb.shape==(height,width,3)
            t=float(frame.pts*frame.time_base);times.append(t)
            assert abs(t-idx/25)<1e-6
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            if not record['interpolated']:
                source=np.array(Image.open(record['source_path']).convert('RGB').crop(tuple(record['crop_xyxy'])))
                if composited:
                    alpha=masks[idx].astype(np.float32)[...,None]/255
                    source=np.rint(source*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)
                if any(padding):source=np.pad(source,((padding[1],padding[3]),(padding[0],padding[2]),(0,0)),constant_values=255)
                assert np.array_equal(source,rgb),f'Original observation changed at slot {idx}'
                real+=1
    assert len(times)==121 and real==25
    result={'state':'passed','fps':25,'frames':121,'size':[width,height],'lossless_expected_slots_checked':real,
            'checked_against':'original RGB plus recorded alpha composite' if composited else 'original RGB',
            'inserted_model_estimates':96,'duplicate_decoded_frames':len(hashes)-len(set(hashes)),
            'first_last_pts_seconds':[times[0],times[-1]],'sample_span_seconds':times[-1]-times[0],
            'max_source_time_error_seconds':max(abs(r['relative_seconds']-r['video_seconds']) for r in mapping['frames'])}
    (ROOT/'research/input-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
