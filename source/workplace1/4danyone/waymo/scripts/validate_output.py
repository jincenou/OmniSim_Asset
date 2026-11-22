import json
import os
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import av

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()


def main(mode):
    out=ROOT/'outputs'/mode
    validation=json.loads((out/'preview/validation.json').read_text())
    assert validation['status']=='validated'
    expected=6 if mode=='smoke' else 24
    paths=sorted((out/'videos').glob('*.mp4'))
    assert len(paths)==expected
    for path in paths:
        with av.open(str(path)) as con:
            stream=con.streams.video[0]
            stream.thread_count=1
            assert stream.average_rate==Fraction(25) and (stream.width,stream.height)==(704,1280)
            count=0
            for index,frame in enumerate(con.decode(video=0)):
                assert frame.pts is not None and frame.pts*stream.time_base==Fraction(index,25), (path,index,frame.pts)
                count+=1
            assert count==121, (path,count)
    with av.open(str(out/'preview/multiview.mp4')) as con:
        stream=con.streams.video[0];stream.thread_count=1
        assert stream.average_rate==Fraction(25)
        assert sum(1 for _ in con.decode(video=0))==121
    result={'state':'validated','mode':mode,'raw_video_headers_checked':expected,'raw_full_decode_evidence':'preview/validation.json',
        'raw_decoded_pts_checked':expected*121,'synchronization':'Every raw decoded PTS equals frame_index / 25 seconds, all views',
        'preview_frames':121,'fps':25,'physical_gpu_ids':list(map(int,os.environ.get('CUDA_VISIBLE_DEVICES','4' if mode=='smoke' else '4,5,6,7').split(','))),
        'visual_review':'pending','updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    (out/'preview/final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    status=json.loads((ROOT/'status.json').read_text());status.update({'stage':mode,'state':'validated_visual_review_pending','updated_at':result['updated_at']})
    (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main(sys.argv[1])
