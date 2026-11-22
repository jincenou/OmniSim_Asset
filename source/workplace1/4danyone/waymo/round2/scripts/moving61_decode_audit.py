"""Independent full decoding/PTS audit; does not overwrite visual rejection."""
import asyncio
import argparse
from fractions import Fraction
import hashlib
import json
import sys
from pathlib import Path
import av

R = Path(__file__).resolve().parents[1]

def check(mode):
    out = R/'experiments'/f'moving_R01_short61_{mode}_v1'/'outputs/smoke'
    paths = sorted((out/'videos').glob('*.mp4'))
    if len(paths) != 6:
        raise ValueError('Expected six generated views')
    rows = []
    for path in paths:
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            stream.thread_count = 1
            if (stream.width,stream.height,stream.average_rate) != (704,1280,Fraction(25)):
                raise ValueError('Unexpected format')
            count = 0
            for index,frame in enumerate(video.decode(video=0)):
                if frame.pts is None or frame.pts*stream.time_base != Fraction(index,25):
                    raise ValueError('Unexpected presentation timestamp')
                count += 1
            if count != 61:
                raise ValueError('Incomplete video')
        rows.append({'path':str(path),'frames':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {'state':'full_decode_and_exact_pts_passed','mode':mode,'views':6,'frames':61,
              'fps':25,'videos':rows,'visual_approval':False,
              'visual_rejection_record':'reviews/moving61_generation.json'}
    (out/'preview/independent_pts_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'mode':mode,'decoded_frames':366,'state':result['state']}

async def main():
    async def child(mode):
        p = await asyncio.create_subprocess_exec(sys.executable,__file__,'--mode',mode)
        code = await p.wait()
        if code:
            raise RuntimeError(f'{mode} audit exited {code}')
        return {'mode':mode,'exit_code':code}
    results = await asyncio.gather(child('M0'),child('M2'))
    print(json.dumps(results),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['M0','M2'])
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(check(args.mode)),flush=True)
    else:
        asyncio.run(main())
