"""Publish one complete baseline/repair comparison after both 24-view outputs validate."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import av
from PIL import Image,ImageDraw,ImageOps
from prepare_person import ROOT,encode,save,sha

PERSON='W02'
TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1'
VIEWS=(0,6,12,18)


def publish():
    paths=[ROOT/'frontend'/PERSON,TRIAL/'frontend'/PERSON]
    for base in paths:
        validation=json.loads((base/'research/generated_validation.json').read_text())
        assert validation['state']=='all24_videos_decoded_same_retained_cameras'
        assert len(validation['videos'])==24
        for row in validation['videos']:
            assert sha(base/'outputs/turbo/videos'/f'{row["view"]:02d}.mp4')==row['sha256']
    evidence=json.loads((TRIAL/'GENERATION_PLAN.json').read_text())
    assert evidence['camera_sha256']==evidence['baseline_camera_sha256']
    assert sha(paths[0]/'preflight/motion/motion.safetensors')==evidence['motion_tensors_sha256']
    assert sha(paths[1]/'preflight/motion/motion.safetensors')==evidence['motion_tensors_sha256']
    def images():
        from contextlib import ExitStack
        with ExitStack() as stack:
            containers=[stack.enter_context(av.open(str(base/'outputs/turbo/videos'/f'{v:02d}.mp4')))
                        for v in VIEWS for base in paths]
            for con in containers:con.streams.video[0].thread_count=1
            iterators=[con.decode(video=0) for con in containers]
            for index,batch in enumerate(zip(*iterators,strict=True)):
                canvas=Image.new('RGB',(1408,1336),(128,128,128))
                for slot,frame in enumerate(batch):
                    assert abs(float(frame.pts*frame.time_base)-index/25)<1e-8
                    view=VIEWS[slot//2];method='OLD INPUT' if slot%2==0 else 'PERSON + BAG INPUT'
                    col=(slot//2%2)*2+slot%2;row=slot//4
                    im=ImageOps.contain(frame.to_image(),(352,640));canvas.paste(im,(col*352,row*668+28))
                    ImageDraw.Draw(canvas).text((col*352+5,row*668+7),f'{method} v{view:02d} t={index/25:.2f}s',fill='white')
                if index==60:canvas.save(TRIAL/'generation_comparison.jpg',quality=94)
                yield canvas
    target=TRIAL/'generation_comparison.mp4';encode(target,images())
    save(TRIAL/'COMPARISON.json',dict(state='both_generations_complete_comparison_unreviewed',
        video_sha256=sha(target),frames=121,fps=25,views=list(VIEWS),seed=42,
        scope='Generated videos only, not Gaussian assets. Same motion/cameras, corrected source foreground only.',
        quality_approved=False,baseline_preserved=True))


async def main():
    global PERSON,TRIAL
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    parser.add_argument('--person',choices=['W02','W04'],default='W02');args=parser.parse_args()
    PERSON=args.person;TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1'
    import fcntl
    lock=(TRIAL/'comparison.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.watch:
        while True:
            path=ROOT/'logs'/f'{PERSON}_bag_generate_process.json';state=json.loads(path.read_text())
            if state['state']=='completed':
                assert state['exit_code']==0
                break
            if state['state']=='failed':
                save(TRIAL/'COMPARISON_STATUS.json',dict(state='generation_failed_no_comparison_published',generation=state))
                raise RuntimeError('Corrected generation failed; retain evidence and original reconstruction')
            # A vanished process without a terminal record is ambiguous, not an
            # instruction to restart generation or publish incomplete media.
            pid=state.get('pid',state['controller_pid'])
            try:os.kill(pid,0)
            except ProcessLookupError:
                save(TRIAL/'COMPARISON_STATUS.json',dict(state='process_missing_without_terminal_record',generation=state))
                raise
            await asyncio.sleep(30)
    publish()
    from publish_previews import gallery
    gallery()
    print('Complete same-seed baseline/repair comparison published; visual review pending',flush=True)


if __name__=='__main__':asyncio.run(main())
