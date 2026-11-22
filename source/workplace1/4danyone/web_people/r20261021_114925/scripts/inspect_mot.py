"""Validate prescribed original image sequence and make a viewing-only preview."""
import argparse
import configparser
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import av
from source_io import read_source, source_identity, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('sequence');args=parser.parse_args()
    assert args.sequence in ('MOT16-09','MOT16-02','MOT16-10')
    paths=list((ROOT/'raw/mot16').glob(f'**/{args.sequence}/seqinfo.ini'));assert len(paths)==1
    path=paths[0].parent
    cfg=configparser.ConfigParser();cfg.read(paths[0]);spec=cfg['Sequence']
    fps=int(spec['frameRate']);length=int(spec['seqLength']);width=int(spec['imWidth']);height=int(spec['imHeight'])
    verified={}
    for receipt in (ROOT/'sources').glob('mot_range_*_receipt.json'):
        verified.update({r['path']:r for r in json.loads(receipt.read_text())['files'] if r['sequence']==args.sequence})
    assert len(verified)==length+1,(args.sequence,len(verified),length)
    for name,record in verified.items():
        assert sha(ROOT/'raw/mot16'/name)==record['sha256']
    preview=ROOT/'sources'/f'{args.sequence}_preview.mp4';assert not preview.exists()
    times=[];pick=np.rint(np.linspace(0,length-1,16)).astype(int).tolist()
    canvas=Image.new('RGB',(4*320,4*204),'#222222');draw=ImageDraw.Draw(canvas)
    with av.open(str(preview),'w') as output:
        stream=output.add_stream('libx264',rate=fps);stream.width=960;stream.height=540;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast','threads':'2'}
        for index,timestamp,rgb in read_source(path):
            times.append(timestamp);im=Image.fromarray(rgb)
            if index in pick:
                pos=pick.index(index);canvas.paste(im.resize((320,180)),((pos%4)*320,(pos//4)*204+24))
                draw.text(((pos%4)*320+3,(pos//4)*204+3),f'{args.sequence} f{index+1} t={timestamp:.2f}',fill='white')
            frame=av.VideoFrame.from_image(im.resize((960,540)));frame.pts=index
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    assert len(times)==length
    with av.open(str(preview)) as check:assert sum(1 for _ in check.decode(video=0))==length
    canvas.save(ROOT/'sources'/f'{args.sequence}_contact.jpg',quality=92)
    probe=subprocess.check_output(['/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(preview)],text=True)
    (ROOT/'sources'/f'{args.sequence}_preview_ffprobe.json').write_text(probe)
    record=dict(source_id=args.sequence,path=str(path),identity=source_identity(path),
        decoded_frames=length,timestamps_s=times,native_fps=fps,width=width,height=height,
        timing_source='official seqinfo.ini frameRate and original frame numbers; no capture exposure timestamps',
        preview=str(preview),preview_scope='960x540 viewing only; original JPEGs feed the pipeline',
        full_decode=True,visual_review=False,annotation_labels_used=False,license='CC BY-NC-SA 3.0')
    (ROOT/'sources'/f'{args.sequence}_decode.json').write_text(json.dumps(record,indent=2))
    print(args.sequence,'all source and preview frames decoded',length,flush=True)


if __name__=='__main__':main()
