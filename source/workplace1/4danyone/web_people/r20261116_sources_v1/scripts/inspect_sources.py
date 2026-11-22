"""Decode all source frames; retain exact PTS and bounded contact sheets."""
from pathlib import Path
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps
ROOT = Path(__file__).resolve().parents[1]

def inspect(path):
    dest = ROOT/'reviews'/path.stem
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]; thumbs=[]; last_sample=-1
    with av.open(str(path)) as con:
        stream=con.streams.video[0]
        stream.thread_type='AUTO'
        meta=dict(width=stream.width,height=stream.height,average_rate=str(stream.average_rate),time_base=str(stream.time_base))
        for i,frame in enumerate(con.decode(stream)):
            t=float(frame.pts*frame.time_base)
            rows.append(dict(index=i,pts=frame.pts,time_base=str(frame.time_base),time_s=t))
            if t>=last_sample+0.49:
                im=ImageOps.contain(frame.to_image(),(240,260))
                tile=Image.new('RGB',(248,290),'#eeeeee');tile.paste(im,((248-im.width)//2,4))
                ImageDraw.Draw(tile).text((5,271),f'f={i} t={t:.3f}s',fill='black')
                thumbs.append(tile);last_sample=t
    times=np.array([r['time_s'] for r in rows])
    meta.update(decoded_frames=len(rows),first_pts_s=float(times[0]),last_pts_s=float(times[-1]),
                strictly_increasing_pts=bool(np.all(np.diff(times)>0)),max_pts_gap_s=float(np.diff(times).max()))
    (dest/'probe.json').write_text(json.dumps(meta,indent=2)+'\n')
    (dest/'source_pts.json').write_text(json.dumps(rows)+'\n')
    for start in range(0,len(thumbs),40):
        batch=thumbs[start:start+40];canvas=Image.new('RGB',(248*8,290*((len(batch)+7)//8)),'white')
        for j,tile in enumerate(batch):canvas.paste(tile,((j%8)*248,(j//8)*290))
        canvas.save(dest/f'contact_{start//40:02d}.jpg',quality=88)
    print(path.stem,json.dumps(meta),flush=True)

if __name__=='__main__':
    import sys
    paths=[Path(x) for x in sys.argv[1:]] or sorted((ROOT/'raw').glob('*'))
    for path in paths:
        if path.suffix in ('.mp4','.webm') and not (ROOT/'reviews'/path.stem/'probe.json').exists():inspect(path)
