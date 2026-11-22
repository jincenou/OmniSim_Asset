"""Synchronize two validated videos for a directly inspectable comparison."""
import argparse
from pathlib import Path
import av
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--left-label',required=True);p.add_argument('--right-label',required=True);a=p.parse_args()
    with av.open(str(a.left)) as left,av.open(str(a.right)) as right,av.open(str(a.output),'w') as out:
        l=left.streams.video[0];r=right.streams.video[0];l.thread_count=1;r.thread_count=1
        assert (l.width,l.height,l.average_rate)==(r.width,r.height,r.average_rate)
        stream=out.add_stream('libx264',rate=l.average_rate);stream.width=l.width*2;stream.height=l.height+32;stream.pix_fmt='yuv420p';stream.thread_count=2
        stream.options={'crf':'18','preset':'fast'};count=0
        for idx,(lf,rf) in enumerate(zip(left.decode(video=0),right.decode(video=0),strict=True)):
            assert lf.pts*lf.time_base==rf.pts*rf.time_base
            sheet=Image.new('RGB',(l.width*2,l.height+32),(25,25,25));sheet.paste(lf.to_image(),(0,32));sheet.paste(rf.to_image(),(l.width,32))
            draw=ImageDraw.Draw(sheet);draw.text((8,8),a.left_label,fill='white');draw.text((l.width+8,8),a.right_label,fill='white')
            if idx==60:sheet.save(a.output.with_suffix('.jpg'),quality=94)
            for packet in stream.encode(av.VideoFrame.from_image(sheet)):out.mux(packet)
            count+=1
        for packet in stream.encode():out.mux(packet)
        assert count==121
    with av.open(str(a.output)) as result:
        result.streams.video[0].thread_count=1
        assert sum(1 for _ in result.decode(video=0))==121
    print(f'Validated comparison: {a.output}',flush=True)


if __name__=='__main__':main()
