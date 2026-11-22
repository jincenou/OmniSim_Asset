"""Bounded CPU source previews, with native timestamps and immutable provenance."""
import asyncio
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=Path('/mnt/public_data/waymo/val')

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def build(row):
    destination=ROOT/'native_previews'/row['review_id']
    destination.mkdir(parents=True,exist_ok=False)
    obs=row['observations'];boxes=np.asarray([o['bbox_xyxy'] for o in obs]);margin=int(.15*np.median(boxes[:,3]-boxes[:,1]))
    crop=(max(0,int(boxes[:,0].min())-margin),max(0,int(boxes[:,1].min())-margin),min(1920,int(boxes[:,2].max())+margin),min(1280,int(boxes[:,3].max())+margin))
    ticks=[o['timestamp_micros']-obs[0]['timestamp_micros'] for o in obs]
    assert all(b>a for a,b in zip(ticks,ticks[1:]))
    video=destination/'source.mp4';partial=destination/'source.partial.mp4';provenance=[];contact={}
    sampled=set(np.linspace(0,len(obs)-1,min(13,len(obs)),dtype=int).tolist())
    with av.open(str(partial),'w') as output:
        stream=output.add_stream('libx264',rate=10)
        stream.width=960;stream.height=420;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000)
        stream.codec_context.time_base=Fraction(1,1000000)
        stream.options={'crf':'20','preset':'fast','bf':'0','threads':'2'}
        for index,o in enumerate(obs):
            path=PUBLIC/row['scene']/'images'/f"{o['frame_index']:03d}_{row['camera_index']}.jpg"
            payload=path.read_bytes()
            with Image.open(io.BytesIO(payload)) as opened:original=opened.convert('RGB')
            assert original.size==(1920,1280),original.size
            annotated=original.copy();ImageDraw.Draw(annotated).rectangle(o['bbox_xyxy'],outline=(255,70,40),width=5)
            canvas=Image.new('RGB',(960,420),(24,24,24));draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f"{row['review_id']} | scene {row['scene']} cam {row['camera_index']} | native frame {o['frame_index']} | {ticks[index]/1e6:.6f}s",fill='white')
            draw.text((10,27),'Original frame + track box                         Fixed crop, native observations only',fill='white')
            full=ImageOps.contain(annotated,(552,368));canvas.paste(full,(0,48))
            patch=original.crop(crop);detail=ImageOps.contain(patch,(400,368));canvas.paste(detail,(556+(400-detail.width)//2,48+(368-detail.height)//2))
            frame=av.VideoFrame.from_image(canvas);frame.pts=int(ticks[index]);frame.time_base=Fraction(1,1000000)
            for packet in stream.encode(frame):output.mux(packet)
            provenance.append({'frame_index':o['frame_index'],'timestamp_micros':o['timestamp_micros'],'source':str(path),'sha256':hashlib.sha256(payload).hexdigest(),'bbox_xyxy':o['bbox_xyxy']})
            if index in sampled:
                tile=ImageOps.contain(patch,(192,250));panel=Image.new('RGB',(200,280),'white');panel.paste(tile,((200-tile.width)//2,22+(250-tile.height)//2));ImageDraw.Draw(panel).text((4,4),f"f{o['frame_index']} {ticks[index]/1e6:.2f}s",fill='black');contact[index]=panel
        for packet in stream.encode():output.mux(packet)
    with av.open(str(partial)) as check:
        decoded=list(check.decode(video=0));actual=[float(f.pts*f.time_base) for f in decoded]
        assert len(decoded)==len(obs)
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(ticks)/1e6)))
        assert error<=1e-6,{'pts_error':error,'actual':actual,'expected':ticks}
    partial.rename(video)
    sheet=Image.new('RGB',(1000,280*int(np.ceil(len(contact)/5))),(230,230,230))
    for slot,(_,tile) in enumerate(sorted(contact.items())):sheet.paste(tile,((slot%5)*200,(slot//5)*280))
    sheet.save(destination/'motion_contact.jpg',quality=92)
    report={'state':'source_video_decoded_and_timestamps_verified','review_id':row['review_id'],'identity_key':row['identity_key'],
      'scene':row['scene'],'camera_index':row['camera_index'],'track_id':row['track_id'],'associated_lidar_track_id':row['associated_lidar_track_id'],
      'frames':len(obs),'source_span_s':ticks[-1]/1e6,'max_decoded_pts_error_s':error,'fixed_crop_xyxy':crop,'crop_changes_over_time':False,
      'preview_resize_is_not_new_observed_detail':True,'source_images':provenance,'video':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
      'motion_contact':str(destination/'motion_contact.jpg'),'human_action':'visual_review_pending','gpu_used':False}
    save(destination/'validation.json',report)
    print(json.dumps({'review_id':row['review_id'],'frames':len(obs),'state':report['state']}),flush=True)
    return {k:v for k,v in report.items() if k!='source_images'}

async def main():
    rows=json.loads((ROOT/'sample_manifest.json').read_text())['candidates'];sem=asyncio.Semaphore(2)
    async def one(row):
        async with sem:
            try:return await asyncio.to_thread(build,row)
            except Exception as exc:
                print(json.dumps({'review_id':row['review_id'],'state':'failed','error':repr(exc)}),flush=True)
                raise
    results=await asyncio.gather(*map(one,rows),return_exceptions=True)
    good=[r for r in results if not isinstance(r,BaseException)];errors=[repr(r) for r in results if isinstance(r,BaseException)]
    save(ROOT/'native_preview_manifest.json',{'completed':len(good),'failures':errors,'previews':good,'gpu_used':False})
    if errors:raise RuntimeError('Source preview failures: '+str(errors))

if __name__=='__main__':asyncio.run(main())
