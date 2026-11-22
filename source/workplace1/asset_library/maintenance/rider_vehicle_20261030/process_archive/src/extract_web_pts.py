import av,json,sys,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];out=W/'sources/web/niu_mqi_native';out.mkdir(exist_ok=True)
url='https://video.golem.de/download/25843?q=high';rows=[]
with av.open(url,timeout=(30,60)) as c:
 st=c.streams.video[0];print('STREAM',st.time_base,st.average_rate,st.width,st.height,flush=True)
 # Seek backwards to decoder keyframe; all retained timestamps are native PTS.
 for shot,lo,hi in [('black_side',8.85,9.75),('blue_front',20.65,21.4)]:
  c.seek(int(lo/st.time_base),stream=st,backward=True)
  number=0
  for f in c.decode(st):
   t=float(f.pts*f.time_base)
   if t<lo:continue
   if t>hi:break
   path=out/f'{shot}_{number:03}.png';f.to_image().save(path)
   rows.append(dict(source_id='golem_25843',shot_id=shot,pts=f.pts,time_base=str(f.time_base),source_time_s=t,frame_index_basis='estimated ordinal from PTS and advertised average rate; exact PTS authoritative',native_frame_index=round(t*float(st.average_rate)),processed_frame_index=number,image_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duplicate=False));number+=1
  print('SHOT',shot,number,flush=True)
(out/('frames_manifest_'+Path(__file__).stem+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('DONE',len(rows),flush=True)
