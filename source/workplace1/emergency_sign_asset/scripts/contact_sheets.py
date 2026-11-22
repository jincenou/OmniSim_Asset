import json, pathlib, subprocess, sys
import cv2
from PIL import Image, ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[1]
FF='/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/'
for sid in sys.argv[1:]:
 p=ROOT/'sources'/sid; media=p/'source.mp4';out=p/'frames';out.mkdir(exist_ok=True)
 raw=subprocess.check_output([FF+'ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,key_frame,pict_type','-of','json',str(media)])
 frames=json.loads(raw)['frames'];(p/'frame_pts.json').write_bytes(raw)
 probe=json.loads(subprocess.check_output([FF+'ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(media)]))['streams'][0]
 width,height=probe['width'],probe['height']
 import numpy as np
 decoder=subprocess.Popen([FF+'ffmpeg','-v','error','-threads','4','-i',str(media),'-f','rawvideo','-pix_fmt','bgr24','-vsync','0','pipe:1'],stdout=subprocess.PIPE)
 sampled=[];next_t=0.
 for i,f in enumerate(frames):
  raw_frame=decoder.stdout.read(width*height*3)
  if len(raw_frame)!=width*height*3: raise RuntimeError(f'{sid} decode failed {i}')
  bgr=np.frombuffer(raw_frame,dtype=np.uint8).reshape(height,width,3)
  t=float(f['best_effort_timestamp_time'])
  if t+1e-6<next_t: continue
  next_t=t+1.;name=f'f{i:06d}_t{t:.3f}.png';cv2.imwrite(str(out/name),bgr)
  sampled.append(dict(frame_index=i,pts_seconds=t,path=str(out/name)))
 if decoder.wait():raise RuntimeError('ffmpeg decoder failed')
 (p/'sampled_frames.json').write_text(json.dumps(sampled,indent=2))
 for start in range(0,len(sampled),36):
  subset=sampled[start:start+36];sheet=Image.new('RGB',(1280,((len(subset)+3)//4)*205),(30,30,30));d=ImageDraw.Draw(sheet)
  for j,r in enumerate(subset):
   im=Image.open(r['path']);im.thumbnail((316,178));x=(j%4)*320;y=(j//4)*205
   sheet.paste(im,(x,y));d.text((x+3,y+180),f"{sid} frame {r['frame_index']} | {r['pts_seconds']:.3f}s",fill='white')
  sheet.save(p/f'contact_{start//36:02d}.jpg')
 print(sid,len(frames),'decoded;',len(sampled),'sampled',flush=True)
