import cv2,json,sys
from pathlib import Path
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parents[1];sid=sys.argv[1];start=float(sys.argv[2]);end=float(sys.argv[3]);step=float(sys.argv[4]);p=W/'sources'/sid
frames=json.loads((p/'frame_pts.json').read_text())['frames'];cap=cv2.VideoCapture(str(p/'source.mp4'));dest=p/f'detail_{start:g}_{end:g}';dest.mkdir(exist_ok=True);rows=[];t=start
while t<=end+1e-6:
 i=min(range(len(frames)),key=lambda i:abs(float(frames[i]['best_effort_timestamp_time'])-t));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read()
 if not ok:raise RuntimeError('decode')
 pts=float(frames[i]['best_effort_timestamp_time']);name=f'f{i:06d}_t{pts:.3f}.png';cv2.imwrite(str(dest/name),bgr);rows.append(dict(frame_index=i,pts_seconds=pts,path=str(dest/name)));t+=step
sheet=Image.new('RGB',(1280,205*((len(rows)+3)//4)),(30,30,30));d=ImageDraw.Draw(sheet)
for j,r in enumerate(rows):
 im=Image.open(r['path']);im.thumbnail((316,178));x=j%4*320;y=j//4*205;sheet.paste(im,(x,y));d.text((x+3,y+180),f"{sid} f{r['frame_index']} {r['pts_seconds']:.3f}s",fill='white')
sheet.save(dest/'contact.jpg');(dest/'frames.json').write_text(json.dumps(rows,indent=2))
