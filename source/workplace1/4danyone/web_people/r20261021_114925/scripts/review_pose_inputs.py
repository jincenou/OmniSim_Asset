"""Fixed temporal contact sheet for source pose and actual segmented input."""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1]
FRAMES=list(range(0,121,10))
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('ids',nargs='+');args=p.parse_args()
 for ident in args.ids:
  d=ROOT/'inputs'/ident;pr=np.load(d/'research/projection.npz');ci=json.loads((d/'research/camera_interface.json').read_text())
  caps=[cv2.VideoCapture(str(d/name)) for name in ['pose_preview.mp4','source_crop.mp4']]
  sheet=Image.new('RGB',(280*7,340*2),'#444444');counts=[0,0]
  for t in range(121):
   rows=[cap.read() for cap in caps];assert all(ok for ok,_ in rows)
   counts=[x+1 for x in counts]
   if t not in FRAMES:continue
   pose,inputim=[Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)) for _,a in rows]
   x1,y1,x2,y2=pr['source_boxes'][t];w=x2-x1;h=y2-y1
   box=np.array([x1-.35*w,y1-.12*h,x2+.35*w,y2+.12*h]);box[[0,2]]*=pose.width/ci['original_size'][0];box[[1,3]]*=pose.height/ci['original_size'][1]
   pose=pose.crop(tuple(box));panel=Image.new('RGB',(280,340),'#444444')
   panel.paste(ImageOps.contain(pose,(140,310)),(0,25));panel.paste(ImageOps.contain(inputim,(140,310)),(140,25))
   ImageDraw.Draw(panel).text((5,4),f'{ident} f{t}: pose | actual input',fill='white')
   i=FRAMES.index(t);sheet.paste(panel,((i%7)*280,(i//7)*340))
  assert all(not cap.read()[0] for cap in caps)
  for cap in caps:cap.release()
  out=d/'research/pose_temporal_review.jpg';sheet.save(out,quality=94)
  record=dict(frames=FRAMES,decoded_frames=counts,source_sha256=digest(d/'pose_preview.mp4'),input_sha256=digest(d/'source_crop.mp4'),sheet_sha256=digest(out),scope='Fixed 13 temporal samples, not full temporal visual approval')
  (d/'research/pose_review_media.json').write_text(json.dumps(record,indent=2)+'\n')
  print(ident,counts)
if __name__=='__main__':main()
