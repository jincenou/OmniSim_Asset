from pathlib import Path
import json,hashlib
import av
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];ROOT=R/'experiments/shortS02_v2';OUT=ROOT/'outputs/smoke'
def main():
 files=sorted((OUT/'videos').glob('*.mp4'));assert len(files)==6
 output=OUT/'short_preview';output.mkdir(exist_ok=False);sheet=Image.new('RGB',(6*176,3*344),'#222222');rows=[]
 for view,path in enumerate(files):
  picked={};count=0
  with av.open(str(path)) as container:
   stream=container.streams.video[0];assert float(stream.average_rate)==25
   for frame in container.decode(stream):
    im=frame.to_image();assert im.size==(704,1280)
    if count in (0,10,20):picked[count]=im
    count+=1
  assert count==21
  for ri,frame in enumerate((0,10,20)):
   sheet.paste(picked[frame].resize((176,320)),(view*176,ri*344+24))
   ImageDraw.Draw(sheet).text((view*176+2,ri*344+3),f'view{view} t={frame/25:.2f}s',fill='white')
  rows.append({'video':str(path),'frames':count,'fps':25,'size':[704,1280],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 sheet.save(output/'contact.jpg',quality=95)
 report={'state':'six_short_videos_decoded_visual_review_pending','views':6,'frames':21,'sample_span_seconds':.8,'encoded_duration_seconds':.84,'source_observations':9,'temporal_padding_to121':False,'source_metadata':str(ROOT/'inputs/frame_mapping.json'),'videos':rows,'scope':'Experimental short-runtime generation; no short dynamic GS asset has been trained; source projection, appearance and motion still require review.'}
 (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state']}))
if __name__=='__main__':main()
