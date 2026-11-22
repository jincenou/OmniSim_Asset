"""Observed-only videos and a clearly labeled source/conditioning/generated preview."""
from pathlib import Path
from argparse import ArgumentParser
from fractions import Fraction
import hashlib,json
import av,numpy as np
from PIL import Image,ImageOps,ImageDraw
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(path,frames,fps,size):
 with av.open(str(path),'w') as out:
  stream=out.add_stream('libx264',rate=Fraction(fps));stream.width,stream.height=size;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'medium'};stream.codec_context.thread_count=2;count=0
  for i,im in enumerate(frames):
   f=av.VideoFrame.from_image(im.convert('RGB'));f.pts=i;f.time_base=Fraction(1,fps)
   for packet in stream.encode(f):out.mux(packet)
   count+=1
  for packet in stream.encode():out.mux(packet)
 with av.open(str(path)) as con:
  con.streams.video[0].thread_count=1;decoded=sum(1 for _ in con.decode(video=0))
 assert decoded==count
 return {'frames':count,'fps':fps,'dimensions':size,'sha256':sha(path)}
def main():
 p=ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();base=R/'samples'/a.label;source=base/'source';native=json.loads((source/'native.json').read_text());out=source/'visualization';out.mkdir(exist_ok=False);records=native['frames'];videos={};size=tuple(native['size'])
 videos['observed_crop.mp4']=encode(out/'observed_crop.mp4',(Image.open(v['path']) for v in records),10,size)
 videos['observed_full_camera.mp4']=encode(out/'observed_full_camera.mp4',(Image.open(v['raw_source_path']) for v in records),10,(1920,1280))
 generation=base/'generation';proof=generation/'research/generated_validation.json'
 if proof.exists():
  mapping=json.loads((source/'inputs/frame_mapping.json').read_text())['frames'];paths=[source/'inputs/main.mp4',generation/'preflight/conditioning/goliath40/00.mp4',generation/'outputs/turbo/videos/00.mp4'];containers=[av.open(str(p)) for p in paths]
  for con in containers:con.streams.video[0].thread_count=1
  readers=[iter(c.decode(video=0)) for c in containers];cache={}
  def chain():
   for row in mapping:
    i=row['left_native_index']
    if i not in cache:cache[i]=Image.open(records[i]['path']).convert('RGB')
    images=[cache[i],*[next(reader).to_image() for reader in readers]];labels=['Waymo observed crop;held','Actual RIFE + masked input','Root-local skeleton;gen cam0','4DAnyone generated cam0'];sheet=Image.new('RGB',(1408,664),(25,25,25));draw=ImageDraw.Draw(sheet)
    for column,(image,label) in enumerate(zip(images,labels)):
     im=ImageOps.contain(image,(352,640));sheet.paste(im,(column*352+(352-im.width)//2,24+(640-im.height)//2));draw.text((column*352+4,5),label,fill='white')
    yield sheet
  try:videos['source_chain.mp4']=encode(out/'source_chain.mp4',chain(),25,(1408,664))
  finally:
   for c in containers:c.close()
 result={'state':'observed_and_pipeline_preview_decoded','label':a.label,'native_observations':len(records),'observed_frame_indices':[x['frame_index'] for x in records],'actual_image_center_timestamps_s':[x['image_center_timestamp_s'] for x in records],'videos':videos,'source_scope':'Observed videos contain only25/49actual observations,no interpolation.10fps is nominal visual playback; measured exposure timestamps retained.','chain_scope':'Source crop held according to measured RIFE bracket; source camera and generated camera are distinct,not pixel-aligned GT; display resizing is not super-resolution.','native_mapping_sha256':sha(source/'native.json')};(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'label':a.label,'videos':list(videos)}),flush=True)
if __name__=='__main__':main()
