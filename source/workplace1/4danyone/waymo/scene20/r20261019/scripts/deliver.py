"""Publish only complete24-view results, then remove this task's disposable pixels."""
import argparse,json,hashlib,shutil
from pathlib import Path
import av,numpy as np
from PIL import Image,ImageDraw
from finish_generation import finish

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(path,images,fps):
 with av.open(str(path),'w') as c:
  s=c.add_stream('libx264',rate=fps);s.width=images[0].width;s.height=images[0].height;s.pix_fmt='yuv420p';s.options={'crf':'19','preset':'fast','threads':'2'}
  for im in images:
   for p in s.encode(av.VideoFrame.from_image(im)):c.mux(p)
  for p in s.encode():c.mux(p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--sample',type=Path,required=True);a.add_argument('--frames',type=int,required=True);args=a.parse_args();b=args.sample;w=b/'.work';src=w/'source';gen=w/'generation';out=gen/'outputs/turbo';finish(w,args.frames)
 (b/'input').mkdir(exist_ok=True);(b/'motion').mkdir(exist_ok=True)
 for source,target in [(src/'inputs/main.mp4',b/'input/model_input.mp4'),(src/'inputs/frame_mapping.json',b/'input/frame_mapping.json'),(src/'inputs/cameras.npz',b/'input/source_cameras.npz'),(src/'segmentation/target_masks.npz',b/'input/target_masks.npz'),(src/'native.json',b/'input/native.json'),(src/'native_cameras.npz',b/'input/native_cameras.npz'),(src/'native_contact.jpg',b/'input/native_contact.jpg'),(out/'cameras.json',b/'cameras.json'),(out/'root_motion.npz',b/'root_motion.npz'),(out/'metadata.json',b/'generation.json'),(gen/'research/generated_validation.json',b/'validation.json')]:shutil.copy2(source,target)
 cameras=json.loads((b/'cameras.json').read_text());cameras['motion_world_to_local_world_file']='root_motion.npz';cameras['root_motion']['provenance']='manifest.json';save(b/'cameras.json',cameras)
 for p in (src/'motion_pair/calibrated_moving').iterdir():
  if p.is_file():shutil.copy2(p,b/'motion'/p.name)
 shutil.copy2(src/'motion_pair/projection_comparison.mp4',b/'motion/source_fit.mp4')
 #24 originals are moved, not copied. Retain official skeleton guides for downstream reconstruction.
 shutil.move(str(out/'videos'),str(b/'videos'));shutil.move(str(out/'skeletons'),str(b/'skeletons'))
 native=json.loads((src/'native.json').read_text());images=[Image.open(x['path']).convert('RGB') for x in native['frames']];encode(b/'input/observed_native.mp4',images,10)
 views=[0,4,8,12,16,20];decoded=[]
 for v in views:
  with av.open(str(b/'videos'/f'{v:02d}.mp4')) as c:
   c.streams.video[0].thread_count=1;decoded.append([f.to_image().resize((264,480)) for f in c.decode(video=0)])
 panels=[]
 for t in range(args.frames):
  im=Image.new('RGB',(792,1000),'#222222');d=ImageDraw.Draw(im)
  for j,v in enumerate(views):
   x=(j%3)*264;y=(j//3)*500;im.paste(decoded[j][t],(x,y+20));d.text((x+4,y+3),f'view{v:02d} frame{t}',fill='white')
  panels.append(im)
 encode(b/'preview.mp4',panels,25)
 contact=Image.new('RGB',(792*3,1000))
 for i,t in enumerate([0,args.frames//2,args.frames-1]):contact.paste(panels[t],(i*792,0))
 contact.save(b/'contact.jpg',quality=93)
 with av.open(str(b/'preview.mp4')) as c:assert sum(1 for _ in c.decode(video=0))==args.frames
 manifest={'state':'complete24_multiview_videos_decoded','candidate':json.loads((b/'candidate.json').read_text()),'frames':args.frames,'views':24,'training':False,'model':'4DAnyone Turbo','native_video_playback':'10fps nominal preview; true exposure timestamps retained in native.json/cameras,not uniform measured timestamps','archived_layout':{'former_source_inputs':'input/model_input.mp4 and input/frame_mapping.json','former_generation_videos':'videos/','former_native_cameras':'input/native_cameras.npz','temporary_paths_in_provenance':'Historical .work paths in original metadata are provenance; raw_source_path and candidate.raw_tfrecord remain original sources,decoded scratch PNGs removed'},'source_pose':'GVHMR estimate, camera calibrated rotational input; not world-registered metric motion','root_local':json.loads((gen/'research/root_local.json').read_text()),'visual_review':'pending; numeric completion does not imply appearance correctness','retained':{str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file() and '.work' not in p.parts and p.name!='manifest.json'},'removed_scratch':['decoded crop PNG','SAM JPEG frames','RIFE duplicate frames/video','GVHMR feature caches','duplicated generation input/motion links'],'source_data_modified':False}
 save(b/'manifest.json',manifest)
 # All published files exist and video checks passed before removing the private temporary tree.
 shutil.rmtree(w)
if __name__=='__main__':main()
