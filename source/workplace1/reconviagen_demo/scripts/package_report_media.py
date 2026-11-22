from pathlib import Path
import json,shutil,io,hashlib
from collections import defaultdict
from PIL import Image,ImageDraw
from native_candidates import frames
R=Path(__file__).resolve().parents[1];O=R/'deliveries/vehicle_experiments_20261023'
for folder in ['videos','images/sources','images/comparisons','images/inputs','images/overview','metrics','reproducibility']:(O/folder).mkdir(parents=True,exist_ok=True)
cases=json.loads((R/'inputs/random10_cases.json').read_text());byfile=defaultdict(list);records=[]
for c in cases:
 cid=c['case_id'];tag=cid.replace('random10_','');src=next(x for x in c['images'] if x['id']==c['flux_reference_id'])
 shutil.copy2(src['path'],O/'images/sources'/f'{tag}_source_crop.png')
 shutil.copy2(R/'inputs'/cid/'flux_reference.png',O/'images/sources'/f'{tag}_source_foreground.png')
 shutil.copy2(R/'inputs'/cid/'real_inputs.jpg',O/'images/inputs'/f'{tag}_real_views.jpg')
 shutil.copy2(R/'inputs'/cid/'segmented_inputs.jpg',O/'images/inputs'/f'{tag}_segmented_flux.jpg')
 shutil.copy2(R/'results'/cid/'comparison_matched_source.jpg',O/'images/comparisons'/f'{tag}_matched.jpg')
 shutil.copy2(R/'results'/cid/'comparison_six_routes.mp4',O/'videos'/f'{tag}_six_routes.mp4')
 byfile[src['raw_tfrecord']].append((tag,src))
 records.append(dict(case_id=cid,source_id=src['id'],context=src['context'],frame_index=src['frame_index'],camera=src['camera'],source_crop_sha256=hashlib.sha256(Path(src['path']).read_bytes()).hexdigest(),crop_box=src['crop_box'],bbox_xyxy=src['box_xyxy']))
for raw,targets in byfile.items():
 needed=defaultdict(list)
 for tag,src in targets:needed[src['frame_index']].append((tag,src))
 for fi,frame in frames(Path(raw)):
  if fi not in needed:continue
  for tag,src in needed.pop(fi):
   data=next(x.image for x in frame.images if x.name==src['camera'])
   dest=O/'images/sources'/f'{tag}_full_frame.jpg';dest.write_bytes(data)
   im=Image.open(io.BytesIO(data)).convert('RGB');draw=ImageDraw.Draw(im);draw.rectangle(src['box_xyxy'],outline='#ff3838',width=5)
   im.thumbnail((1280,1280));im.save(O/'images/sources'/f'{tag}_location.jpg',quality=88)
   original=Image.open(src['path']).convert('RGB');assert im.width>0
   # Stored PNG crop should match original JPEG decoded pixels exactly.
   crop=Image.open(io.BytesIO(data)).convert('RGB').crop(src['crop_box']);assert crop.tobytes()==original.tobytes()
  if not needed:break
 assert not needed
for name in ['ablation_count_seed','refinement_R_F_comparison','trellis_real_assist_seed_comparison']:
 shutil.copy2(R/'results/random10_02_white_pickup'/f'{name}.mp4',O/'videos'/f'pickup_{name}.mp4')
 p=R/'results/random10_02_white_pickup'/f'{name}.jpg'
 if p.exists():shutil.copy2(p,O/'images/comparisons'/f'pickup_{name}.jpg')
for name in ['random10_flux_overview.jpg','random10_matched_overview_1.jpg','random10_matched_overview_2.jpg','random10_metrics.png']:
 shutil.copy2(R/'docs'/name,O/'images/overview'/name)
(O/'reproducibility/source_provenance.json').write_text(json.dumps(records,indent=2))
print('MEDIA COMPLETE',len(cases),sum(p.stat().st_size for p in O.rglob('*') if p.is_file())/1024**2,'MiB')
