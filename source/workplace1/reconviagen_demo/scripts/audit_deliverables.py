"""Verify fresh input integrity, original loader contract, and decoded videos."""
from pathlib import Path
import json,hashlib,subprocess,os
from asset_adapter import inspect,ROOT
from PIL import Image
import imageio.v2 as imageio

cases=['fresh_blue_suv','fresh_black_pickup','fresh_silver_suv']
records=[]
for case in cases:
 manifest=json.loads((ROOT/'inputs'/case/'segmented_manifest.json').read_text())
 real=[x for x in manifest['images'] if x['source_type']=='real' and x['split']=='train']
 held=[x for x in manifest['images'] if x['split']=='heldout']
 flux=[x for x in manifest['images'] if x['source_type']=='flux']
 assert (len(real),len(held),len(flux))==(4,2,2)
 assert not {x['sha256'] for x in real+flux}&{x['sha256'] for x in held}
 for x in manifest['images']:
  assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
  im=Image.open(x['rgba_path']);assert im.mode=='RGBA' and im.getchannel('A').getextrema()[1]>0
 for variant in ['B0','R','F','H']:
  out=ROOT/'results'/case/f'{variant}_seed42';asset=out/'asset';stats,_=inspect(asset/'canonical.ply')
  exp=json.loads((asset/'experiment.json').read_text());assert exp['heading_verified']
  cfg=json.loads((asset/'render_config.json').read_text());assert cfg['video_frames']==96
  for i in range(16):
   assert Image.open(asset/f'view_{i:02}.png').size==(512,512)
   assert Image.open(asset/f'alpha_{i:02}.png').getextrema()[1]>0
  reader=imageio.get_reader(asset/'turntable.mp4');count=sum(1 for _ in reader);reader.close();assert count==96
  records.append(dict(case=case,variant=variant,loader_validation=stats,heading_verified=True,decoded_video_frames=count,views=16,rgba_and_rgb_hash_verified=True,quality_status='review'))
result=dict(base_matrix=records,base_matrix_verified=len(records)==12,scope='Structural compatibility and artifact integrity; not photometric or production quality approval',heldout_metrics='N/A: input-only object-to-asset registration not validated')
(ROOT/'docs/deliverable_audit.json').write_text(json.dumps(result,indent=2));print('Verified 12 base variants, 1152 decoded orbit frames, 192 view images, frozen input hashes')
