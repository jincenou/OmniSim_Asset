"""Compose six real Gaussian orbit renders and a fixed-source-view comparison."""
from pathlib import Path
import argparse,json
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageOps
from evaluate_asset_metrics import foreground
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['B0','R','F','H','B0_real1','B0_real4']
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args();out=ROOT/'results'/a.case
 assets=[out/f'{v}_seed42'/'asset' for v in VARIANTS]
 for asset in assets:
  if not (asset/'heading_review.json').exists():raise ValueError('Heading review pending: '+str(asset))
 configs=[json.loads((p/'render_config.json').read_text()) for p in assets]
 shared=['size','focal_px','distance_m','elevation_deg','center_m','background','azimuths_deg','length_m','video_frames','video_fps']
 for cfg in configs[1:]:
  if any(cfg[k]!=configs[0][k] for k in shared):raise ValueError('Comparison cameras are not shared')
 readers=[imageio.get_reader(str(p/'turntable.mp4')) for p in assets];n=configs[0]['video_frames'];size=configs[0]['size']
 video=out/'comparison_six_routes.mp4'
 with imageio.get_writer(str(video),fps=configs[0]['video_fps'],codec='libx264',quality=8,macro_block_size=1) as writer:
  for index in range(n):
   panel=Image.new('RGB',(3*size,2*(size+32)),'#ddd');draw=ImageDraw.Draw(panel)
   for j,(reader,v) in enumerate(zip(readers,VARIANTS)):
    frame=reader.get_data(index);x=j%3*size;y=j//3*(size+32);panel.paste(Image.fromarray(frame),(x,y+32));draw.text((x+8,y+8),v,fill='black')
   writer.append_data(np.array(panel))
 for reader in readers:reader.close()
 cam=json.loads((ROOT/'inputs'/a.case/'evaluation/camera.json').read_text());source,_,_=foreground(cam['source_rgb'],cam['source_mask'])
 images=[source]
 for v in VARIANTS:
  p=out/f'{v}_seed42'/'evaluation';im,_,_=foreground(p/'matched_source.png',p/'matched_source_alpha.png');images.append(im)
 poster=Image.new('RGB',(7*280,310),'#ddd');draw=ImageDraw.Draw(poster)
 for j,(im,label) in enumerate(zip(images,['SOURCE']+VARIANTS)):
  poster.paste(ImageOps.contain(im,(280,280)),(j*280,30));draw.text((j*280+5,7),label,fill='black')
 poster.save(out/'comparison_matched_source.jpg',quality=95)
 (out/'comparison_six_routes.json').write_text(json.dumps(dict(case_id=a.case,variants=VARIANTS,source_view_id=cam['source_id'],video=str(video),decoded_frames_per_variant=n,heading_reviews=[str(p/'heading_review.json') for p in assets],shared_camera_keys=shared,scope='Actual standalone Gaussian renders. New batch scene insertion not evaluated.',matched_source='Metadata-derived fixed view; separate foreground normalization; no metric-based view choice'),indent=2))
 print('Comparison complete',a.case,flush=True)
if __name__=='__main__':main()
