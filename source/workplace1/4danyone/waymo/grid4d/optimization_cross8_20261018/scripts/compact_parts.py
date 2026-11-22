"""Regenerate scientific comparison sheets without blank unknown-ROI rows; no model pixels edited."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from publish_cross8 import NEW,ROOT,media,descriptors
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);label=ap.parse_args().label
    out=NEW/'comparison'/label;v=json.loads((out/'validation.json').read_text());selection=json.loads((ROOT/'gallery'/label/'selection.json').read_text())['parts']
    valid=[s for s in selection if 'crop_xyxy' in s];unknown=[s['part'] for s in selection if 'crop_xyxy' not in s]
    assert unknown and len(valid)>0
    titles=['Generated reference','HUST grid61','A 52999','Fused 32999'];columns=[ROOT/'tmp/baseline_raw'/label/'4dgs_grid61',descriptors(label,'A')['raw'],descriptors(label,'fused')['raw']]
    before={name:sha(out/name) for name in ['parts_rgb.jpg','parts_alpha.jpg']}
    sheets={m:Image.new('RGB',(800,72+236*len(valid)),'#282828') for m in ['rgb','alpha']}
    for view in sorted({r['view'] for r in valid}):
        arrays=[]
        for p in columns:
            with np.load(p/f'view{view:02d}.npz') as z:arrays.append((z['rgb'],z['alpha']))
        for row,chosen in enumerate(valid):
            if chosen['view']!=view:continue
            frame=chosen['representative_frame'];x0,y0,x1,y1=chosen['crop_xyxy']
            rgba=np.asarray(Image.open(media.BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
            images=[rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])]+[np.clip(rgb[frame]+.5*(1-a[frame,...,None]),0,1) for rgb,a in arrays]
            alphas=[rgba[...,3]]+[a[frame] for rgb,a in arrays]
            for mode,values in [('rgb',images),('alpha',alphas)]:
                sheet=sheets[mode];ink=ImageDraw.Draw(sheet);y=32+row*236
                for col,(title,im) in enumerate(zip(titles,values)):
                    patch=Image.fromarray(np.rint(im[y0:y1,x0:x1]*255).astype(np.uint8)).convert('RGB');patch=ImageOps.contain(patch,(200,200));sheet.paste(patch,(col*200+(200-patch.width)//2,y+32+(200-patch.height)//2));ink.text((col*200+3,y+2),title+' '+chosen['part'],fill='white');ink.text((col*200+3,y+16),f'v{view:02d} f{frame}',fill='white')
    for mode,sheet in sheets.items():
        ink=ImageDraw.Draw(sheet);ink.text((5,8),'Frozen historical crops; unmasked predictions; estimated body cores',fill='white');ink.text((5,sheet.height-27),'Unknown reference cores: '+', '.join(unknown)+' (not zero error)',fill='white');sheet.save(out/f'parts_{mode}.jpg',quality=95)
    v['part_layout']=dict(state='compact_known_reference_cores',unknown=unknown,metric_denominators_unchanged=True,previous_layout_sha256=before,current_sha256={name:sha(out/name) for name in before},source_predictions_unchanged=True);(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
    print(label,'scientific sheets regenerated with explicitunknown annotation',flush=True)
if __name__=='__main__':main()
