"""Revisit all unfulfilled requested instances; preserve the original wave."""
import collections, itertools, json, sys, time
from pathlib import Path
import ijson
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent/'vehicles2_json_20261106'
sys.path.insert(0,str(ORIGINAL))
from prepare import DATA, INDEX, read, write, rot

def main():
    published={a['instance_token'] for f in (ORIGINAL/'chunks').glob('*/completed.json') for a in read(f)['published']}
    rows=[a for a in read(ORIGINAL/'inventory.json')['assets'] if a['instance_token'] not in published]
    assert len(rows)==5, f'Expected 5 original unfulfilled targets, found {len(rows)}'
    wanted={a['instance_token'] for a in rows};anns=collections.defaultdict(list)
    with (DATA/'v1.0-trainval/sample_annotation.json').open('rb') as f:
        for a in ijson.items(f,'item',use_float=True):
            if a['instance_token'] in wanted:anns[a['instance_token']].append(a)
    signs=np.array(list(itertools.product([-1,1],repeat=3)));output=[]
    for r in rows:
        frames=collections.defaultdict(list)
        for f in read(INDEX/(r['scene_name']+'.json'))['frames']:frames[f['sample_token']].append(f)
        views=[]
        for a in anns[r['instance_token']]:
            size=[a['size'][1],a['size'][0],a['size'][2]]
            world=(signs*np.array(size)/2)@rot(a['rotation']).T+np.array(a['translation'])
            for f in frames.get(a['sample_token'],[]):
                tf=np.linalg.inv(f['camera_to_world']);xyz=world@tf[:3,:3].T+tf[:3,3]
                if xyz[:,2].min()<=.1:continue
                uv=xyz@np.array(f['camera_intrinsic']).T;uv=uv[:,:2]/uv[:,2:]
                low=uv.min(0);high=uv.max(0);clow=np.maximum(low,[0,0]);chigh=np.minimum(high,[1600,900]);wh=chigh-clow
                if (wh<[30,24]).any() or not Path(f['path']).is_file():continue
                fraction=float(wh.prod()/np.prod(high-low));edge=bool((low<3).any() or (high>[1597,897]).any())
                views.append(dict(frame={k:v for k,v in f.items() if k!='motion_paths'},annotation_token=a['token'],bbox_xyxy=[*clow.tolist(),*chigh.tolist()],unclipped_bbox=[*low.tolist(),*high.tolist()],size_lwh_m=size,visibility_token=a['visibility_token'],clipped_fraction=fraction,edge=edge,score=float(wh.prod())*fraction*(.55 if edge else 1)))
        views.sort(key=lambda v:-v['score']);selected=[]
        for v in views:
            if any(v['frame']['camera_name']==o['frame']['camera_name'] and abs(v['frame']['timestamp_us']-o['frame']['timestamp_us'])<1000000 for o in selected):continue
            selected.append(v)
            if len(selected)>=12:break
        folder=ROOT/'sources'/r['obj_id'];folder.mkdir(parents=True,exist_ok=True)
        for j,v in enumerate(selected):
            out=folder/f'view_{j:02d}';out.mkdir(exist_ok=True)
            im=Image.open(v['frame']['path']).convert('RGB');b=v['bbox_xyxy'];box=[max(0,int(b[0])-5),max(0,int(b[1])-5),min(1600,int(b[2])+6),min(900,int(b[3])+6)]
            crop=im.crop(box);crop.save(out/'original.jpg')
            # Monotonic exposure/contrast preprocessing only; no invented pixels.
            mean=float(np.asarray(crop.convert('L')).mean());gamma=.55 if mean<65 else 1.0
            enhanced=ImageOps.autocontrast(crop,cutoff=.5) if mean<65 else crop.copy()
            if gamma!=1:enhanced=enhanced.point([round(255*(i/255)**gamma) for i in range(256)]*3)
            enhanced.save(out/'enhanced.png')
            sheet=Image.new('RGB',(1000,430),'white');draw=ImageDraw.Draw(sheet)
            for n,ii in enumerate([crop,enhanced]):
                ii=ii.copy();ii.thumbnail((490,390));sheet.paste(ii,(500*n+(490-ii.width)//2,35+(390-ii.height)//2))
            draw.text((5,5),'A: EXACT ANNOTATION CROP',fill='black');draw.text((505,5),'B: EXPOSURE ONLY (SAME PIXELS)',fill='black');sheet.save(out/'review.jpg')
            v.update(folder=str(out),original_crop=str(out/'original.jpg'),enhanced_crop=str(out/'enhanced.png'),review_image=str(out/'review.jpg'),crop_box=box,exposure_gamma=gamma)
        rec=dict(r,candidates=selected,eligible_observations=len(views),recovery_reason='old_generated_reject' if (ORIGINAL/'sources'/r['obj_id']/'ready.json').exists() else 'old_input_rejected',status='prepared')
        write(folder/'candidates.json',rec);output.append(rec)
        print(r['scene_name'],r['asset_category'],len(views),len(selected),flush=True)
    write(ROOT/'inventory.json',dict(total=len(output),assets=output,updated=time.time(),original_wave=str(ORIGINAL),policy='Expanded exact-instance observations, clipped views allowed for generative completion only'))
    for start in range(0,len(output),8):
        subset=output[start:start+8];sheet=Image.new('RGB',(1200,220*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            draw.text((4,i*220+2),r['obj_id'],fill='black')
            for j,v in enumerate(r['candidates'][:3]):
                im=Image.open(v['enhanced_crop']);im.thumbnail((390,190));sheet.paste(im,(j*400,i*220+24))
        sheet.save(ROOT/f'input_overview_{start:02d}.jpg')

if __name__=='__main__':main()
