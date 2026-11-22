"""Only complete-arm terminal diagnostics, on previously fixed failure windows."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

N=Path(__file__).resolve().parents[1];ROOT=N.parent
BASE=ROOT.parent/'random8/r20261015_113332'
def read(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',choices=['N06','N11'],required=True);args=parser.parse_args();label=args.label
    plan=read(N/'PLAN.json')
    for arm in plan['arms']:
        assert read(N/'assets'/f'{label}_{arm}/replay_validation.json')['passed']
        assert (N/'results'/f'{label}_{arm}_32999/expanded.json').exists()
    out=N/'research/visual_windows'/label;out.mkdir(parents=True,exist_ok=False)
    names=['reference','A','G2','P1','P2','P3','P4']
    paths={a:N/'cache/raw'/f'{label}_{a}_32999' for a in plan['arms']}
    paths['A']=Path(plan['inputs'][label]['old_references']['A']['raw'])
    paths['G2']=ROOT/'evq_primitive_local_20261019/cache/raw'/f'{label}_G2_32999'
    bad=plan['evaluation']['fixed_bad_windows'][label];receipts=[]
    for view in [bad['view']]+([8,16] if label=='N11' else []):
        arrays={}
        for method,path in paths.items():
            with np.load(path/f'view{view:02d}.npz') as z:arrays[method]=(z['rgb'],z['alpha'])
        def panel(times,box):
            x0,y0,x1,y1=map(int,box);w,h=x1-x0,y1-y0
            canvas=Image.new('RGB',(w*len(names),(h+26)*len(times)),'#222222');draw=ImageDraw.Draw(canvas)
            for row,frame in enumerate(times):
                rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
                reference=rgba[:,:,:3]*rgba[:,:,3:]+.5*(1-rgba[:,:,3:])
                for col,method in enumerate(names):
                    rgb=reference if method=='reference' else np.clip(arrays[method][0][frame]+.5*(1-arrays[method][1][frame,...,None]),0,1)
                    image=Image.fromarray(np.rint(rgb[y0:y1,x0:x1]*255).astype(np.uint8))
                    canvas.paste(image,(col*w,row*(h+26)+26));draw.text((col*w+3,row*(h+26)+5),f'{method} f{frame}',fill='white')
            return canvas
        if view==bad['view']:
            with np.load(N/'data/expanded_roi'/label/f'view{view:02d}.npz') as z:box=z['boxes'][0]
            for first in range(bad['frames'][0],bad['frames'][1]+1,5):
                times=list(range(first,min(first+5,bad['frames'][1]+1)))
                path=out/f'window_{times[0]:03d}_{times[-1]:03d}.jpg';panel(times,box).save(path,quality=95)
                receipts.append(dict(file=str(path.relative_to(N)),view=view,frames=times,box=list(map(int,box)),basis='unchanged preregistered expanded left ROI'))
        if label=='N11':
            source=ROOT/'evq_primitive_local_20261019/comparison/N11'/f'visual_crop_v{view:02d}.json'
            box=read(source)['boxes'][0]['box'];path=out/f'feet_last_v{view:02d}.jpg';panel([120],box).save(path,quality=95)
            receipts.append(dict(file=str(path.relative_to(N)),view=view,frames=[120],box=box,basis=str(source),semantic_foot_GT=False))
    (out/'manifest.json').write_text(json.dumps(dict(label=label,columns=names,files=receipts,manual_review='pending',no_partial_arm_comparison=True),ensure_ascii=False,indent=2)+'\n')
    print(label,'fixed review pages ready',flush=True)


if __name__=='__main__':main()
