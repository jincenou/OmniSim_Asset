import argparse,hashlib,io,json,sys,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from runtime import NEW,ROOT,save
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_diagnostics import BASE,PARTS,COLORS,project,conservative

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);args=ap.parse_args();begin=time.monotonic();cv2.setNumThreads(1)
    label=args.label;plan=json.loads((NEW/'PLAN.json').read_text());assert label in plan['query_labels']
    source=plan['anchors'][label]['input'];manifest=json.loads(Path(source['manifest']).read_text());sample=BASE/'samples'/label
    out=NEW/'data'/label;out.mkdir(exist_ok=False);n=manifest['frame_count'];bins=np.array_split(np.arange(n),8);binof={int(f):i for i,b in enumerate(bins) for f in b};middle=[int(b[(len(b)-1)//2]) for b in bins]
    points_path=sample/'generation/preflight/conditioning/keypoints_3d.npy';points=np.load(points_path);assert len(points)==n
    cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras'];hashes={(r['view'],r['frame']):r['sha256'] for r in manifest['input_hashes']}
    candidates=[];scoring=[];reviews=Image.new('RGB',(1056,2010),'#282828');draw=ImageDraw.Draw(reviews)
    for view in manifest['train_views']:
        assert view not in [0,8,16]
        cam=next(c for c in cams if c['camera_id']==view);K=np.asarray(cam['K'],float);K[:2]*=.5;score_cores=[]
        for frame in range(n):
            image_path=sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png';content=image_path.read_bytes();digest=hashlib.sha256(content).hexdigest();assert digest==hashes[view,frame]
            rgba=np.asarray(Image.open(io.BytesIO(content)),np.float32)/255;uv,_=project(points[frame],cam['camera_to_world'],K);core,support=conservative(uv,rgba[...,3]>.5,label,view)
            area=[int((core==k).sum()) for k in range(1,7)];record=dict(index=len(candidates),uid=view*n+frame,view=view,frame=frame,bin=binof[frame],areas=area,foreground_pixels=int((rgba[...,3]>.5).sum()),input_sha256=digest,core_status=['valid_proxy' if a>=64 else 'unknown_or_insufficient_support' for a in area])
            candidates.append(record)
            if frame in middle:score_cores.append(core);scoring.append(dict(candidate_index=record['index'],view=view,frame=frame,bin=binof[frame]))
            if view in [1,9,17] and frame in [0,(n-1)//2,n-1]:
                row=[1,9,17].index(view);col=[0,(n-1)//2,n-1].index(frame);rgb=(rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]))*255
                for k,color in enumerate(COLORS,1):rgb[core==k]=.55*rgb[core==k]+.45*color
                reviews.paste(Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)),(col*352,row*670+30));draw.text((col*352+5,row*670+8),f'{label} TRAIN v{view:02d} f{frame} PROXY CORE',fill='white')
        assert len(score_cores)==8;np.savez_compressed(out/f'cores{view:02d}.npz',labels=np.stack(score_cores))
    reviews.save(out/'review.jpg',quality=94)
    result=dict(state='train_only_candidates_ready_visual_review_pending',label=label,parts=PARTS,frame_count=n,train_views=manifest['train_views'],representative_frames=middle,time_bins=[b.tolist() for b in bins],observations=candidates,scoring=scoring,points_sha256=hashlib.sha256(points_path.read_bytes()).hexdigest(),core_function_sha256=hashlib.sha256((ROOT/'scripts/prepare_diagnostics.py').read_bytes()).hexdigest(),seconds=time.monotonic()-begin,limits='Conservative projected reference cores,not semanticGT or materialcorrespondence; unknowns retained')
    save(out/'candidates.json',result);print(label,'candidate complete',len(candidates),len(scoring),result['seconds'],flush=True)
if __name__=='__main__':main()
