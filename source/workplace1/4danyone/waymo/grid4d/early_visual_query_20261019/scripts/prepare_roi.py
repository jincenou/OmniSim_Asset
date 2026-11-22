"""Freeze reference-only expanded proxy regions before observing new predictions."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;BASE=ROOT.parent/'random8/r20261015_113332'
sys.path.insert(0,str(ROOT/'scripts'))
from roi_candidates import project,segment
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cv2.setNumThreads(2);root=NEW/'data/expanded_roi';root.mkdir(exist_ok=False)
    rules=dict(provenance='Pretraining engineering choices authorized by user; proxy regions, not semantic truth',arm_bones=[[5,7],[7,62]],other_arm_bones=[[6,8],[8,41]],leg_bones=[[11,13],[13,15]],other_leg_bones=[[12,14],[14,18]],arm_radius_torso=.18,leg_radius_torso=.22,distal_extension_segment_fraction=.25,min_reference_alpha_mass=64,min_roi_pixels=256,max_other_core_fraction=.25,local_LPIPS_min_reference_pixels=128,local_LPIPS_min_box_side=64,local_LPIPS='AlexNet; fixed per-view per-side union box across reference times, pad to at least64, clip image',occlusion='No true visibility GT. Projected ambiguous/other-body overlap flagged unknown; unobserved physical occlusion remains unverified.',RGB_MSE='mean over RGB elements of whole fixed ROI, including its fixed background pixels',soft_undercoverage='sum(ROI*relu(reference_alpha-pred_alpha))/sum(ROI*reference_alpha)',positive_excess='sum(ROI*relu(pred_alpha-reference_alpha))/sum(ROI*reference_alpha)',temporal='same fixed reference optical flow; same expanded region in consecutive warped frames; report valid fraction')
    allmeta={}
    for label in ['N06','N11']:
        out=root/label;out.mkdir();sample=BASE/'samples'/label;pointsfile=sample/'generation/preflight/conditioning/keypoints_3d.npy';points=np.load(pointsfile);camsfile=sample/'generation/outputs/turbo/cameras.json';cams=json.loads(camsfile.read_text())['cameras'];rows=[];files={}
        selected=[0,(len(points)-1)//2,len(points)-1,34 if label=='N06' else 93];contact=Image.new('RGB',(352*4,670*3),'#333333');ink=ImageDraw.Draw(contact)
        for row,v in enumerate([0,8,16]):
            c=next(c for c in cams if c['camera_id']==v);K=np.array(c['K'],float);K[:2]*=.5;masks=[];known=[]
            cores=np.load(ROOT/'audit/roi'/label/f'view{v:02d}.npz')['labels']
            for f,pts in enumerate(points):
                uv,depth=project(pts,c['camera_to_world'],K);torso=max(float((np.linalg.norm(uv[5]-uv[9])+np.linalg.norm(uv[6]-uv[10]))/2),10.)
                rgba=np.asarray(Image.open(sample/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'),np.float32)/255;alpha=rgba[...,3];both=[];valid=[]
                for side in [0,1]:
                    bones=([[11,13],[13,15]] if side==0 else [[12,14],[14,18]]) if label=='N06' else ([[5,7],[7,62]] if side==0 else [[6,8],[8,41]])
                    r=(.22 if label=='N06' else .18)*torso;last_a,last_b=bones[-1];end=uv[last_b]+.25*(uv[last_b]-uv[last_a]);dist=np.minimum.reduce([segment(uv[a],uv[b]) for a,b in bones]+[segment(uv[last_b],end)]);mask=dist<=r
                    mass=float(alpha[mask].sum());area=int(mask.sum());part_id=(5 if label=='N06' else 3)+side
                    other=float(((cores[f]!=0)&(cores[f]!=part_id)&mask&(alpha>.5)).sum()/max(1,int((mask&(alpha>.5)).sum())))
                    ids=np.unique(bones);infront=bool(np.isfinite(uv[ids]).all() and (depth[ids]>0).all());ok=infront and mass>=64 and area>=256 and other<=.25
                    both.append(mask);valid.append(ok)
                    rows.append(dict(view=v,frame=f,side=side,part=('left_' if side==0 else 'right_')+('distal_leg_foot' if label=='N06' else 'forearm_wrist_arm'),roi_pixels=area,reference_alpha_mass=mass,reference_foreground_pixels=int((mask&(alpha>.5)).sum()),other_core_overlap_fraction=other,projected_front=infront,known=ok,status='proxy_measurable_not_semantic_GT' if ok else 'unknown_projection_area_or_overlap',physical_occlusion='unverified'))
                masks.append(both);known.append(valid)
                if f in selected:
                    col=selected.index(f);rgb=(rgba[...,:3]*alpha[...,None]+.5*(1-alpha[...,None]))*255
                    for side,color in enumerate([[40,230,80],[255,180,30]]):
                        edge=both[side]^(cv2.erode(both[side].astype(np.uint8),np.ones((3,3),np.uint8))>0);rgb[edge]=color
                    contact.paste(Image.fromarray(rgb.astype(np.uint8)),(352*col,670*row+30));ink.text((352*col+3,670*row+8),f'{label} v{v:02d} f{f} valid L/R {valid}',fill='white')
            maskstack=np.array(masks);boxes=[]
            for side in [0,1]:
                ys,xs=np.nonzero(maskstack[:,side].any(0));x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
                cx=(x0+x1)//2;cy=(y0+y1)//2;w=max(64,x1-x0);h=max(64,y1-y0);x0=max(0,min(352-w,cx-w//2));y0=max(0,min(640-h,cy-h//2));boxes.append([x0,y0,x0+w,y0+h])
            p=out/f'view{v:02d}.npz';np.savez_compressed(p,masks=maskstack,known=np.array(known),boxes=np.array(boxes));files[p.name]=sha(p)
        contact.save(out/'review.jpg',quality=93);allmeta[label]=dict(points_sha256=sha(pointsfile),cameras_sha256=sha(camsfile),files=files,rows=rows,known_rows=sum(r['known'] for r in rows),total_rows=len(rows),visual_review='pending')
        (out/'manifest.json').write_text(json.dumps(allmeta[label],indent=2)+'\n')
    (root/'rules.json').write_text(json.dumps(rules,indent=2)+'\n');print({k:(v['known_rows'],v['total_rows']) for k,v in allmeta.items()})
if __name__=='__main__':main()
