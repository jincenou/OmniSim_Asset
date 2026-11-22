"""Source-camera O0/O1/O2 comparison with common COCO regressor and evidence."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch,av
from PIL import Image,ImageDraw
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]/'repo'
sys.path.insert(0,str(REPO))
from fdanyone.skeleton.pipeline import _gvhmr_geometry_context
EDGES=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sample');parser.add_argument('--variant',choices=['v1','root_only','local_temporal','rohm'],default='v1');args=parser.parse_args();torch.set_num_threads(2)
    root=ROOT/'motion_candidates'/args.sample if args.variant=='v1' else ROOT/'motion_candidates_v2'/args.sample/args.variant
    report=json.loads((root/'report.json').read_text());source=Path(report['source'])
    result=load_file(str(root/'motion_candidates.safetensors'));old=load_file(str(source/'motion/motion.safetensors'))
    K=old['K_fullimg'].numpy();kp=old['observed_keypoints_2d'].numpy();valid=result['evidence_supported_visible_candidate'].numpy()
    raw={k:old['smpl_params_incam.'+k] for k in ['body_pose','betas','global_orient','transl']}
    # All variants share frozen shape for the parameter-comparison overlay.
    raw['betas']=result['betas']
    proposal_key='proposal_initial.body_pose' if 'proposal_initial.body_pose' in result else 'proposal_O1.body_pose'
    variants={'O0 original':raw,'RoHM proposal' if args.variant=='rohm' else 'O1 bounded proposal':dict(raw,body_pose=result[proposal_key]),
      'O2 camera constrained':dict(raw,body_pose=result['body_pose'],global_orient=result['incam.global_orient'],transl=result['incam.transl'])}
    projected={};joints={}
    with _gvhmr_geometry_context(REPO/'third_party/GVHMR'):
        from hmr4d.utils.smplx_utils import make_smplx
        model=make_smplx('supermotion_coco17').eval()
        with torch.inference_mode():
            for name,params in variants.items():
                points=model(**params).numpy();h=np.einsum('tij,tnj->tni',K,points)
                projected[name]=h[...,:2]/h[...,2:];joints[name]=points
    frames=[]
    with av.open(str(source/'input/model_input.mp4')) as c:
        c.streams.video[0].thread_count=1
        for frame in c.decode(video=0):frames.append(frame.to_image())
    assert len(frames)==len(kp)
    w,h=frames[0].size;panels=[]
    for t,frame in enumerate(frames):
        panel=Image.new('RGB',(w*3,h+26),'#222222');d=ImageDraw.Draw(panel)
        for column,(name,uv) in enumerate(projected.items()):
            im=frame.copy();draw=ImageDraw.Draw(im)
            for a,b in EDGES:draw.line([tuple(uv[t,a]),tuple(uv[t,b])],fill='lime',width=2)
            for j,(x,y,score) in enumerate(kp[t]):
                color='red' if valid[t,j] else 'orange';draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            panel.paste(im,(column*w,26));d.text((column*w+3,5),f'{name if column<2 else args.variant} t={t/25:.2f}',fill='white')
        panels.append(panel)
    with av.open(str(root/'projection_comparison.mp4'),'w') as output:
        stream=output.add_stream('libx264',rate=25);stream.width=w*3;stream.height=h+26;stream.pix_fmt='yuv420p';stream.options={'crf':'19','threads':'2'}
        for panel in panels:
            for packet in stream.encode(av.VideoFrame.from_image(panel)):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    slots=np.rint(np.linspace(0,len(frames)-1,5)).astype(int);contact=Image.new('RGB',(w*3,(h+26)*len(slots)))
    for row,slot in enumerate(slots):contact.paste(panels[slot],(0,row*(h+26)))
    contact.save(root/'projection_contact.jpg',quality=94)
    with av.open(str(root/'projection_comparison.mp4')) as c:assert sum(1 for _ in c.decode(video=0))==len(frames)
    # Remove per-frame rigid root rotation and translation before measuring local articulation.
    from scipy.spatial.transform import Rotation
    temporal={};times=result['times_seconds'].numpy();scale=report['body_scale_m'];dt=np.diff(times)
    for name,points in joints.items():
        params=variants[name];R=Rotation.from_rotvec(params['global_orient'].numpy()).as_matrix()
        local=np.einsum('tji,tnj->tni',R,points-params['transl'].numpy()[:,None,:])
        local-=local[:,11:13].mean(1,keepdims=True)
        velocity=np.diff(local,axis=0)/dt[:,None,None]
        acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None,None]
        temporal[name]={'local_motion_range_bodyheight_rms':float(np.sqrt(np.mean(np.ptp(local,axis=0)**2))/scale),
          'local_acceleration_bodyheight_per_s2_rms':float(np.sqrt(np.mean(acceleration**2))/scale)}
    (root/'projection_review.json').write_text(json.dumps({'videos_fully_decoded':True,'frames':len(frames),'common_regressor':'GVHMR SmplxLiteCoco17','temporal_diagnostics':temporal,
      'quality_approved':False,'manual_review':'pending','scope':'Local acceleration/range are estimates, not jitter ground truth; small range can mean oversmoothing. Green=fitted joints, red=supported detector, orange=unknown. Source-camera fit does not prove occluded motion.'},indent=2)+'\n')
    print(json.dumps({'sample':args.sample,'temporal':temporal,'state':'review_media_complete_not_quality_passed'},indent=2))
if __name__=='__main__':main()
