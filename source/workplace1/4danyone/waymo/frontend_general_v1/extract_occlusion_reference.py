"""Evaluation-only native detections; never used by motion reconstruction."""
from pathlib import Path
import argparse,json,os,sys,hashlib
import torch,numpy as np
from PIL import Image
G=Path(__file__).resolve().parent;W=G.parent;REPO=W.parent/'repo';D=W/'frontend_quality_20261020';O=G/'occlusion_v1'
sys.path[:0]=[str(W/'scene20/r20261019/scripts'),str(REPO),str(W/'scripts')]
from fdanyone.motion import gvhmr
from prepare import encode
from interpolate import device_check

def main():
 a=argparse.ArgumentParser();a.add_argument('sample',choices=['P24','P36']);args=a.parse_args();torch.set_num_threads(2);device=device_check()
 out=O/'evaluation_reference'/args.sample;out.mkdir(parents=True,exist_ok=False)
 native=json.loads((D/'motion_unmasked'/args.sample/'native.json').read_text());images=[Image.open(row['path']).convert('RGB') for row in native['frames']]
 video=out/'native_observations.mp4';encode(video,images,10)
 import av
 with av.open(str(video)) as container:
  decoded=[frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
 assert len(decoded)==len(images) and all(np.array_equal(x,np.asarray(y)) for x,y in zip(decoded,images,strict=True))
 root,revision=gvhmr.validate_gvhmr(REPO/'third_party/GVHMR')
 with gvhmr.gvhmr_imports(root),gvhmr._legacy_checkpoint_loading():
  gvhmr.install_pytorch3d_compat();gvhmr._register_inference_store()
  from hmr4d.utils.preproc.vitpose import VitPoseExtractor
  from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
  boxes=torch.tensor([r['bbox_input_xyxy'] for r in native['frames']],dtype=torch.float32)
  kp=VitPoseExtractor().extract(str(video),get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float())
 np.savez_compressed(out/'native_detections.npz',keypoints=kp.detach().cpu().numpy(),boxes=boxes.numpy(),native_timestamps=np.array([r['image_center_timestamp_s'] for r in native['frames']]))
 (out/'report.json').write_text(json.dumps({'scope':'evaluation only; automatic 2D detector proxy on actual native frames; not pose ground truth','sample':args.sample,'frames':len(images),'lossless_native_video_decode':True,'playback_rate_nominal':10,'actual_exposure_timestamps_saved':True,'device':device,'gvhmr_revision':revision,'sha256':hashlib.sha256((out/'native_detections.npz').read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
