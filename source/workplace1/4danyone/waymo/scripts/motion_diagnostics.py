"""Describe estimated foot stability without treating it as motion ground truth."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'repo'))
from fdanyone.skeleton.keypoints import KEYPOINT_NAMES


def main():
    points=np.load(ROOT/'preflight/conditioning/keypoints_3d.npy')
    assert points.shape==(121,len(KEYPOINT_NAMES),3) and np.isfinite(points).all()
    feet={}
    for name in ('left-heel','right-heel','left-ankle','right-ankle'):
        p=points[:,KEYPOINT_NAMES.index(name)]
        feet[name]={'horizontal_range_m':np.ptp(p[:,[0,2]],axis=0).tolist(),
                    'height_range_m':float(np.ptp(p[:,1])),
                    'horizontal_path_m':float(np.linalg.norm(np.diff(p[:,[0,2]],axis=0),axis=1).sum()),
                    'max_step_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).max())}
    output={'space':'4DAnyone canonical estimated motion, not Waymo metric ground truth',
            'duration_seconds':4.8,'feet':feet,
            'all_keypoints_max_step_m':float(np.linalg.norm(np.diff(points,axis=0),axis=2).max()),
            'interpretation':'Standing-clip diagnostic; displacement can include model drift and real small motion. This does not establish ground-contact correctness or zero foot sliding.'}
    (ROOT/'preflight/motion_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
