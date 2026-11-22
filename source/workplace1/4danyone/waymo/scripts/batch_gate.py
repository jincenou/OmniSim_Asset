"""Declared automatic screening; passing is not a visual-quality endorsement."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT'])


def main():
    masks=np.load(ROOT/'segmentation/target_masks.npz')['masks']>127
    mapping=json.loads((ROOT/'inputs/frame_mapping.json').read_text())
    boxes=np.array([f['bbox_input_xyxy'] for f in mapping['frames']])
    area=masks.sum((1,2))/np.maximum((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),1)
    core=[]
    for mask,b in zip(masks,boxes,strict=True):
        x0,y0,x1,y1=b;w=x1-x0;h=y1-y0
        xa,ya,xb,yb=np.rint([x0+.3*w,y0+.2*h,x0+.7*w,y0+.6*h]).astype(int)
        patch=mask[max(ya,0):min(yb,mask.shape[0]),max(xa,0):min(xb,mask.shape[1])]
        core.append(float(patch.mean()) if patch.size else 0.)
    passed=bool(np.min(area)>.15 and np.median(area)>.3 and np.min(area)/max(np.median(area),1e-6)>.55 and np.min(core)>.3 and np.quantile(core,.1)>.55)
    record={'passed':passed,'mask_bbox_area_min':float(area.min()),'mask_bbox_area_median':float(np.median(area)),
            'torso_coverage_min':min(core),'torso_coverage_p10':float(np.quantile(core,.1)),
            'thresholds':{'area_min':.15,'area_median':.3,'min_to_median_area':.55,'torso_min':.3,'torso_p10':.55},
            'scope':'Automated rejection of empty masks and major occlusion; does not certify identity or carried-object completeness'}
    if sys.argv[1]=='final':
        framing=json.loads((ROOT/'preflight/crop_audit.json').read_text())
        record['framing']=framing;passed=passed and framing['gate_passed'] and len(framing['target_body_coverage'])==24
        record['passed']=passed
        (ROOT/'preflight/review.json').write_text(json.dumps({'allow_generation':passed,'review_type':'automatic_batch_screening',
            'not_manually_reviewed':True,'checks':record,'limitations':['Generated appearance requires later visual audit','Waymo world placement not calibrated']},indent=2)+'\n')
    (ROOT/'research'/f'batch_gate_{sys.argv[1]}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
