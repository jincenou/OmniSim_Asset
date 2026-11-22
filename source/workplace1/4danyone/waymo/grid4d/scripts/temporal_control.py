"""Translation control distinguishes real motion from moving reconstruction error."""
import json
import cv2
import numpy as np
from prepare_diagnostics import ROOT,remap


def main():
    rng=np.random.default_rng(820)
    rgb=cv2.GaussianBlur(rng.random((640,352,3),dtype=np.float32),(5,5),.8)*.6+.2
    flow=np.empty((640,352,2),np.float32);flow[...,0]=-3;flow[...,1]=-2
    next_rgb=remap(rgb,flow)
    bias=np.zeros_like(rgb);bias[180:360,100:220,0]=.1
    pred=rgb+bias;next_pred=next_rgb+remap(bias,flow)
    valid=np.zeros((640,352),bool);valid[30:-30,30:-30]=True
    residual=(next_pred-remap(pred,flow))-(next_rgb-remap(rgb,flow))
    compensated=float(np.square(residual[valid]).mean())
    uncompensated=float(np.square(((next_pred-pred)-(next_rgb-rgb))[valid]).mean())
    assert compensated<1e-12 and uncompensated>1e-6
    blink=next_pred.copy();blink[200:320,120:200,1]+=.1
    blink_residual=(blink-remap(pred,flow))-(next_rgb-remap(rgb,flow))
    blink_score=float(np.square(blink_residual[valid]).mean());assert blink_score>1e-5
    a=cv2.cvtColor(np.rint(rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    b=cv2.cvtColor(np.rint(next_rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    estimated=cv2.calcOpticalFlowFarneback(b,a,None,.5,3,15,3,5,1.2,0)
    e=np.linalg.norm(estimated-flow,axis=-1)[valid]
    assert float(np.median(e))<.5 and float(np.quantile(e,.95))<1.
    report=dict(state='passed',known_translation=[3,2],constant_moving_bias_compensated_mse=compensated,
        same_bias_uncompensated_mse=uncompensated,added_blink_compensated_mse=blink_score,
        estimated_flow_median_error_px=float(np.median(e)),estimated_flow_p95_error_px=float(np.quantile(e,.95)),
        limitation='Synthetic control validates warp direction and motion separation on this texture; real correspondences still filtered and coverage reported')
    (ROOT/'research/temporal_control.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
