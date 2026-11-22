"""Read-only processed calibration consistency; does not certify source projection."""
import asyncio
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=Path('/mnt/public_data/waymo/val')

def check(row):
    base=PUBLIC/row['scene'];cam=row['camera_index'];intr=np.loadtxt(base/'intrinsics'/f'{cam}.txt').reshape(-1)
    local=np.loadtxt(base/'cam_to_ego'/f'{cam}.txt');cache=json.loads(Path(row['cache']).read_text())
    obs=row['observations'];samples=[obs[0],obs[len(obs)//2],obs[-1]]
    matrices=[np.loadtxt(base/'cam_to_world'/f"{o['frame_index']:03d}_{cam}.txt") for o in samples]
    ego=np.asarray(cache['poses'])[[o['frame_index'] for o in samples]]
    matrices=np.asarray(matrices)
    compose=float(np.max(np.abs(matrices-ego@local)))
    rotations=matrices[:,:3,:3]
    orth=float(np.max(np.abs(rotations@np.swapaxes(rotations,-1,-2)-np.eye(3))))
    det=float(np.max(np.abs(np.linalg.det(rotations)-1)))
    time=np.asarray([o['timestamp_micros'] for o in row['observations']],dtype=np.int64)
    passed=bool(np.isfinite(intr).all() and intr.shape==(9,) and intr[0]>0 and intr[1]>0 and compose<1e-5 and orth<1e-5 and det<1e-5 and np.all(np.diff(time)>0))
    return {'review_id':row['review_id'],'sampled_frame_indices':[o['frame_index'] for o in samples],
      'passed_processed_consistency':passed,'compose_max_abs':compose,
      'rotation_orthogonality_max_abs':orth,'rotation_det_error':det,'native_intrinsics_distortion':intr.tolist(),
      'time_deltas_s_range':[float(np.diff(time).min()/1e6),float(np.diff(time).max()/1e6)],
      'verified':'Stored camera-to-world agrees with cached ego-to-world times stored camera-to-ego',
      'not_verified':['raw extrinsic convention and image exposure pose','lens undistortion status of processed image','SMPL source projection and metric scale','camera axis conversion for GVHMR']}

async def main():
    rows=json.loads((ROOT/'sample_manifest.json').read_text())['candidates'];sem=asyncio.Semaphore(2)
    async def one(row):
        async with sem:return await asyncio.to_thread(check,row)
    results=await asyncio.gather(*map(one,rows))
    report={'passed_processed_consistency':all(r['passed_processed_consistency'] for r in results),'count':len(results),'source_projection_verified':False,'results':results}
    (ROOT/'calibration_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    if not report['passed_processed_consistency']:raise RuntimeError('Processed calibration inconsistency; inspect audit')
if __name__=='__main__':asyncio.run(main())
