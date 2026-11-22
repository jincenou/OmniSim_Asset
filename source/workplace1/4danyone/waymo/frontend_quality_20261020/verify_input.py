"""Verify the unmasked motion ablation against delivered scene20 data."""
from pathlib import Path
import os,json,hashlib
import av
import numpy as np

ROOT=Path(os.environ['FDANYONE_SAMPLE_ROOT'])
OLD=Path(os.environ['FDANYONE_BASELINE'])
new=json.loads((ROOT/'native.json').read_text());old=json.loads((OLD/'input/native.json').read_text())
assert len(new['frames'])==len(old['frames'])
assert [x['sha256'] for x in new['frames']]==[x['sha256'] for x in old['frames']]
a=np.load(ROOT/'inputs/cameras.npz');b=np.load(OLD/'input/source_cameras.npz')
errors={k:float(np.abs(a[k]-b[k]).max()) for k in b.files}
assert max(errors.values())<1e-9,errors
masks=np.load(OLD/'input/target_masks.npz')['masks']
with av.open(str(ROOT/'inputs/main.mp4')) as left,av.open(str(OLD/'input/model_input.mp4')) as right:
 left.streams.video[0].thread_count=right.streams.video[0].thread_count=1
 count=0;abs_sum=0;elements=0;maximum=0
 for i,(f,g) in enumerate(zip(left.decode(video=0),right.decode(video=0),strict=True)):
  rgb=f.to_ndarray(format='rgb24');alpha=masks[i].astype(np.float32)[...,None]/255
  composed=np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)
  diff=np.abs(composed.astype(np.int16)-g.to_ndarray(format='rgb24').astype(np.int16))
  count+=1;abs_sum+=int(diff.sum());elements+=diff.size;maximum=max(maximum,int(diff.max()))
assert count==len(masks)==int(os.environ['FDANYONE_FRAMES'])
mae=abs_sum/elements
result=dict(state='verified' if mae<=.05 and maximum<=8 else 'reproduction_mismatch',
 frames=count,native_png_hashes_identical=True,camera_max_errors=errors,
 recomposed_masked_mae_rgb8=mae,recomposed_masked_max_abs_rgb8=maximum,
 exact_input_pixel_pairing=maximum==0,
 acceptance='Before motion inference: native hashes exact,camera arrays <1e-9; recomposition MAE <=0.05 and max <=8 RGB8. Nonzero values retained as reproduction nuisance, not declared exact.',
 motion_input_variant='unsegmented_crop',appearance_input='baseline unchanged; not regenerated',
 baseline=str(OLD))
(ROOT/'input_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert result['state']=='verified',result
print(json.dumps(result))
