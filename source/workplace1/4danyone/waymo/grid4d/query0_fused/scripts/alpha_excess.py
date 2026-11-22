"""Uniform 39-observation outside alpha/excess, directly from raw alpha arrays."""
import asyncio,json,sys,os
import numpy as np
from PIL import Image
from analyze_query import NEW,ROOT,PLAN,BASE,descriptor,read,write
from runtime import save
def one(label,method):
    d=descriptor(label,method);m=read(d['metrics']);cached=m.get('dense_samples',[]);by={(r['view'],r['frame']):r for r in cached};rows=[]
    supplemental=NEW/'results'/f'{label}_alpha_48000'/'metrics.json'
    if method=='alpha48' and supplemental.exists():by={(r['view'],r['frame']):r for r in read(supplemental).get('dense_samples',[])}
    for view in [0,8,16]:
        samples=[r for r in m['samples'] if r['view']==view];need=any('outside_positive_excess' not in by.get((view,r['frame']),{}) for r in samples);alpha=None
        if need:
            with np.load(d['raw']/f'view{view:02d}.npz') as z:alpha=z['alpha']
        for r in samples:
            frame=r['frame'];v=by.get((view,frame),{})
            if 'outside_positive_excess' in v:outside,ref,excess,pixels=[v[k] for k in ['original_outside_alpha','reference_outside_soft_alpha','outside_positive_excess','outside_pixels']]
            else:
                refa=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)[...,3]/255;mask=refa<.5;a=alpha[frame];outside=float(a[mask].mean());ref=float(refa[mask].mean());excess=float(np.maximum(a-refa,0)[mask].mean());pixels=int(mask.sum())
            assert abs(outside-r['outside_alpha_mean'])<1e-6
            rows.append(dict(label=label,method=method,view=view,frame=frame,outside_pixels=pixels,original_outside_alpha=outside,reference_outside_soft_alpha=ref,outside_positive_excess=excess))
    return rows
async def main():
    sem=asyncio.Semaphore(2)
    async def bounded(l,m):
        async with sem:
            proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'--one',l,m,stdout=asyncio.subprocess.PIPE,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
            stdout,_=await proc.communicate();assert proc.returncode==0,(l,m,proc.returncode);return json.loads(stdout)
    rows=await asyncio.gather(*(bounded(l,m) for l in PLAN['labels'] for m in ['HUST','F','alpha48','A']+(['U','P','E','Q'] if l in PLAN['query_labels'] else [])))
    write(NEW/'outside_alpha.csv',[r for group in rows for r in group]);save(NEW/'research/outside_alpha_audit.json',dict(state='passed',rows=sum(map(len,rows)),protocol='Same39 global observations; alpha from raw float arrays or existing exact dense metrics. Outside is reference alpha<0.5; positive excess=max(pred-reference,0), all original outside denominators preserved. Full-time regional values also retained for all new terminals.'))
    print('outside alpha complete',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:
        assert sys.argv[1]=='--one';print(json.dumps(one(sys.argv[2],sys.argv[3]),allow_nan=False))
    else:asyncio.run(main())
