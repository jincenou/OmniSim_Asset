"""CPU analysis of completed checkpoints and frozen per-frame metrics."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
OUT=R/'diagnostics/walking_temporal'
def quantiles(x):return dict(zip(('p05','p50','p95'),map(float,np.quantile(x,[.05,.5,.95]))))
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    report={'scope':'Generated-view consistency and parameter diagnostics, not source-motion accuracy or causal proof',
            'seconds_per_normalized_time':4.8,'spatial_units':'backend normalized scene units, not Waymo meters','recipes':{}}
    references=None
    for recipe in ('q0','budget_half_50k','capacity50k'):
        metrics=json.loads((R/'benchmark/results'/f'walk063_rootlocal24_{recipe}'/'metrics.json').read_text())
        if references is None:references=metrics['inputs']
        if references!=metrics['inputs']:raise ValueError('Reference images differ')
        checkpoint=Path(metrics['job']['checkpoint'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        splats=state['splats']
        if not all(torch.isfinite(v).all() for v in splats.values()):raise ValueError('Nonfinite checkpoint')
        times=splats['times'].detach().numpy().ravel();sigma=splats['durations'].exp().detach().numpy().ravel()
        base=splats['opacities'].sigmoid().detach().numpy().ravel()
        velocity=np.linalg.norm(splats['velocities'].detach().numpy(),axis=-1)
        weights=np.exp(-.5*((np.arange(121)[:,None]/120-times[None])/sigma[None])**2)*base[None]
        samples=metrics['samples'];curves={}
        for key in ('foreground_psnr','lpips','outside_alpha_per_foreground_pixel'):
            curves[key]=[float(np.mean([s[key] for s in samples if s['frame']==f])) for f in range(121)]
        for ax,key in zip(axes,curves,strict=True):ax.plot(np.arange(121)/25,curves[key],label=recipe);ax.set_ylabel(key);ax.grid(alpha=.25)
        windows={name:{key:float(np.mean(np.array(values)[ids])) for key,values in curves.items()}
                 for name,ids in [('early',slice(0,40)),('middle',slice(40,81)),('late',slice(81,121))]}
        report['recipes'][recipe]={'checkpoint':str(checkpoint),'checkpoint_sha256':metrics['job']['checkpoint_sha256'],
            'gaussians':len(times),'duration_sigma_seconds':quantiles(sigma*4.8),
            'velocity_per_second':quantiles(velocity/4.8),'displacement_over_one_sigma':quantiles(velocity*sigma),
            'time_centers_outside_clip':int(((times<0)|(times>1)).sum()),
            'base_opacity':quantiles(base),'effective_point_count_alpha_gt005':quantiles((weights>.05).sum(1)),
            'endpoint_effective_points':[(weights[[0,-1]]>.05).sum(1).tolist()][0],
            'windows':windows,'curves':curves,
            'worst_lpips_frames':sorted(range(121),key=lambda f:curves['lpips'][f],reverse=True)[:8]}
        del state,splats,weights
    axes[-1].set_xlabel('Video seconds');axes[0].legend();fig.tight_layout();fig.savefig(OUT/'per_time.png',dpi=160);plt.close(fig)
    (OUT/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('duration_sigma_seconds','effective_point_count_alpha_gt005','windows','worst_lpips_frames')} for k,v in report['recipes'].items()}))

if __name__=='__main__':main()
