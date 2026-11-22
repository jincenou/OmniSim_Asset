"""Preserve raw output, repair saturated opacity serialization, export/reload."""
from pathlib import Path
import argparse,json
import numpy as np
from plyfile import PlyData,PlyElement
from asset_adapter import ROOT,export

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True)
    p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=ROOT/'results'/a.case/f'{a.variant}_seed{a.seed}'
    source=out/'raw.ply';v=PlyData.read(str(source))['vertex'].data.copy()
    for name in v.dtype.names:
        if name!='opacity' and not np.isfinite(v[name]).all():raise ValueError(name)
    opacity=v['opacity'];bad=np.isinf(opacity)
    if np.isnan(opacity).any():raise ValueError('NaN opacity cannot be repaired')
    report=dict(source=str(source),nonfinite_opacity_count=int(bad.sum()),
        positive_infinity_count=int(np.isposinf(opacity).sum()),negative_infinity_count=int(np.isneginf(opacity).sum()),
        repair='Replace only infinite opacity logits with +/- logit(1 - 1e-7); preserve original raw.ply',
        maximum_activated_opacity_change=1e-7)
    if bad.any():
        bound=float(np.log((1-1e-7)/1e-7));opacity[bad]=np.sign(opacity[bad])*bound
        source=out/'finite.ply'
        if source.exists():raise FileExistsError(source)
        PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(source))
    (out/'serialization_repair.json').write_text(json.dumps(report,indent=2))
    case=json.loads((ROOT/'inputs'/a.case/'manifest.json').read_text())
    export(source,out/'asset',case,a.variant,'-y' if a.variant in ('B0','B0_real1','B0_real4','B0_repeat') else 'z')

if __name__=='__main__':main()
