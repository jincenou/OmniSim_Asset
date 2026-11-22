"""O1 proposal primitive; evidence is supplied, never upgraded by interpolation."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def bounded_rotation_gaps(rotvec,times,reliable,max_gap_seconds):
    p=np.asarray(rotvec,dtype=float);t=np.asarray(times,dtype=float);valid=np.asarray(reliable,dtype=bool)
    if p.ndim!=3 or p.shape[-1]!=3 or valid.shape!=p.shape[:2] or t.shape!=(len(p),):raise ValueError('Incompatible arrays')
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.all(np.diff(t)>0) or max_gap_seconds<=0:raise ValueError('Invalid values')
    out=p.copy();changed=np.zeros(valid.shape,dtype=bool);records=[]
    for j in range(p.shape[1]):
        edge=np.diff(np.r_[False,~valid[:,j],False].astype(np.int8))
        for start,end in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1),strict=True):
            bounded=start>0 and end<len(p)
            duration=float(t[end]-t[start-1]) if bounded else None
            eligible=bounded and duration<=max_gap_seconds+1e-9
            record=dict(joint=j,start=int(start),end_exclusive=int(end),anchor_interval_seconds=duration,
                proposed=bool(eligible),reason='bounded_short_gap' if eligible else 'unbounded_or_long_unknown')
            records.append(record)
            if not eligible:continue
            anchor=[start-1,end]
            out[start:end,j]=Slerp(t[anchor],Rotation.from_rotvec(p[anchor,j]))(t[start:end]).as_rotvec()
            changed[start:end,j]=True
    return out,changed,records
