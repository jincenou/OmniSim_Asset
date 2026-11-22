"""Reference-only partitions; every metric retains the complete fixed denominator."""
import numpy as np
from boundary_loss import regions,REGIONS


def rows_for_frame(rgb_black,alpha,rgba,core,identity,method,view,frame,domain):
    refalpha=rgba[...,3];ref=rgba[...,:3]*refalpha[...,None]+.5*(1-refalpha[...,None]);pred=np.clip(rgb_black+.5*(1-alpha[...,None]),0,1)
    partition=regions(refalpha);fg=refalpha>=.5;error=np.abs(alpha-refalpha);over=np.maximum(alpha-refalpha,0);under=np.maximum(refalpha-alpha,0)
    rgb_mse=np.square(pred-ref).mean(-1);rgb_mae=np.abs(pred-ref).mean(-1)
    rows=[]
    for code,name in REGIONS.items():
        mask=partition==code;area=int(mask.sum());trusted=mask&(core>0)
        row=dict(label=identity,method=method,domain=domain,view=view,frame=frame,region=name,area=area,unknown_pixels=int((mask&(core==0)).sum()),trusted_core_pixels=int(trusted.sum()),alpha_mae=None,mean_pred_alpha=None,mean_ref_alpha=None,positive_excess_alpha=None,undercoverage_alpha=None,rgb_mse=None,rgb_mae=None,trusted_core_rgb_mse=None)
        if area:row.update(alpha_mae=float(error[mask].mean()),mean_pred_alpha=float(alpha[mask].mean()),mean_ref_alpha=float(refalpha[mask].mean()),positive_excess_alpha=float(over[mask].mean()),undercoverage_alpha=float(under[mask].mean()),rgb_mse=float(rgb_mse[mask].mean()),rgb_mae=float(rgb_mae[mask].mean()))
        if trusted.any():row['trusted_core_rgb_mse']=float(rgb_mse[trusted].mean())
        rows.append(row)
    outside=~fg
    return rows,dict(original_outside_alpha=float(alpha[outside].mean()),reference_outside_soft_alpha=float(refalpha[outside].mean()),outside_positive_excess=float(over[outside].mean()),outside_pixels=int(outside.sum()))
