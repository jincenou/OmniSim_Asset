"""Report the bounded initialization2x2, without selecting/hiding failed runs."""
import csv
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path

R=Path(__file__).resolve().parents[1]
D=R/'diagnostics/weak_velocity'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert read(D/'paired_audit.json')['state']=='only_initial_velocities_changed'
    assert read(D/'paired_audit_sigma8.json')['sigma_only_parameter_changes']=={'zero':['durations'],'body':['durations']}
    roots={'old5k_reference':R/'reconstruction/outputs/r07_short61_24',**{k:D/k for k in ('zero','body','zero_sigma8','body_sigma8')}}
    rows=[];per_time=[]
    for variant,root in roots.items():
        p=root/'quality_metrics/metrics.json';m=read(p)
        assert m['frames']==list(range(0,61,5)) and m['views']==[0,8,16,1,9,17]
        assert max(m['ply_roundtrip']['all_parameter_max_abs'].values())<=1e-6 and m['ply_roundtrip']['render_max_abs']<=1e-5
        for split,backgrounds in m['aggregates'].items():
            for bg,metrics in backgrounds.items():
                rows.append({'variant':variant,'split':split,'background':float(bg),'steps':m['steps'],'gaussians':m['gaussians'],**metrics})
        samples=[x for x in m['samples'] if x['split']=='validation_generated' and x['background']==.5]
        assert len(samples)==39
        for frame in m['frames']:
            group=[x for x in samples if x['frame']==frame];assert len(group)==3
            per_time.append({'variant':variant,'frame':frame,'seconds':frame/25,**{key:sum(x[key] for x in group)/3 for key in ['full_psnr','foreground_psnr','crop_lpips','silhouette_iou','outside_alpha_mean']}})
    output={'updated_at':datetime.now(timezone.utc).isoformat(),'state':'four_controls_and_old_reference_evaluated',
        'scope':'3 heldout generated cameras x13 time samples x3 backgrounds; not real Waymo new-view truth; one trajectory and seed.',
        'rows':rows,'per_time_gray':per_time,'sources':{k:{'metrics':str(v/'quality_metrics/metrics.json'),'sha256':sha(v/'quality_metrics/metrics.json')} for k,v in roots.items()}}
    (D/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    with (D/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gray={x['variant']:x for x in rows if x['split']=='validation_generated' and x['background']==.5}
    table=['|配置|步数|初始sigma帧|前景PSNR↑|裁剪LPIPS↓|IoU↑|mask外alpha↓|','|---|---:|---:|---:|---:|---:|---:|']
    for variant in roots:
        x=gray[variant];sigma=8 if variant.endswith('sigma8') else 15
        table.append(f'|{variant}|{x["steps"]}|{sigma}|{x["foreground_psnr"]:.3f}|{x["crop_lpips"]:.4f}|{x["silhouette_iou"]:.4f}|{x["outside_alpha_mean"]:.5f}|')
    text='''# R07运动初始化与初始时间宽度对照

单轨迹、单seed、同一次24视角生成、同一352×640 RGBA、21/3视角划分、11,222点。四组15k同计划从头优化，初始参数哈希逐项核对；不是继续旧5k训练。旧5k只作不同计划的参考。

只比较两个明确变化：初始速度zero/body；初始高斯时间sigma为15帧(0.6秒)或8帧(0.32秒)。后者只改初始值，优化期间最大持续时间约束不变，不能与此前“限制最大duration8”实验混为一谈。

body来自已用于生成条件的GVHMR对应人体顶点速度，按距离衰减投给同时间轮廓表面。未使用留出图像初始化，也没有声称是衣服/包的真实材料点跟踪。所有初始字段除对应变量外逐项一致；训练后的重定位自然可能产生不同路径。

下面是灰背景39个留出生成观测的平均结果。黑/白背景、训练视角诊断和每个采样时间都在results.json；没有真实Waymo新视角真值。

'''+ '\n'.join(table) + '\n\n'
    for suffix in ('','_sigma8'):
        z=gray['zero'+suffix];b=gray['body'+suffix]
        text+=f'同一初始sigma下，body相对zero：LPIPS变化{(b["crop_lpips"]/z["crop_lpips"]-1)*100:+.2f}%，前景PSNR变化{b["foreground_psnr"]-z["foreground_psnr"]:+.3f}dB（sigma={8 if suffix else 15}帧）。\n\n'
    text+='[原5k与第一对完整动作](comparison/view08.mp4) · [四个15k配置完整动作](comparison_sigma8/view08.mp4) · [第一对参数审计](paired_audit.json) · [第二对及sigma变化审计](paired_audit_sigma8.json)\n\n本报告为数值结果；抽样画面判断与不确定性另见visual_review.json。不能据此判定人体运动先验普遍无效，也不能认定当前表示无法表达行走。\n'
    (D/'REPORT.md').write_text(text)
    cache=R/'tmp/matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    for ax,key,title in zip(axes,['crop_lpips','foreground_psnr','outside_alpha_mean'],['Crop LPIPS (lower better)','Foreground PSNR (higher better)','Outside alpha (lower better)'],strict=True):
        for variant in roots:
            sample=[x for x in per_time if x['variant']==variant]
            ax.plot([x['seconds'] for x in sample],[x[key] for x in sample],marker='.',label=variant)
        ax.set_xlabel('Time (seconds)');ax.set_title(title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle('R07 / held-out GENERATED views / gray background / no real novel-view ground truth',fontsize=10)
    fig.savefig(D/'time_metrics.png',dpi=170);fig.savefig(D/'time_metrics.pdf');plt.close(fig)
    print(json.dumps({'state':output['state'],'gray_results':gray}),flush=True)

if __name__=='__main__':main()
