"""Reuse existing evidence; publish only complete HUST/A/fused comparison families."""
import csv,json
from pathlib import Path
import publish_quality as media
NEW=Path(__file__).resolve().parents[1];ROOT=NEW.parent;V3=ROOT/'optimization_quality_v3_20261018';PLAN=json.loads((NEW/'PLAN.json').read_text())
def descriptors(label,arm):
    key=label+'_'+arm
    if key in PLAN['reuse']:return {k:Path(v) for k,v in PLAN['reuse'][key].items() if k in ['run','results','raw']}
    it=53000 if arm=='A' else 33000;name=key+'_'+str(it)
    return dict(run=NEW/'runs'/key,results=NEW/'results'/name,raw=NEW/'cache/raw'/name)
def ready(label):return all((descriptors(label,a)['results']/'parts.csv').exists() and (descriptors(label,a)['raw']/'complete.json').exists() for a in ['A','fused'])
def publish_label(label):
    if label in ['N01','N07']:return
    out=NEW/'comparison'/label
    assert not out.exists(),out
    job=dict(name=label,label=label,columns=[('HUST grid61',ROOT/'tmp/baseline_raw'/label/'4dgs_grid61'),('A 52999',descriptors(label,'A')['raw']),('Fused 32999',descriptors(label,'fused')['raw'])],note='Same fixed input and seed; separate stochastic GPU trajectories. Compare complete recipes, not a pure single-factor causal estimate.')
    media.publish(job)
def overview():
    rows=[];html=['<!doctype html><meta charset="utf-8"><title>8人完整A与融合版对照</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto;padding:20px}video,img{max-width:100%}section{padding:20px;border-top:1px solid #ddd}</style><h1>8人固定配方对照</h1><p>主样本N01/02/03/04/06/07/08，备用N11单列。生成参考开发指标；未完成对象仅显示状态，不生成缺列视频。</p><p><a href="../REPORT.md">执行说明</a> · <a href="../paired_metrics.csv">指标表</a></p>']
    for label in PLAN['labels']:
        html.append(f'<section><h2>{label}'+('（备用）' if label=='N11' else '')+'</h2>')
        for method in ['HUST61','A','fused']:
            p=media.BASE/'samples'/label/'evaluations/4dgs_grid61/results.json' if method=='HUST61' else descriptors(label,method)['results']/'metrics.json'
            if p.exists():
                d=json.loads(p.read_text());rows.append(dict(label=label,cohort='reserve' if label=='N11' else 'main',method=method,**{k:d['mean'][k] for k in ['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']},source=str(p)))
        if label in ['N01','N07']:
            href=f'../../optimization_quality_v3_20261018/comparison/{label}_A_fusion'
            html.append('<p>复用已完成对照：参考/HUST/原生/alpha/A/融合六列。</p>')
        elif (NEW/'comparison'/label/'validation.json').exists():
            href=label;html.append('<p>列顺序：生成参考 → HUST grid61 → 完整A → 融合版。</p>')
        else:html.append('<p>训练/评测中；完整视频尚未发布。</p></section>');continue
        html.append(f'<img loading="lazy" src="{href}/parts_rgb.jpg">')
        for view in [0,8,16]:html.append(f'<p>视角{view:02d}</p><video controls loop preload="metadata" src="{href}/view{view:02d}.mp4"></video>')
        html.append('</section>')
    with (NEW/'paired_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p=NEW/'comparison/index.html';tmp=p.with_suffix('.writing');tmp.write_text('\n'.join(html));tmp.replace(p)
