"""Complete fixed-terminal families only; reuse matching old alpha/A media."""
import asyncio,json,os,time,sys
from pathlib import Path
from analyze_query import descriptor,PLAN,NEW,ROOT
from publish_quality import publish,sha
from runtime import save

def jobs():
    result=[]
    for label in PLAN['query_labels']:
        result.append(dict(name=label+'_query',label=label,columns=[(title,descriptor(label,m)['raw']) for m,title in [('F','F 32999'),('U','U 34999'),('P','P 34999'),('E','E 34999'),('Q','Q 34999')]],note='Same fused parent; 2000 updates each. U uniform; P balanced parts; E foreground error; Q part-time demand.'))
        result.append(dict(name=label+'_reference',label=label,columns=[(title,descriptor(label,m)['raw']) for m,title in [('HUST','HUST grid61'),('alpha48','Alpha 47999'),('A','A 52999'),('Q','Q 34999')]],note='Different fixed recipes; same generated reference, cameras and timeline. No per-person checkpoint selection.'))
    for label in ['N02','N03','N08']:
        result.append(dict(name=label+'_alpha',label=label,columns=[(title,descriptor(label,m)['raw']) for m,title in [('alpha48','Alpha 47999'),('A','A 52999')]],note='Fixed-stage alpha/A evaluation only, no new optimization. Unknown body cores remain unknown.'))
    return result

async def main():
    sem=asyncio.Semaphore(2);items=jobs()
    async def bounded(job):
        async with sem:
            dest=NEW/'comparison'/job['name']
            if dest.exists():return json.loads((dest/'validation.json').read_text())
            env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'--family',job['name'],env=env)
            code=await proc.wait();assert code==0,(job['name'],code)
            result=json.loads((dest/'validation.json').read_text());print('published',job['name'],flush=True);return result
    results=await asyncio.gather(*(bounded(j) for j in items));reused=[]
    for label in ['N01','N07']:
        folder=ROOT/'optimization_quality_v3_20261018/comparison'/f'{label}_A_fusion';val=json.loads((folder/'validation.json').read_text())
        assert all(sha(folder/r['path'])==r['sha256'] for r in val['videos'])
        reused.append(dict(label=label,folder=str(folder),validation=val,source='Previously fully decoded media reused unchanged'))
    html=['<!doctype html><meta charset="utf-8"><title>Grid4D Query-0 / Alpha-A</title><style>body{font:16px sans-serif;max-width:1700px;margin:auto;padding:24px;background:#eee;color:#222}section{background:white;padding:20px;margin:20px 0}video,img{max-width:100%;height:auto}summary{cursor:pointer}a{color:#1461a2}</style><h1>Grid4D Query-0 与八人 Alpha / A 固定阶段对照</h1><p>Waymo → 4DAnyone Turbo 固定多视角视频 → Grid4D 动态资产。这里的参考是生成视频；H={0,8,16} 是已使用过的开发视角，不是真实 Waymo 新视角真值。</p><p>三人 U/P/E/Q 均从同一 F32999 追加 2000 次更新。Q 未通过预注册增益要求，未运行条件性 A+U/Q。N11 为备用困难对象，单列。</p><p><a href="../FINDINGS.md">结果报告</a> · <a href="../paired_metrics.csv">指标 CSV</a> · <a href="../costs.csv">成本</a> · <a href="../gate_results.json">完整判据</a> · <a href="../ASSETS.json">动态资产</a></p><p>局部图是冻结的历史裁剪；视频均全帧自动解码。人工视觉检查范围见 visual_review.json，自动解码不等于质量合格。</p>']
    for job,result in zip(items,results):
        name=job['name'];html.append(f'<section><h2>{name}</h2><p>{job["note"]}</p><p>列：'+ ' → '.join(result['columns'])+'</p>')
        html.append(f'<img loading="lazy" src="{name}/parts_rgb.jpg"><details><summary>Alpha 局部</summary><img loading="lazy" src="{name}/parts_alpha.jpg"></details>')
        for view in [0,8,16]:html.append(f'<details><summary>完整视角 {view:02d}</summary><video controls loop preload="none" src="{name}/view{view:02d}.mp4"></video><img loading="lazy" src="{name}/view{view:02d}_contact.jpg"></details>')
        html.append('</section>')
    for item in reused:
        rel=os.path.relpath(item['folder'],NEW/'comparison');html.append(f'<section><h2>{item["label"]} Alpha / A 已有完整对照（复用）</h2><p>列：'+ ' → '.join(item['validation']['columns'])+f'</p><img loading="lazy" src="{rel}/parts_rgb.jpg">')
        for view in [0,8,16]:html.append(f'<details><summary>完整视角 {view:02d}</summary><video controls loop preload="none" src="{rel}/view{view:02d}.mp4"></video></details>')
        html.append('</section>')
    html.append('<section><h2>采样与时序诊断</h2>'+''.join(f'<p><a href="../research/{l}_diagnostics.png">{l} 左右部位曲线与曝光分布</a></p>' for l in PLAN['query_labels'])+'</section>')
    (NEW/'comparison/index.html').write_text('\n'.join(html));save(NEW/'comparison/media_validation.json',dict(state='complete',new_families=results,reused_families=reused,new_videos=27,reused_videos=6,scope='Full decode and recorded checksums; human review separate'))
if __name__=='__main__':
    if len(sys.argv)>1:
        assert sys.argv[1]=='--family';publish(next(j for j in jobs() if j['name']==sys.argv[2]))
    else:asyncio.run(main())
