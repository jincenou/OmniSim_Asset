"""Local, file-openable review page. Only link verified existing artifacts."""
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import quote,unquote
from datetime import datetime,timezone
from backend_variants import completed_variants

R=Path(__file__).resolve().parents[1];W=R.parent;G=R/'gallery'


def read(path):return json.loads(path.read_text())
def url(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    return quote(os.path.relpath(path,G),safe='/._-')


def main():
    G.mkdir(exist_ok=True);items=[]
    reviews=read(R/'reviews/baseline12.json')['items']
    for a in read(W/'batch100/closeout12.json')['assets']:
        label=a['label'];asset=Path(a['asset']).parent;demo=R/'demos'/label
        job=read(R/'demo_jobs'/f'{label}.json');mapping=read(Path(job['mapping']))
        state=read(demo/'validation.json') if (demo/'validation.json').exists() else None
        videos={};links={'动态高斯 PLY':url(a['asset']),'Checkpoint':url(asset/'ckpts/ckpt_4999.pt'),
            '资产说明':url(asset/'asset.json'),'已有固定视角动作':url(a['preview']),
            '实际输入':url(job['input']),'生成多视角':url(Path(job['source'])/'preview/multiview.mp4')}
        if state:
            for name,title in [('four_panel','四联屏：真实观测 → 输入 → 生成 → 高斯'),('fixed_views','固定相机／时间变化'),('orbit_fixed_time','固定时刻／相机环绕'),('backgrounds','黑／灰／白背景')]:
                videos[title]=url(demo/f'{name}.mp4')
            links.update({'视频验收':url(demo/'validation.json'),'手及附件／脚部放大':url(demo/'details_view00_060.jpg')})
        comparison=R/'comparisons'/label
        if (comparison/'validation.json').is_file():
            read(comparison/'validation.json')
            videos.update({f'生成 / 5k / 15k / 原生15k / 50k 同步对照：view{view}':url(comparison/f'view{view:02d}.mp4') for view in (0,8,16)})
            links['训练预算对照验收']=url(comparison/'validation.json')
        optimization=R/'comparisons'/(label+'_optimization')
        if (optimization/'validation.json').is_file():
            read(optimization/'validation.json')
            videos.update({f'生成 / 15k / 50k / 延长重定位 / 增加密度：view{view}':url(optimization/f'view{view:02d}.mp4') for view in (0,8,16)})
            links['重定位与容量对照验收']=url(optimization/'validation.json')
            links['高密度50k动态PLY']=url(R/'quality'/f'{label}_capacity50k/ckpt_49999.ftgs.ply')
        independent=R/'comparisons'/(label+'_validation')
        if (independent/'validation.json').is_file():
            read(independent/'validation.json')
            videos.update({f'独立轨迹复核：生成 / 5k / 原密度50k / 高密度50k：view{view}':url(independent/f'view{view:02d}.mp4') for view in (0,8,16)})
            links['独立轨迹对照验收']=url(independent/'validation.json')
        metrics=R/'quality'/f'baseline_{label}'/'quality_metrics/metrics.json';m=read(metrics)
        links['39张留出生成图像评测']=url(metrics)
        notes='5k 基线，技术通过；视觉质量未批准。'
        action='待复核';accessory='待复核'
        if label=='phone_full':action='站立／上肢动作';accessory='手机';notes+='已抽样看到脸、手和衣服模糊。'
        if label=='pilot_043_twentyfour':action='站立／上肢动作';accessory='未确认';notes+='已抽样看到轮廓拖影、脸手模糊。'
        if label=='batch_0089_twentyfour':accessory='瓶／背包候选';notes+='附件形状不稳定，重建模糊及漂浮点。'
        review=reviews[label];action=review['action'];accessory=review['accessory'];notes=review['observation']
        links['抽样人工复核']=url(R/'reviews/baseline12.json')
        quality='生成失败对照' if review['issue_category']=='generation_failure' else '待提质／待复核'
        items.append({'id':label,'group':'旧批次12','scene':str(a['source_scene']),'track':a['source_track'],
            'action':action,'accessory':accessory,'quality':quality,'technical':'已通过',
            'notes':notes,'poster':url(demo/'review_contact.jpg') if state else url(asset/'preview/time_views.jpg'),
            'videos':videos,'links':links,'metrics_gray':m['aggregates']['validation_generated']['0.5'],
            'demo_validation':state is not None,'source_mapping':url(job['mapping']),
            'real_frames':mapping.get('real_source_frames'),'interpolated_slots':sum(bool(x['interpolated']) for x in mapping['frames'])})
    walk=R/'reconstruction/outputs/walk063_rootlocal24';d=walk/'demo';proof=read(d/'replay_validation.json')
    items.append({'id':'walk063_rootlocal24','group':'新行走实验','scene':'187','track':read(walk/'asset.json')['source_track'],
        'action':'行走','accessory':'背包候选','quality':'待提质／待复核','technical':'已通过',
        'notes':'先分离水平根位移再生成；保留垂直起伏和关节动作。外部原轨迹和合成新轨迹回放数值通过，未完成道路尺度／地面配准；仍有明显模糊和漂浮点。',
        'poster':url(d/'review_contact.jpg'),'videos':{title:url(d/f'{name}.mp4') for name,title in [('four_panel','四联屏'),('fixed_views','固定相机动作'),('orbit_fixed_time','固定时刻环绕'),('backgrounds','三种背景'),('replay_original','恢复原根轨迹'),('replay_new','合成新轨迹')]},
        'links':{'动态高斯 PLY':url(walk/'ckpt_4999.ftgs.ply'),'Checkpoint':url(walk/'ckpts/ckpt_4999.pt'),'资产说明':url(walk/'asset.json'),'轨迹验收':url(d/'replay_validation.json'),'定量评测':url(walk/'quality_metrics/metrics.json')},'demo_validation':True})
    detail=d/'details/validation.json'
    if detail.is_file():
        items[-1]['links']['行走手脚与附件放大']=url(d/'details/view00_frame060.jpg')
        items[-1]['links']['细节预览来源']=url(detail)
    comparison=R/'comparisons/walk063_rootlocal24_validation'
    if (comparison/'validation.json').is_file():
        items[-1]['videos']['5k / 50k / 增加容量动作对照']=url(comparison/'view08.mp4')
        items[-1]['links']['动作优化对照验收']=url(comparison/'validation.json')
    for label,action,accessory in [('batch_0007_rootlocal24','站立到迈步','未确认'),('batch_0021_rootlocal24','行走／转身','未确认'),('batch_0024_rootlocal24','行走','未确认'),('r07_short61_24','行走／短片段','手提袋'),('m02_short61_24','行走／持续移动自车','肩包／围巾')]:
        asset=R/'reconstruction/outputs'/label;demo=R/'demos'/label
        info=read(asset/'asset.json');validation=read(demo/'validation.json');job=validation['job']
        review=read(R/'reviews'/f'{label}_asset.json');metrics=asset/'quality_metrics/metrics.json'
        m=read(metrics)
        if m['ply_roundtrip']['render_max_abs']>1e-5:raise ValueError('PLY render validation failed')
        items.append({'id':label,'group':'新增行走实验','scene':info['source_scene'],'track':info['source_track'],
            'action':action,'accessory':accessory,'quality':'待提质／待复核','technical':'已通过',
            'notes':' '.join(review.get('observations',review.get('findings',[]))),
            'poster':url(demo/'review_contact.jpg'),
            'videos':{title:url(demo/f'{name}.mp4') for name,title in [('four_panel','四联屏'),('fixed_views','固定相机动作'),('orbit_fixed_time','固定时刻环绕'),('backgrounds','三种背景')]},
            'links':{'实际输入':url(job['input']),'输入时间来源':url(job['mapping']),
                '生成多视角':url(Path(job['source'])/'preview/multiview.mp4'),
                '动态高斯 PLY':url(asset/info['asset']),'Checkpoint':url(asset/info['checkpoint']),
                '资产说明':url(asset/'asset.json'),'视频验收':url(demo/'validation.json'),
                '生成观测拟合指标':url(metrics),'抽样复核':url(R/'reviews'/f'{label}_asset.json')},
            'metrics_gray':m['aggregates']['validation_generated']['0.5'],'demo_validation':True})
    for item in items:
        if item['id']=='r07_short61_24':
            diagnostic=R/'diagnostics/weak_velocity'
            item['videos']['15k零速度sigma8四联屏（仍模糊）']=url(diagnostic/'zero_sigma8/demo/four_panel.mp4')
            item['videos']['四组初始化配置动作对照']=url(diagnostic/'comparison_sigma8/view08.mp4')
            item['videos']['直接优化前后：生成 / 原5k / 普通15k / sigma8的15k']=url(diagnostic/'before_after/view08.mp4')
            item['links']['初始化对照指标']=url(diagnostic/'REPORT.md')
            item['links']['15k对照资产说明']=url(diagnostic/'zero_sigma8/asset.json')
        if item['id']=='m02_short61_24':
            item['quality']='重建视觉失败'
            diagnostic=R/'diagnostics/m02_duration'
            item['videos']['直接优化前后：生成 / 原5k / 普通15k / sigma8的15k']=url(diagnostic/'comparison/view00.mp4')
            for variant in ('sigma15','sigma8'):
                item['videos'][f'15k {variant}四联屏（脚部仍淡化）']=url(diagnostic/variant/'demo/four_panel.mp4')
                item['links'][f'15k {variant}动态PLY']=url(diagnostic/variant/'ckpt_14999.ftgs.ply')
            item['links']['初始时间宽度数值与覆盖统计']=url(diagnostic/'results.json')
            item['links']['初始时间宽度抽样视觉复核']=url(diagnostic/'comparison/review.json')
        comparison=R/'comparisons'/f'{item["id"]}_temporal'
        if (comparison/'validation.json').is_file():
            item['videos']['时间配置对照：生成 / dense50k / 延长重分配 / duration8']=url(comparison/'view08.mp4')
            item['links']['时间配置指标']=url(R/'benchmark/TEMPORAL_REPORT.md')
    for name in ('green_walk','green_portrait'):
        root=W/'experiments'/name;video=root/'outputs/smoke/preview/multiview.mp4';poster=root/'outputs/smoke/preview/time_views.jpg'
        mapping=read(root/'inputs/frame_mapping.json')
        items.append({'id':name,'group':'保留失败','scene':str(mapping['scene']),'track':mapping['track_id'],
            'action':'行走分支／待复核','accessory':'未确认','quality':'生成失败对照','technical':'未批准重建',
            'notes':'历史构图／衣着保持失败，保留原始生成结果；不计入12个资产，不宣称增加高斯训练能修复。',
            'poster':url(poster),'videos':{'失败生成视频':url(video)},'links':{'实际输入':url(root/'inputs/main.mp4'),'输入来源':url(root/'inputs/frame_mapping.json')},'demo_validation':False})
    for item in items:
        versions=completed_variants(item['id'])
        item['backend_metrics']=[]
        for version in versions:
            root=version['root'];title=version['title']
            item['backend_metrics'].append({'run':version['run'],'title':title,**version['metrics']['mean']['0.5']})
            item['links'][title+' 动态包说明']=url(root/'asset.json')
            item['links'][title+' 指标及重放']=url(root/'evaluation/results.json')
            if (root/'ftgs_comparison/validation.json').is_file():
                item['videos'][title+' 与FTGS原始/优化版本同步对照']=url(root/'ftgs_comparison/view08.mp4')
            if (root/'temporal_comparison/results.json').is_file():
                item['videos'][title+' 生成/默认/时间设置同步对照']=url(root/'temporal_comparison/view08.mp4')
                item['links'][title+' 同预算参数与指标']=url(root/'temporal_comparison/results.json')
            if version['validated_demo']:
                for name,description in [('four_panel','真实→输入→生成→高斯'),('fixed_views','固定视角动作'),('orbit_fixed_time','固定时刻几何'),('backgrounds','黑灰白背景')]:
                    item['videos'][title+' '+description]=url(root/'demo'/f'{name}.mp4')
                item['links'][title+' 视频与时间验收']=url(root/'demo/validation.json')
        if versions:item['notes']+=' 已增加4DGaussians候选后端的指标和展示；不同版本不增加人物轨迹数，仍未生产批准。'
        prefixes={'m02_short61_24':'m02_61','r07_short61_24':'r07_61'}
        if item['id'] in prefixes:
            pair=R/'alternatives/4dgaussians/comparisons'/(prefixes[item['id']]+'_alpha_l1')
            if (pair/'results.json').is_file():
                item['videos']['同预算：默认RGB / 增加alphaL1直接对照（含负结果）']=url(pair/'view08.mp4')
                item['links']['alpha单因素数值与参数审计']=url(pair/'results.json')
        if versions:
            item['links']['新后端最新六条轨迹和预算结论']=url(R/'research/short_backend_experiments.md')
            item['links']['完整指标含SSIM']=url(R/'research/latest_metrics.md')
            item['links']['行走轮廓一致性诊断']=url(R/'research/hull_consistency.md')
            item['links']['同预算时间网格与平滑对照']=url(R/'research/temporal_controls.md')
    manifest={'updated_at':datetime.now(timezone.utc).isoformat(),'baseline_assets':12,'new_walking_assets':6,
        'technical_assets':18,'visual_approved_count':0,'count_scope':'12 baseline distinct tracks plus6 separate root-local walking/action tracks including61-frame R07 and M02; failure variants not counted as additional approved assets',
        'private_local_only':True,'items':items}
    (R/'gallery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for item in items:
        esc=html.escape
        videos=''.join(f'<details><summary>{esc(title)}</summary><video controls preload="none" src="{esc(src)}"></video><a href="{esc(src)}">打开视频</a></details>' for title,src in item['videos'].items())
        links=' · '.join(f'<a href="{esc(src)}">{esc(title)}</a>' for title,src in item['links'].items())
        metric=item.get('metrics_gray');metrics='' if not metric else f"<p>灰背景留出评测：前景 PSNR {metric['foreground_psnr']:.2f} · LPIPS {metric['crop_lpips']:.4f} · 外部 alpha {metric['outside_alpha_mean']:.4f}</p>"
        if item['backend_metrics']:
            metrics+='<table><tr><th>候选后端</th><th>前景PSNR↑</th><th>LPIPS↓</th><th>IoU↑</th></tr>'+''.join(f'<tr><td>{esc(row["title"])}</td><td>{row["foreground_psnr"]:.3f}</td><td>{row["crop_lpips"]:.4f}</td><td>{row["silhouette_iou"]:.4f}</td></tr>' for row in item['backend_metrics'])+'</table>'
        attrs=' '.join(f'data-{key}="{esc(item[key],quote=True)}"' for key in ('action','accessory','scene','quality','group'))
        cards.append(f'<article {attrs}><h2>{esc(item["id"])}</h2><p class="tag">{esc(item["group"])} · 场景 {esc(item["scene"])} · {esc(item["action"])}</p><p>技术：{esc(item["technical"])}　视觉：{esc(item["quality"])}</p><p>{esc(item["notes"])}</p><img loading="lazy" src="{esc(item["poster"])}" alt="{esc(item["id"])} 时序与多视角复核图">{metrics}{videos}<p class="links">{links}</p></article>')
    filters=''.join(f'<label>{title}<select id="{key}"><option value="">全部</option>'+''.join(f'<option>{html.escape(value)}</option>' for value in sorted({item[key] for item in items}))+ '</select></label>' for key,title in [('group','分组'),('action','动作'),('accessory','附件'),('scene','场景'),('quality','质量')])
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Waymo · 4DAnyone 动态资产复核</title>
<style>body{margin:0;background:#11171d;color:#e3eaf0;font:16px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:20px}a{color:#91ccff}header p{max-width:1000px}.filters{display:flex;flex-wrap:wrap;gap:16px;padding:18px 0}label{display:flex;gap:8px;align-items:center}select{padding:8px;background:#23313e;color:white;border:1px solid #526576;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:20px}article{background:#1b2630;border:1px solid #344654;border-radius:10px;padding:20px}article[hidden]{display:none}article img{display:block;max-width:100%;max-height:580px;margin:auto;object-fit:contain;background:#111}video{display:block;width:100%;max-height:650px;background:black}details{padding:12px 0;border-bottom:1px solid #344654}summary{cursor:pointer}.tag{color:#9ed7bc}.links{font-size:14px}footer{margin-top:30px;color:#becbd5}</style>
<main><header><h1>Waymo → 4DAnyone → 动态高斯</h1><p>旧批次12条＋6条新增根位移分离的行走/动作轨迹，共18条技术完成资产，包含25张实拍映射为61帧的R07、M02短片段。M02的高斯有明显腿脚缺失，已标为重建视觉失败。技术完成与视觉质量分开记录，当前没有批准为最终高质量的资产。页面保留失败结果，不滤除模型漂浮点。</p><p>四联屏只有“生成—高斯”共享相机。源视频含真实帧和插值帧；固定时刻环绕用于看几何，不代表人物动作。根轨迹仍在估计坐标内，尚不能直接当作 Waymo 世界坐标。</p><p><a href="../REPORT.md">报告</a> · <a href="../gallery_manifest.json">完整清单</a> · <a href="../native_results.json">同分辨率质量对照</a> · <a href="../moving_camera/index.html">新行走与相机对照</a> · <a href="../diagnostics/single_time/REPORT.md">单时刻诊断</a> · <a href="../diagnostics/time_budget/REPORT.md">等预算时间范围对照</a></p></header>
'''+f'<div class="filters">{filters}</div><p id="count" aria-live="polite"></p><section class="cards">'+''.join(cards)+'''</section><footer>本地文件页面，无外部资源、遥测、上传或公网服务。视频按需加载。未标注动作的样本仍待逐条复核。</footer></main>
<script>const fields=['group','action','accessory','scene','quality'];const cards=[...document.querySelectorAll('article')];function applyFilters(){let count=0;cards.forEach(card=>{card.hidden=!fields.every(field=>!document.getElementById(field).value||card.dataset[field]===document.getElementById(field).value);if(!card.hidden)count++;});document.getElementById('count').textContent=`显示 ${count} / ${cards.length} 项（失败分支不计入资产完成数）`;}fields.forEach(field=>document.getElementById(field).addEventListener('change',applyFilters));applyFilters();</script></html>'''
    (G/'index.html').write_text(text)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.articles=0
        def handle_starttag(self,tag,attrs):
            self.articles+=tag=='article'
            self.links.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();parser.feed(text);assert parser.articles==len(items)
    for link in parser.links:
        if not (G/unquote(link)).is_file():raise FileNotFoundError(link)
    (G/'validation.json').write_text(json.dumps({'state':'all_local_links_exist','articles':parser.articles,'links':len(parser.links),'browser_visual_check':'not yet available','video_decode':'see per-item renderer validation; failure source previews retained as historical evidence'},indent=2)+'\n')
    print(json.dumps({'gallery':str(G/'index.html'),'items':len(items),'demo_packages_ready':sum(x['demo_validation'] for x in items)}))


if __name__=='__main__':main()
