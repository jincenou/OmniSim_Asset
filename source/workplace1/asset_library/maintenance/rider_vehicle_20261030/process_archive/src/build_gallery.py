import json,html,base64,mimetypes,re
from pathlib import Path
L=Path('/mnt/ojc/workplace1/asset_library');O=L/'previews/rider_vehicle_demos';O.mkdir(parents=True,exist_ok=True)
labels={'nuscenes_b2710614':'黄色外衣骑乘者 · 灰色踏板车（非电动）','niu_blue':'绿色外衣骑乘者 · 蓝色电动踏板车','niu_black':'黑色外衣骑乘者 · 黑色电动踏板车'}
cards=[];manifest=[]
for name,label in labels.items():
 folder={'nuscenes_b2710614':'H_rider_nuscenes_b2710614','niu_blue':'B0_rider_niu_blue','niu_black':'H_rider_niu_black'}[name]
 p=L/'rigid/motorcycle'/folder/'placement.json'
 asset_url='../../rigid/motorcycle/'+folder
 if not p.exists():continue
 a=json.loads(p.read_text());q=json.loads((p.parent/'quality.json').read_text());manifest.append(a)
 ins=''.join(f'<div><h3>{html.escape(mode)} · 真实 SplatAD 场景</h3><video controls loop preload="metadata" src="{asset_url}/{path}"></video></div>' for mode,path in a['insertion'].items())
 cards.append(f'''<article><h2>{label}</h2><p>{q['ply_validation']['gaussian_count']:,} Gaussians · 单个人车整体 · +X 前进 / +Z 向上</p><video controls loop autoplay muted playsinline src="{asset_url}/turntable.mp4"></video><details><summary>16 视角与生成源图</summary><img src="{asset_url}/evidence.jpg"><img class="source" src="{asset_url}/source/anchor.png"><img class="source" src="{asset_url}/source/flux.png"></details><p><a href="{asset_url}/asset.ply">Gaussian PLY</a> · <a href="{asset_url}/asset.json">加载与尺寸配置</a> · <a href="{asset_url}/quality.json">质检及限制</a> · <a href="{asset_url}/provenance.json">来源</a></p>{ins}</article>''')
(O/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>固定骑姿人车组合 · 3D Gaussian demos</title><style>body{font:16px/1.65 system-ui;margin:32px auto;max-width:1280px;padding:0 24px;background:#101820;color:#e7edf4}h1{font-size:30px}article{background:#1b2835;border-radius:14px;padding:24px;margin:24px 0}video{max-width:100%;width:760px;background:#333;border-radius:8px}img{max-width:100%}.source{max-width:45%;vertical-align:top;margin:1%}a{color:#87d9ed}summary{cursor:pointer;padding:14px 0}.note{color:#bdcad5}</style><h1>固定骑姿人车组合 · 3D Gaussian demos</h1><p class="note">3 个不同资产；模型版本不重复计数。2 个电动踏板车组合，1 个 nuScenes 非电动对照。人、车作为一个刚体整体移动；车轮、车把和肢体不独立运动。</p><p class="note">已观测衣着、车辆和固定接触关系用于约束生成。背面、面部微细节和文字含生成推断；尺寸含估计误差。公开视频素材未附明确复用许可，本包用于内部研究展示。</p>'''+''.join(cards)+'<p><a href="../../maintenance/rider_vehicle_20261030/REPORT.md">工程报告</a> · <a href="../../rigid/asset_library_rigid.json">资产清单</a></p></html>')
# Inline presentation media so downloading this single file is sufficient.
p=O/'index.html'
s=p.read_text()
def data_uri(path):
    mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
def inline_src(match):
    return 'src="'+data_uri((O/match.group(1)).resolve())+'"'
s=re.sub(r'src="([^"]+)"',inline_src,s)
def inline_link(match):
    target,label=match.groups()
    path=(O/target).resolve()
    if path.suffix=='.ply':
        return '<span title="'+html.escape(str(path),quote=True)+'">Gaussian PLY（保存在服务器资产目录）</span>'
    if target=='../../rigid/asset_library_rigid.json':
        payload=json.dumps(dict(asset_count=len(manifest),assets=manifest),ensure_ascii=False,indent=2).encode()
        uri='data:application/json;base64,'+base64.b64encode(payload).decode('ascii')
        filename='rider_vehicle_demo_manifest.json'
    else:
        uri=data_uri(path)
        filename=path.parent.name+'_'+path.name if path.suffix=='.json' else path.name
    return '<a href="'+uri+'" download="'+html.escape(filename,quote=True)+'">'+label+'</a>'
s=re.sub(r'<a href="([^"]+)">(.*?)</a>',inline_link,s)
s=s.replace('</h1>','</h1><p class="note">单文件离线预览：下载本 HTML 后，用 Chrome、Edge 或 Firefox 打开即可观看环绕与实景插入视频。视频可点击播放，16 视角图片可展开查看。</p>',1)
p.write_text(s)
print(p, 'standalone bytes:', p.stat().st_size)
