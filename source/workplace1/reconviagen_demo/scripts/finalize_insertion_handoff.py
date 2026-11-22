from pathlib import Path
import sys,json,hashlib,zipfile,concurrent.futures,shutil
import numpy as np
from PIL import Image
import imageio_ffmpeg
R=Path(__file__).resolve().parents[1];S=R/'scene_validation';D=R/'deliveries/insertion_handoff_20261023';sys.path.insert(0,str(R/'tools/report_deps'));import markdown

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
# Full image from one delivered replay vs the same original camera sample.
replay=S/'replay_handoff_H';orig=S/'results/scene-0103/random10_04_white_suv/H'
a=np.asarray(Image.open(replay/'frames/f000.png')).astype(int);b=np.asarray(Image.open(orig/'frames/f000.png')).astype(int);assert a.shape==b.shape
diff=np.abs(a-b);assert diff.max()==0,dict(max=int(diff.max()),mean=float(diff.mean()))
for name in ['color_calibration.json','ground_grid.npz','contact_trajectory.json']:assert sha(replay/name)==sha(orig/name),name
ra=dict(status='PASS',scope='One frame x6 cameras, scene-0103 / random10_04_white_suv / H, through delivered replay.py + packaged asset and trajectory',image_shape=list(a.shape),max_abs_pixel_difference=int(diff.max()),mean_abs_pixel_difference=float(diff.mean()),identical_color_ground_and_contact=True,limitations='Other23 runs were loader/hash/pose-audited, not rerendered in this handoff task; unmodified old composite renderer was not rerendered.')
(D/'REPLAY_VALIDATION.json').write_text(json.dumps(ra,indent=2));shutil.copy2(S/'logs/replay_handoff_H.log',D/'provenance/replay_validation.log')
shutil.copy2(replay/'meta.json',D/'provenance/replay_meta.json')
# Decode all delivered native/context videos, including original full-resolution ones.
paths=[p for folder in ['runs','previews','background_previews'] for p in (D/folder).rglob('*.mp4')]
def check_video(p):
 frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(p));assert frames==60,(p,frames)
 return dict(path=str(p.relative_to(D)),decoded_frames=frames,duration_s=seconds,sha256=sha(p))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:videos=list(ex.map(check_video,paths))
assert len(videos)==33,len(videos)
(D/'VIDEO_VALIDATION.json').write_text(json.dumps(dict(status='PASS',videos=videos,total_decoded_frames=sum(v['decoded_frames'] for v in videos)),indent=2))
# Keep a clear index to the new nonvehicle assets without claiming scene integration.
N=R/'nonvehicle10';nonvehicle=[]
for c in json.loads((N/'inputs/cases.json').read_text()):
 for v in ['B0','R','H','B0_real1']:
  base=N/'results'/c['case_id']/f'{v}_seed42';nonvehicle.append(dict(case_id=c['case_id'],variant=v,ply=str(base/'asset/canonical.ply'),metadata=str(base/'asset/metadata.json'),scene_insertion_status='NOT_RUN',metric_scale_status='normalized units; no validated scene metric scale'))
(D/'NONVEHICLE_STATUS.json').write_text(json.dumps(nonvehicle,indent=2))
css='body{font:16px system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:auto;padding:28px;line-height:1.65}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:9px}video,img{max-width:100%}section{margin:24px 0;padding:20px;background:#f5f7fa}code{background:#edf0f5}a{color:#1c5c9c}'
(D/'README.html').write_text('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown((D/'README.md').read_text(),extensions=['tables','fenced_code']))
runs=json.loads((D/'INSERTIONS.json').read_text())
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>插入资产完整交接</title><style>'+css+'</style><h1>插入资产完整交接</h1><p>3场景 × 2辆车 × 4路线 = 24组实际插入；8份原生高斯资产。每组60帧、12fps、六相机。新增非车辆尚未插入。</p><p><a href="README.html">交接说明与复跑方法</a> · <a href="reports/REPORT.html">实验效果报告</a> · <a href="INSERTIONS.csv">文件索引CSV</a> · <a href="ASSETS.json">资产与尺度</a> · <a href="VALIDATION.json">交接验证</a> · <a href="REPLAY_VALIDATION.json">实际复跑检验</a> · <a href="EXTERNAL_DEPENDENCIES.json">背景/环境依赖</a></p><p>one-frame × six cameras复跑与原始PNG逐像素一致；全部33个场景/对比/背景视频已重新完整解码。仍有悬浮感、接触阴影缺失和偏色，不能视为生产入库验收。</p>'
for sid in ['scene-0103','scene-0104','scene-0277']:
 html+=f'<h2>{sid}</h2><p><a href="scenes/{sid}/cameras.json">实际相机与坐标</a> · <a href="scenes/{sid}/{sid}_plan.json">原轨迹计划</a> · <a href="background_previews/{sid}/insert_splatad_6cam.mp4">无插入背景六相机视频</a></p>'
 for cid in ['random10_04_white_suv','random10_02_white_pickup']:
  html+=f'<section><h3>{cid}</h3><video controls preload="none" src="previews/{sid}/{cid}/comparison_scene.mp4"></video><p><a href="previews/{sid}/{cid}/comparison_frame_030.jpg">中间帧对比图</a></p><table><tr><th>路线</th><th>六相机原视频</th><th>旧格式场景包</th><th>逐相机物体变换</th><th>诊断</th></tr>'
  for r in runs:
   if (r['scene'],r['case_id'])!=(sid,cid):continue
   html+=f'<tr><td>{r["variant"]}</td><td><a href="{r["video"]}">播放</a></td><td><a href="{r["package"]}/scenario.json">scenario.json</a> / <a href="{r["package"]}/trajectories.npz">trajectories.npz</a></td><td><a href="{r["pose_npz"]}">pose NPZ</a></td><td><a href="{r["run"]}/visibility_diagnostics.json">逐帧可见性</a> / <a href="{r["run"]}/meta.json">参数</a></td></tr>'
  html+='</table></section>'
html+='<h2>补充包</h2><p>背景checkpoint和全部无损PNG单独打包；主包可独立播放现有视频。异机复跑仍需要nuScenes数据和对应Python环境。<a href="SUPPLEMENT_ARCHIVES.json">补充包大小与SHA256</a> · <a href="NONVEHICLE_STATUS.json">非车辆尚未插入清单</a></p></html>'
(D/'START_HERE.html').write_text(html)
# Check generated local links before archiving.
from html.parser import HTMLParser
from urllib.parse import unquote
class Links(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v and not v.startswith(('http:','https:','#','data:')):
    assert (D/unquote(v.split('#')[0])).exists(),v
Links().feed(html)
for name in ['finalize_insertion_handoff.py','finalize_handoff_supplements.py']:shutil.copy2(R/'scripts'/name,D/'code'/name)
# Runtime caches from the smoke replay are not handoff payloads.
exclude={'runtime_cache','__pycache__'}
files=[p for p in sorted(D.rglob('*')) if p.is_file() and not any(x in p.relative_to(D).parts for x in exclude) and p.name!='SHA256SUMS.txt']
(D/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(D))+'\n' for p in files));files.append(D/'SHA256SUMS.txt')
archive=D.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
 for p in files:z.write(p,D.name+'/'+str(p.relative_to(D)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
h=sha(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n')
summary=dict(core_archive=str(archive),core_bytes=archive.stat().st_size,core_sha256=h,files=len(files),assets=8,insertions=24,videos_checked=len(videos),replay=ra,supplements=json.loads((D/'SUPPLEMENT_ARCHIVES.json').read_text()))
(R/'docs/insertion_handoff_delivery.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
