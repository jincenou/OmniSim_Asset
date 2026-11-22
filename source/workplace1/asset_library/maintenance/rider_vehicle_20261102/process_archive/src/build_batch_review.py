"""Standalone offline review: embeds all media, no server dependencies."""
from pathlib import Path
import base64,json,html,hashlib
W=Path(__file__).resolve().parents[1];root=W.parents[2];out=root/'previews/rider_vehicle_batch_20261103';out.mkdir(exist_ok=True)
def uri(p,mime):return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>人车组合批次审核</title><style>body{max-width:1100px;margin:24px auto;padding:12px;background:#16212c;color:#eee;font:16px sans-serif}img,video{max-width:100%;max-height:750px}article{padding:20px;margin:20px 0;background:#243443}p{line-height:1.6}a{color:#a8dcff}</style><h1>人车整体 Gaussian：本批审核</h1><p>这是候选审核页，不是验收通过清单。每个人车组合只计一个身份。姿态固定，车轮、车把和肢体不单独运动。全部媒体内嵌，下载此 HTML 可离线查看。</p>']
for name,label in [('unu','unu'),('gogoro','Gogoro'),('govecs','Govecs'),('bicycle_nusc_780be296','自行车'),('niu_nqi_leather','NIU 黑色皮衣骑乘者'),('niu_nqi_light','NIU 浅色上衣骑乘者')]:
 r=W/'runs'/name;selection=json.loads((r/'selection_review.json').read_text());method=selection.get('selected_method') or json.loads((r/'selected/selected.json').read_text())['method'];state=selection.get('status','pending_review')
 parts.append('<article><h2>'+label+'</h2><p>候选方法：'+html.escape(method)+'；状态：'+html.escape(state)+'</p>')
 limits=selection.get('limitations',[])
 if selection.get('scale_concern'):limits=limits+[selection['scale_concern']]
 if limits:parts.append('<p>'+html.escape(' / '.join(limits))+'</p>')
 sheet=r/'selected/render/evidence.jpg'
 if not sheet.exists():sheet=r/method/'curation/axis_render/evidence.jpg'
 if sheet.exists():parts.append('<img alt="环绕审核图" src="'+uri(sheet,'image/jpeg')+'">')
 video=sheet.with_name('turntable.mp4')
 if video.exists():parts.append('<video controls preload="none" src="'+uri(video,'video/mp4')+'"></video>')
 for ins in sorted(r.glob('insertion_*')):
  v=ins/'insert_splatad_6cam.mp4';meta=ins/'meta.json'
  if v.exists() and meta.exists():
   current_hash=hashlib.sha256((r/'selected/canonical.ply').read_bytes()).hexdigest()
   version_note='（旧候选版本，不能用于当前模型验收）' if json.loads(meta.read_text()).get('asset_sha256')!=current_hash else ''
   review_path=ins/'visual_review.json'
   if review_path.exists():version_note+=' ['+json.loads(review_path.read_text()).get('status','unreviewed')+']'
   parts.append('<h3>'+html.escape(ins.name+version_note)+'</h3><video controls preload="none" src="'+uri(v,'video/mp4')+'"></video>')
 parts.append('</article>')
parts.append('</html>');p=out/'index.html';tmp=out/'index.html.tmp';tmp.write_text(''.join(parts));tmp.replace(p);print(p,p.stat().st_size)
