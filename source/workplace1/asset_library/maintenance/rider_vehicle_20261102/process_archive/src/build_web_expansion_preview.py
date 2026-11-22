from pathlib import Path
import json,base64,html,io,subprocess,shutil,hashlib
from PIL import Image,ImageOps
W=Path(__file__).resolve().parents[1];root=W.parents[2];src=W/'sources/web_expansion_20261103'
def img(path):
 p=Path(path)
 if not p.exists():return '<p>尚未产出</p>'
 im=ImageOps.contain(Image.open(p).convert('RGB'),(600,600));b=io.BytesIO();im.save(b,format='JPEG',quality=86)
 return '<img src="data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
def video(path):
 p=Path(path)
 if not p.exists():return ''
 return '<video controls loop muted preload="metadata" style="width:100%;max-height:520px" src="data:video/mp4;base64,'+base64.b64encode(p.read_bytes()).decode()+'"></video>'
cards=[];records=[]
for q in json.loads((src/'queue.json').read_text()):
 name=q['name'];run=W/'runs'/name;source=json.loads((W/'sources'/name/'source.json').read_text());f=run/'F/generation.json';s=json.loads(f.read_text())['assets'][0] if f.exists() else {};flux=s.get('stages',{}).get('flux',{});images=flux.get('images',[]);h=run/'H0/generation.json';hs=json.loads(h.read_text())['assets'][0] if h.exists() else {};recon=hs.get('stages',{}).get('H',{}).get('status','not_started')
 status='重建已完成，待环绕质检' if recon=='succeeded' else ('标准图已生成，重建未完成' if images else '标准图生成中')
 if (run/'selected/load_validation.json').exists():status='已通过环绕结构初检与加载验证，待场景插入验收'
 dest=root/'rigid/motorcycle'/('rider_web_'+name);quality=json.loads((dest/'quality.json').read_text()) if (dest/'quality.json').exists() else {};accepted=quality.get('status')=='demo_accepted'
 if accepted:status='已归档，通过内部 demo 验收'
 cards.append('<article><h2>'+html.escape(name)+'</h2><p>'+status+' · 辅助真实图 0 张</p><a href="'+html.escape(source['page'],quote=True)+'">来源页面</a><div class="pair"><figure>'+img(W/'packets'/name/'anchor.jpg')+'<figcaption>源图整体前景</figcaption></figure><figure>'+(img(images[0]['path']) if images else '<p>生成中</p>')+'<figcaption>FLUX 标准图</figcaption></figure></div></article>')
 orbit=run/'selected/render/turntable.mp4'
 if not orbit.exists():orbit=run/'H0/curation/axis_render/turntable.mp4'
 if orbit.exists():cards[-1]=cards[-1].replace('</article>','<h3>3D 环绕预览</h3>'+video(orbit)+'</article>')
 ins=run/'insertion_near_follow';mp=ins/'meta.json';flag=run/'insert_near_follow.exit'
 if mp.exists() and flag.exists() and flag.read_text().strip()=='0':
  meta=json.loads(mp.read_text());assert meta['asset_sha256']==hashlib.sha256((run/'selected/canonical.ply').read_bytes()).hexdigest()
  front=ins/'front_preview.mp4'
  if not front.exists():
   temp=ins/'front_preview.tmp.mp4'
   subprocess.run(['/mnt/ojc/miniconda3/envs/splatad/bin/ffmpeg','-v','error','-y','-i',str(ins/'insert_splatad_6cam.mp4'),'-vf','crop=iw/3:ih/2:iw/3:0','-c:v','libx264','-crf','20','-preset','fast','-an',str(temp)],check=True)
   temp.replace(front)
  if (dest/'package.json').exists():
   target=dest/'insertion/near_follow/front_preview.mp4';shutil.copy2(front,target)
   p=dest/'package.json';pack=json.loads(p.read_text());pack['files']=sorted(str(f.relative_to(dest)) for f in dest.rglob('*') if f.is_file() and f!=p);p.write_text(json.dumps(pack,indent=2))
  cards[-1]=cards[-1].replace('</article>','<h3>短轨迹插入：前方相机</h3>'+video(front)+'</article>')
 records.append(dict(identity=name,flux=flux.get('status','pending'),H=recon,accepted=accepted,package=str(dest.relative_to(root)) if (dest/'package.json').exists() else None))
out=root/'previews/web_rider_expansion_20261103';out.mkdir(exist_ok=True)
summary=f"本批 {len(records)} 个，{sum(r['accepted'] for r in records)} 个通过内部 demo 验收。单张清晰源图，0 张辅助真实图。人车姿态固定，车轮和肢体不单独运动。背面细节及面部由模型补全，未拟合投影阴影；不等同于精确扫描或生产级写实资产。全部图片与视频已内嵌，可单独下载离线查看。"
(out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>网络电动车骑乘者批量生成</title><style>body{font-family:system-ui;background:#eef2f5;margin:30px;color:#17202a}main{max-width:1250px;margin:auto}article{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:flex;gap:20px}figure{width:48%;margin:10px 0}img{width:100%;height:480px;object-fit:contain}h2{font-size:19px}</style><main><h1>新增网络电动车＋固定骑乘者</h1><p>'+summary+'</p>'+''.join(cards)+'</main></html>')
(out/'status.json').write_text(json.dumps(records,indent=2));print(out/'index.html')
