"""Human-observed review labels; no model rerun, copying, or automatic quality approval."""
import json, os, sys, hashlib, csv, html
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

D=Path(__file__).resolve().parent
G=D.parent
LABELS={'serious':'明显问题','usable_demo':'抽样质量尚可','pending_review':'待复核'}
DETAIL={
3:('serious','遮挡/分割 → 白色伪影','路杆遮挡在实际输入变成纵向白缺口；生成出现白立片及附着物。骨架无数值告警，不能仅归因于GVHMR。'),
8:('serious','分割主体丢失 + 姿态异常','原观测浅色上衣、腿部在实际输入大量消失；后期主体关联需再查。骨架出现前倾，生成变为深色人物并伴随白色云团。'),
18:('serious','生成跨视角不一致','v12在slot60/80背部出现额外手形；v00仍为手在胸前。生成新增附肢，具体RCP/DiT子阶段尚不能分开定位。'),
42:('serious','分割白洞 + 跨视角不一致','黑色携带物/腰部在输入变白；生成白色挂件、缺口，v00与反向v12均出现正脸。'),
44:('serious','末段输入残缺 → 白色大块','slot60实际输入几乎只剩局部肢体，生成出现遮挡人体的白墙；截断、遮挡与分割的贡献尚未分离。'),
64:('serious','携带物分割不连续','原观测黑伞前段保留、后段被分割移除；生成从黑伞转成白色伞形/大片白幕。骨架未整体崩坏。'),
85:('serious','弱观测 → 骨架深蹲/生成短腿','夜间且被车遮挡，实际输入仅少量上半身；源骨架已持续深屈膝，生成蹲姿/短腿。不可据隐藏腿部断言真实动作，但当前条件不足以支持可靠动作资产。'),
118:('serious','遮挡/分割 → 骨架翻转 → 异常动作','slot30输入仅剩窄条，源骨架已倒转、蜷缩，目标骨架及生成同步异常。slot60头部缺失，生成白块。'),
123:('serious','头部前处理污染 → 生成拖影','原生夜间观测模糊；实际输入头部多时刻破碎/半透明，生成头后黑色大片拖影。尚未分离分割、重采样和生成各自贡献。'),
145:('serious','下半身遮挡 → 骨架深蹲','前30slot输入主要为上半身，骨架呈深蹲/跪姿；40slot后下肢显露，骨架和生成恢复直立。隐藏阶段动作无真值。'),
159:('serious','下半身遮挡 → 骨架深蹲 + 跨视角不一致','后段被前景设施遮挡，输入失去腿部；骨架及生成变为跪蹲/短腿，反向v12仍多次呈正脸。'),
199:('serious','分割误删背包/身体 → 伪影及骨架异常','原黑背包被输入处理成白色；末段输入身体碎裂，骨架深蹲，生成白色拖尾及异常蹲姿。'),
225:('serious','生成白幕/朝向不一致','输入骨架稳定、无数值告警，但v00/v12均有罩住人体的白色幕状物且均出现正脸。具体生成子阶段未定位。'),
56:('serious','跨视角朝向不一致','七时刻v00和相反方向v12均出现正脸及胸前持物，虽单视角流畅，不能作为一致的多视角人体。'),
88:('serious','跨视角朝向不一致','七时刻v00/v12均生成正脸/胸前图案；实际输入主要为背侧观测，背包/携带物也未准确保留。'),
119:('serious','中段分割主体丢失 → 白色人形','首中末概览遗漏了slot80：另一人遮挡时实际输入大部分主体消失，生成出现大片白色人形；部分时刻也有相反视角同时正脸。'),
189:('serious','中段遮挡/分割 → 白色人形','slot10输入下肢被去除，生成旁侧出现白色人形/大块；首中末概览未显示该问题。'),
0:('pending_review','外观待核','姿态较稳定，但未观测背面被推断为浅色外观；缺少背面真值，不直接判定严重错误或通过。'),
1:('pending_review','低光照证据不足','原观测极暗，生成外观清晰不等于身份准确；无法确认生成服装/脸与原人一致。'),
10:('pending_review','携带物颜色错误','主体姿态较稳定，原黑包在实际输入被去掉，生成成白包；需明确人体与携带物验收范围。'),
59:('pending_review','末段遮挡/条件缺失','多数抽样时刻行走轮廓完整，末段植被遮挡后输入只剩小块；不能以多数正常代替完整动作验收。'),
14:('usable_demo','抽样未见重大崩坏','七时刻v00/v12及六视角首中末检查，行走轮廓较完整。首帧有遮挡仍可形成合理形体；未观测正脸、手部细节是生成推断。'),
47:('usable_demo','抽样未见重大崩坏','行走、背包与灰色衣着在抽样前后视图较连贯；脸部纹理和全时间闪烁未作质量保证。'),
153:('usable_demo','抽样未见重大崩坏','行走轮廓及灰上衣/黑裤在抽样视图连贯；背面输入无法验证生成正脸。'),
160:('usable_demo','仅人体抽样质量尚可','人体轮廓与红紫色上衣较连贯；手提袋已被分割去掉，未重建，不能用于要求保留携带物的展示。'),
224:('usable_demo','抽样未见重大崩坏','黑色短袖人物在抽样视角/时刻形体较完整；衣服文字、脸和背面纹理不保证准确。'),
}
OVERVIEW_SERIOUS={
16:'末帧躯干后大片白片',22:'末帧头身连成异常黑色长片',26:'首帧人物旁大片白色漂浮物',
34:'末帧人体旁横向异常片状结构',38:'末帧头后额外轮廓/片状附着物',41:'末帧身体旁大块立片',
51:'首帧身体旁白色幕状物',63:'中帧头身异常黑色方片',67:'头部被长形异常外观覆盖',
69:'头周及身体旁巨大白色伞/片状结构',70:'身体旁白色人形大块',76:'末帧白色大块遮盖人体',
84:'末帧白墙遮盖人体',89:'末帧身体旁巨大黑色片状拖影',97:'头部外出现重叠头形轮廓',
115:'腿部及身体旁巨大白色立片',120:'白色长片附着人体周围',131:'身体旁白色立片',
132:'末帧头身后黑色大矩形',140:'头后及背部漂浮轮廓/片状外观',141:'末帧身体旁大型白色人形结构',
144:'末帧身体旁白色立面',154:'末帧下半身延伸巨大黑色片状物',172:'末帧白色立墙',
198:'末帧人物被大块白色立面覆盖',201:'首末帧人物旁巨大白色立片',204:'末帧白色平面覆盖人物',
226:'首末帧人体旁大块白色结构',
}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()

def save(path,obj):
 temp=path.with_suffix(path.suffix+'.writing')
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 temp.replace(path)

def plan():
 rows=json.loads((D/'overview_index.json').read_text());out=[]
 for r in rows:
  i=r['index'];p=Path(r['output'])
  category,stage,note=DETAIL.get(i,('pending_review','尚未完成逐阶段复核','已查看概览，证据不足以归入质量尚可；需检查完整时序和更多视角。'))
  if i in OVERVIEW_SERIOUS:category,stage,note='serious','仅确认生成异常，未追溯上游',OVERVIEW_SERIOUS[i]
  target=G/'reviewed'/category/p.relative_to(G/'scenes')
  scope='overview triage only'
  if i in DETAIL:
   e=json.loads((D/f'evidence_{i:03d}.json').read_text());e['viewed']=True;e['review_note']=note
   e['viewing_scope']='7 fixed times; original/input/pose/target skeleton00/generated00/generated12; no full-video human review'
   save(D/f'evidence_{i:03d}.json',e)
   scope=e['viewing_scope']
   if i in [0,10,14,47,56,59,88,119,153,160,189,224]:scope+='; also 6 generated views at first/mid/end and 5 native observations'
  out.append(dict(index=i,id=r['id'],category=category,stage=stage,note=note,scope=scope,
   old_output=str(p),output=str(target),package=r['package'],frames=r['frames'],
   overview=r['overview'],evidence=(str(D/f'evidence_{i:03d}.jpg') if i in DETAIL else None),
   numerical_flags=r['flags'],full_temporal_quality_approved=False))
 result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),counts=dict(Counter(r['category'] for r in out)),
  review_policy='Post-hoc human triage, not a preregistered quality benchmark. Occlusion alone is not rejection. Usable means sampled ordinary human demo only; no ground-truth identity or full-time guarantee.',
  overview_scope='227 cases: native middle observation, generated00 first/mid/end and generated12 middle; 26 receive deeper seven-time review.',
  rows=out)
 save(D/'classification.json',result)
 with (D/'classification.csv').open('w',newline='') as f:
  fields=['index','id','category','stage','note','scope','output','old_output','package']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in out)
 print(result['counts'],flush=True)

def page(path,rows,title):
 def rel(p):return html.escape(os.path.relpath(p,path.parent),quote=True)
 parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
 '<title>'+html.escape(title)+'</title><style>body{font:16px system-ui;margin:24px auto;max-width:1200px;background:#16191e;color:#eee}a{color:#8fcaff}article{border:1px solid #555;padding:16px;margin:20px 0}video{width:min(400px,100%)}small{display:block;color:#ccc}nav{line-height:2}img{max-width:100%}</style>',
 '<h1>'+html.escape(title)+'</h1><p>227条生成技术完成。质量分类基于明确记录的抽样范围；待复核不等于通过。输入和骨架仍可查看。仅移动目录，未重新生成视频。</p>',
 '<nav><a href="'+rel(D/'REPORT.md')+'">诊断报告</a> · <a href="'+rel(D/'classification.csv')+'">逐例清单</a> · <a href="'+rel(G/'index.html')+'">全部对象</a></nav><nav>']
 for c,label in LABELS.items():parts.append('<a href="'+rel(G/'reviewed'/c/'index.html')+'">'+label+'</a> · ')
 parts.append('</nav><p>“抽样质量尚可”只表示可作一般人体demo候选，不保证原人外观、携带物、全时序动作或可直接进入4D重建。</p>')
 for r in rows:
  b=Path(r['output']);parts.extend(['<article id="'+html.escape(r['id'])+'"><h2>'+html.escape(r['id'])+' — '+LABELS[r['category']]+'</h2>',
   '<p>'+html.escape(r['note'])+'</p><video controls preload="none" poster="'+rel(b/'contact.jpg')+'" src="'+rel(b/'preview.mp4')+'"></video><p>',
   '<a href="'+rel(b/'preview.mp4')+'">多视角视频</a> · <a href="'+rel(b/'contact.jpg')+'">六视角首中末图</a> · ',
   '<a href="'+rel(b/'input/observed_native.mp4')+'">原始观测</a> · <a href="'+rel(b/'input/inputs/main.mp4')+'">实际输入</a> · ',
   '<a href="'+rel(b/'input/motion_pair/projection_comparison.mp4')+'">骨架对照</a> · <a href="'+rel(b/'videos/00.mp4')+'">v00</a> · <a href="'+rel(b/'videos/12.mp4')+'">v12</a>'])
  if r['evidence']:parts.append(' · <a href="'+rel(r['evidence'])+'">六阶段诊断图</a>')
  parts.append('</p><small>'+html.escape(r['scope'])+'</small></article>')
 parts.append('</html>');path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.html.writing');temp.write_text(''.join(parts));temp.replace(path)

def apply():
 review=json.loads((D/'classification.json').read_text());rows=review['rows']
 assert json.loads((G/'STATUS.json').read_text())['state']=='finished'
 assert not (D/'move_complete.json').exists(),'Already completed; do not run twice'
 inventory=[]
 for r in rows:
  old=Path(r['old_output']);new=Path(r['output'])
  assert old.is_dir() and not old.is_symlink() and not new.exists(),str(old)
  assert len(list((old/'videos').glob('*.mp4')))==24
  files=[];links=[]
  for directory,dirs,names in os.walk(old,followlinks=False):
   for name in dirs+names:
    f=Path(directory)/name
    if f.is_symlink():
     links.append(dict(path=str(f.relative_to(old)),target=os.readlink(f),resolved=str(f.resolve())));assert f.exists()
    elif f.is_file():
     s=f.stat();files.append(dict(path=str(f.relative_to(old)),inode=s.st_ino,device=s.st_dev,size=s.st_size,sha256=digest(f)))
  m=json.loads((old/'manifest.json').read_text());hashes={f['path']:f['sha256'] for f in files}
  assert all(hashes[k]==v for k,v in m['retained'].items()),r['id']
  inventory.append(dict(id=r['id'],old=str(old),new=str(new),files=files,links=links))
 print('Pre-move checks passed',len(inventory),flush=True)
 save(D/'move_inventory.json',dict(classification_sha256=digest(D/'classification.json'),objects=inventory))
 for x in inventory:
  old=Path(x['old']);new=Path(x['new']);new.parent.mkdir(parents=True,exist_ok=True)
  assert old.stat().st_dev==new.parent.stat().st_dev
  old.rename(new)
  try:old.symlink_to(os.path.relpath(new,old.parent),target_is_directory=True)
  except BaseException:
   new.rename(old);raise
  with (D/'move_journal.jsonl').open('a') as f:f.write(json.dumps(dict(id=x['id'],old=str(old),new=str(new)),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 for x in inventory:
  old=Path(x['old']);new=Path(x['new']);assert old.is_symlink() and old.resolve()==new
  for f in x['files']:
   p=new/f['path'];s=p.stat();assert(s.st_ino,s.st_dev,s.st_size)==(f['inode'],f['device'],f['size'])
   assert digest(p)==f['sha256'] and (old/f['path']).stat().st_ino==s.st_ino
  for l in x['links']:
   p=new/l['path'];assert p.is_symlink() and os.readlink(p)==l['target'] and str(p.resolve())==l['resolved'] and p.exists()
 page(G/'index.html',rows,'4DAnyone 227条生成结果：问题诊断与分类')
 for c,label in LABELS.items():page(G/'reviewed'/c/'index.html',[r for r in rows if r['category']==c],label)
 from html.parser import HTMLParser
 class Links(HTMLParser):
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ('href','src','poster'):
     assert (self.base/v).exists(),str(self.base/v)
 for p in [G/'index.html',*(G/'reviewed'/c/'index.html' for c in LABELS)]:
  parser=Links();parser.base=p.parent;parser.feed(p.read_text())
 save(D/'move_complete.json',dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),counts=review['counts'],
  moved_objects=len(rows),regular_files=sum(len(x['files']) for x in inventory),
  bytes_moved=sum(f['size'] for x in inventory for f in x['files']),copied_video_bytes=0,
  checks=['pre/post SHA256 for all regular files','original manifest retained hashes','same inode/device/size','24 videos each','input/motion symlinks unchanged','old paths resolve via directory symlinks','all gallery href/src/poster targets exist']))
 print('Move and gallery verification complete',review['counts'],flush=True)

if __name__=='__main__':
 if sys.argv[1:]==['--plan']:plan()
 elif sys.argv[1:]==['--apply']:apply()
 else:raise SystemExit('Use --plan or --apply')
