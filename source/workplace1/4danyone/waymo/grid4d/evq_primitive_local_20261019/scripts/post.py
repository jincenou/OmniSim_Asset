import asyncio,json,os,time
from pathlib import Path
N=Path(__file__).resolve().parents[1]
async def call(name,args):
 with (N/'logs'/f'post_{name}.log').open('a') as f:
  p=await asyncio.create_subprocess_exec('/mnt/ojc/miniconda3/envs/grid4d/bin/python',str(N/'scripts'/args[0]),*args[1:],stdout=f,stderr=f);rc=await p.wait();assert rc==0,(name,rc)
async def main():
 while not (N/'pipeline_queue.complete.json').exists():await asyncio.sleep(20)
 assert json.loads((N/'pipeline_queue.complete.json').read_text())['all_passed'],'Pipeline failure requires evidence-based repair'
 await asyncio.gather(call('audit',['audit_completed.py']),call('analyze',['analyze.py']))
 await asyncio.gather(*(call(label,['media.py','--label',label]) for label in ['N06','N11']))
 html=['<!doctype html><meta charset="utf-8"><title>Primitive 局部外观读取</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;background:#181818;color:#eee}video,img{width:100%}a{color:#9df}</style><h1>G0 / G1 / G2 完整终态对照</h1><p>参考为4DAnyone生成视频；数值与视频解码通过不代表质量通过。N06与N11分列。</p><a href="../METRICS.md">指标</a> | <a href="../ASSETS.json">资产</a>']
 for label in ['N06','N11']:
  html.append(f'<h2>{label}</h2>')
  for v in [0,8,16]:html.append(f'<h3>view{v:02d}: reference / G0 / G1 / G2</h3><video controls preload="metadata" src="{label}/ablation_v{v:02d}.mp4"></video><a href="{label}/ablation_v{v:02d}_contact.jpg">首中末图</a><p>历史参照：reference / R1 / A / G2</p><video controls preload="metadata" src="{label}/history_v{v:02d}.mp4"></video>')
  html.append(f'<h3>固定失败窗口</h3><img src="{label}/fixed_bad_window_left.jpg">')
 (N/'comparison/index.html').write_text('\n'.join(html));(N/'post.complete.json').write_text(json.dumps({'numeric_media_audit_complete':True,'visual_review':'pending','at':time.time()},indent=2))
if __name__=='__main__':asyncio.run(main())
