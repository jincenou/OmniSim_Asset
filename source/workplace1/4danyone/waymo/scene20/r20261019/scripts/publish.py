"""Only completed results enter gallery. Compact failed scratch with evidence."""
import asyncio,json,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def compact_failure(b,job):
 if job.get('stage') not in ['camera','crop','interpolate','seg_prepare','segment','composite']:return
 w=b/'.work'
 if not w.exists() or (b/'failure.json').exists():return
 src=w/'source';keep=b/'input';keep.mkdir(exist_ok=True)
 for rel,target in [('native_contact.jpg','native_contact.jpg'),('inputs/main.mp4','model_input.mp4'),('inputs/frame_mapping.json','frame_mapping.json'),('native.json','native.json'),('segmentation/target_masks.npz','target_masks.npz'),('research/batch_gate_mask.json','mask_gate.json')]:
  p=src/rel
  if p.is_file():shutil.copy2(p,keep/target)
 (b/'failure.json').write_text(json.dumps({'job':job,'no_generated_result_claimed':True,'retained_input_and_gate_evidence':True,'discarded':'this attempt decoded crop/SAM frames and duplicate temporary video','original_data_modified':False},indent=2));shutil.rmtree(w)
def publish(state):
 rows=json.loads((R/'selection_manifest.json').read_text())['candidates'];complete=[];failed=[]
 if (R/'reserve_admitted.json').exists():
  admitted=set(json.loads((R/'reserve_admitted.json').read_text())['ids']);rows += [r for r in json.loads((R/'reserve_manifest.json').read_text())['candidates'] if r['sample_id'] in admitted]
 html=['<!doctype html><meta charset="utf-8"><title>Waymo 20场景 多视角视频</title><style>body{background:#181818;color:#eee;font-family:sans-serif;max-width:1200px;margin:auto}video{max-width:100%;max-height:720px}a{color:#9df}</style><h1>Waymo → 4DAnyone Turbo</h1><p>20个新身份，按场景组织。仅展示完整24视角并通过逐帧解码的结果；视觉质量仍需逐人检查。</p>']
 for row in rows:
  key=row['sample_id'];b=R/'scenes'/row['scene']/key;job=state['jobs'].get(key,{})
  if job.get('state')=='failed':compact_failure(b,job);failed.append(key)
  if job.get('state')!='complete':continue
  assert (b/'manifest.json').exists();complete.append(key);rel=b.relative_to(R)
  html.append(f'<h2>scene {row["scene"]} / {key}</h2><p>输入原生帧 {row["native_frame_count"]}；生成 {row["output_slots"]} 帧 ×24视角。</p><a href="{rel}/input/observed_native.mp4">真实观测裁剪</a> | <a href="{rel}/input/model_input.mp4">模型输入</a> | <a href="{rel}/contact.jpg">关键帧总览</a> | <a href="{rel}/manifest.json">来源记录</a><br><video controls preload="metadata" src="{rel}/preview.mp4"></video>')
 html.append(f'<p>当前完整结果{len(complete)}/20；前处理或执行失败{len(failed)}：{", ".join(failed)}。失败保留，未冒充成功。</p>');p=R/'index.writing';p.write_text('\n'.join(html));p.replace(R/'index.html')
 report=f'''# Waymo 20场景：4DAnyone 多视角视频

当前完成 **{len(complete)}/20**，失败 **{len(failed)}**。后台状态以 [BATCH_STATUS.json](BATCH_STATUS.json) 为准。[完整视频入口](index.html)。

使用公开冻结权重推理，不训练4DAnyone、不新增本批高斯后端训练。每人24视角，704×1280、25fps；25/49个原生观测经已验证插值链路对应61/121帧，不复制末帧填充。根据真实相机参数处理运动相机，人体动作由GVHMR估计；不保证真实世界轨迹或隐藏面的准确性。

路径为 `scenes/场景/Pxx/`：`input/`保留真实观测和实际输入；`videos/`保留24个原始视频；`preview.mp4`为六视角同步预览；`contact.jpg`为关键帧；`cameras.json`、`root_motion.npz`、`motion/`、`skeletons/`供后续重建复用；`manifest.json`和`validation.json`记录来源与验收。`.work`只在执行期间存在，成功或失败留证后清理临时图片与重复缓存。

20个身份来自20个场景，按固定种子从此前输入合格池选取，排除旧Round2/Random8等已用身份。人体是否行走仍需观察，不能用车辆位移代替行为标注。失败不静默换人、不放宽mask门槛。已确认前处理失败后，最多64条预先排序的其他场景输入备用轨迹（仅合格输入才进入生成），仅补足前处理硬失败至20个成功视频，扣除尚在运行的名额后提前使用空闲GPU；原始候选与备用清单分别保留。

后台 `tmux -L fdscene20 list-sessions`；与Grid4D六组实验共用GPU0–7锁。空闲卡领取下一人，其他项目进程不被终止。磁盘不足时等待而不覆盖旧成果。

完整：{', '.join(complete) or '尚无'}。
失败：{', '.join(failed) or '无'}。
'''
 (R/'REPORT.md').write_text(report)
async def main():
 while True:
  if (R/'STATUS.json').exists():
   state=json.loads((R/'STATUS.json').read_text())
   if (R/'RESERVE_STATUS.json').exists():
    reserve=json.loads((R/'RESERVE_STATUS.json').read_text());state['jobs'].update(reserve['jobs'])
    if reserve['state']=='running':state['state']='running'
   if (R/'reserve_pending.flag').exists():state['state']='running'
   state['completed']=sum(j.get('state')=='complete' for j in state['jobs'].values());state['attempted']=len(state['jobs']);state['target']=20
   temp=R/'BATCH_STATUS.writing';temp.write_text(json.dumps(state,indent=2));temp.replace(R/'BATCH_STATUS.json');publish(state)
   if state['state']!='running' and not (R/'reserve_pending.flag').exists():break
  await asyncio.sleep(30)
if __name__=='__main__':asyncio.run(main())
