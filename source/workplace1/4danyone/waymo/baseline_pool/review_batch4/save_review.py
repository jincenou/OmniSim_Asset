import json, hashlib, datetime
from pathlib import Path
P=Path(__file__).resolve().parent
selection=json.loads((P/'selection.json').read_text())
notes={
'D7ca160e7d8293f':(True,'retain_input_candidate','25帧原生观测为车旁站立、轻微转身及小幅挪步；61帧输入/骨架未见倒转、深蹲塌缩或明显身份跳换。首段脚旁残留车体灰块，前臂轻微残影；末段脚投影与可见脚存在偏差。保留适中门槛输入候选，不保证隐藏面、手部或精确站立抖动。'),
'Dbd14079ba4b4b9':(False,'hold_window_foreground_identity_failure','原生裁剪中红/绿衣行人仍可见，实际输入却先保留前方深色上衣行人，随其向右离开，slot9–60没有大于0.5的前景mask像素，画面近空；GVHMR仍输出近静止骨架，不能证明动作可靠。分割身份/前景传播出现严重问题；预测轨迹自身是否混轨尚未独立验证，不能只归因遮挡或直接淘汰整个人。当前窗口不作为可用输入。'),
'T375eb7ec80e4d6':(True,'retain_input_candidate','25原生帧显示背身行走；61帧输入及骨架总体保持交替迈步，未见翻转、深蹲塌缩或混人。末段约slot53–60腿部有灰色拖影/扩张，原观测也有车辆遮挡，局部脚与手对齐不精确。保留适中门槛输入候选；隐藏关节、精细脚步和未见正面不作真实性保证。'),
'Tacd375b536bad0':(False,'hold_window_occlusion_pose_support_insufficient','人群交叉，实际输入中同一深色衣物目标多次只剩窄条；slot13–27、40–43和58–60有大面积身体缺失。骨架在残缺时段发生朝向/屈膝变化，当前视觉证据不足以确认是真实动作；数值已有局部关节快速变化告警。局部恢复时段动作尚可，保留该身份并优先查其他窗口，不将整个人淘汰；当前整段未通过。')}
results=[]
for row in selection['selected']:
 package=Path(row['package']); dest=P/(row['split']+'_'+row['scene']+'_'+row['review_id'])
 manifest=json.loads((dest/'manifest.json').read_text())
 seen={'native':[],'projection':[]}
 for e in manifest['evidence']:
  for field in ['source','sheet']:
   assert hashlib.sha256(Path(e[field]).read_bytes()).hexdigest()==e[field+'_sha256']
  seen[Path(e['sheet']).stem.split('_')[0]].extend(e['frames'])
 assert seen['native']==list(range(25)) and seen['projection']==list(range(61))
 passed,state,note=notes[row['review_id']]
 review={'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'state':state,'input_review_passed':passed,'notes':note,
 'review_method':'All 25 native and 61 source/projection frames viewed in aspect-preserving contact sheets; not real-time video playback or 3D ground-truth validation.',
 'reviewed_native_frames':list(range(25)),'reviewed_projection_frames':list(range(61)),
 'evidence_manifest':str(dest/'manifest.json'),'evidence_manifest_sha256':hashlib.sha256((dest/'manifest.json').read_bytes()).hexdigest(),
 'selection_manifest':str(P/'selection.json'),'identity_rejected':False,'occlusion_alone_rejected':False,
 'quality_accepted':False,'generation_started':False,'full_video_visually_reviewed':False,
 'all_source_projection_frames_visually_reviewed':True,
 'limits':['No unobserved-joint ground truth.','No inference about whole-pool acceptance rate.','Technical READY remains immutable.','Input candidate is not generated quality acceptance.']}
 assert not (package/'review.json').exists()
 content=json.dumps(review,ensure_ascii=False,indent=2)+'\n'
 (package/'review.json').write_text(content);(dest/'review.json').write_text(content)
 results.append({'package':str(package),'review_id':row['review_id'],'state':state,'input_review_passed':passed,'evidence':str(dest/'manifest.json')})
(P/'results.json').write_text(json.dumps({'reviewed':4,'input_candidates':2,'held_windows':2,'identity_rejected':0,'generation_started':False,'rows':results},ensure_ascii=False,indent=2)+'\n')
print('Verified evidence hashes and all-frame indices; four reviews written.')
