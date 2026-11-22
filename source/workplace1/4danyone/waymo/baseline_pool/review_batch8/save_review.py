import json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((O/'selection.json').read_text());rows=[]
notes=[('hold_window_person_segmentation_lost','深色外套人物原生49帧连续正常行走，首几帧部分身体靠右边界，之后身体完整。实际输入slot2–120人体完全消失，骨架仍输出并逐渐近似站立，不能代表原步态。batch_gate_mask记录119个连续低支持槽且对应mask面积为0；最早明确失效在分割/输入阶段，不是原片全程遮挡。保留身份，当前整窗不可用。'),('retain_completion_candidate_occlusion_and_terminal_truncation','黑外套、蓝裤、橙色包人物主要站立，前景另一行人穿过。实际输入约slot10–35躯干/腿大面积丢失，期间骨架出现明显侧转和屈曲，源图不足以支持这些隐藏动作，不能因数值无告警就判通过。后半段主体与站立骨架较稳定；slot120左侧身体被截断，原生末帧也出现黑边。保留后半段同人窗口价值；本次不直接把黑边归因于源相机或裁剪程序，需源帧核查。')]
for row,(state,note) in zip(selection['selected'],notes):
 package=Path(row['package']);assert sha(package/'READY.json')==row['ready_sha256'];manifest=O/row['review_id']/'manifest.json';m=json.loads(manifest.read_text())
 for e in m['evidence']:
  assert sha(Path(e['source']))==e['source_sha256'] and sha(Path(e['sheet']))==e['sheet_sha256']
 result=dict(reviewed_at=time.time(),state=state,input_review_passed=False,notes=note,review_method='All49 native and121 projection frames inspected as contact sheets, not real-time temporal playback',reviewed_native_frames=list(range(49)),reviewed_projection_frames=list(range(121)),evidence_manifest=str(manifest),evidence_manifest_sha256=sha(manifest),quality_accepted=False,identity_rejected=False,occlusion_alone_rejected=False,generation_started=False,generation_allowed=False,full_video_visually_reviewed=False,all_source_projection_frames_visually_reviewed=True)
 assert not (package/'review.json').exists();(package/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');rows.append(dict(**row,**result))
(O/'results.json').write_text(json.dumps(dict(at=time.time(),selection_sha256=sha(O/'selection.json'),rows=rows,counts={'input_candidate':0,'completion_candidate':1,'held_window':1},reviewed_frames={'native':98,'projection':242},new_people=0,all_source_and_sheet_hashes_verified=True),ensure_ascii=False,indent=2)+'\n')
(O/'REPORT.md').write_text('# 第八批：两条121槽输入复核\n\n固定名单后查看两个此前未审查场景：train有数值告警、val无告警各一条，以固定路径哈希选择。不是总体通过率抽样。共检查98个原生帧与242个输入/骨架槽，12张接触表及来源哈希已核验。\n\n'+ '\n\n'.join('**'+r['split']+'/'+r['scene']+' '+r['review_id']+'**：'+r['notes'] for r in rows)+'\n\n两条都不拒绝身份，但不判整窗可用。优先按原筛选门槛检查后续较完整窗口；不新增修复模块、RGB生成或高斯训练。记录见 results.json；总入口为 ../reviewed_inputs.html。\n')
print('Two complete temporal contact reviews saved; no quality acceptance')
