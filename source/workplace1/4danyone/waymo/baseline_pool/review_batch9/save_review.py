import json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((O/'selection.json').read_text());rows=[]
notes=[('retain_completion_candidate_terminal_truncation','灰色外套人物被前景红背包行人部分遮挡，实际输入多处腿部仅保留局部，骨架仍输出连续交替步态。不能仅凭这些局部缺失排除人物；不过slot52–60可见脚部投影与输入偏离、末段人体被左边界截断，整窗不判通过。前面较完整时段可保留为补全候选；隐藏肢体真实动作尚不能核实，需检查同人避开终端截断的窗口。数值无告警不等于视觉通过。'),('retain_completion_candidate_initial_occlusion_pose_error','紫红上衣、浅紫裤人物起初被车辆和前方行人遮住下半身。slot0–3实际输入主要只剩上身，GVHMR给出很深的屈膝/蹲伏骨架，后续完整观测显示正常直立行走，源视频没有支持这次蹲起动作的可见证据。slot约12以后输入恢复完整，骨架连续步态较合理，保留同人的后续窗口价值；整窗暂不通过，不因遮挡本身拒绝身份。')]

for row,(state,note) in zip(selection['selected'],notes):
 package=Path(row['package']);assert sha(package/'READY.json')==row['ready_sha256'];manifest=O/row['review_id']/'manifest.json';m=json.loads(manifest.read_text())
 for e in m['evidence']:
  assert sha(Path(e['source']))==e['source_sha256'] and sha(Path(e['sheet']))==e['sheet_sha256']
 result=dict(reviewed_at=time.time(),state=state,input_review_passed=False,notes=note,review_method='All25 native and61 projection frames inspected as contact sheets, not real-time temporal playback',reviewed_native_frames=list(range(25)),reviewed_projection_frames=list(range(61)),evidence_manifest=str(manifest),evidence_manifest_sha256=sha(manifest),quality_accepted=False,identity_rejected=False,occlusion_alone_rejected=False,generation_started=False,generation_allowed=False,full_video_visually_reviewed=False,all_source_projection_frames_visually_reviewed=True)
 assert not (package/'review.json').exists();(package/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');rows.append(dict(**row,**result))
(O/'results.json').write_text(json.dumps(dict(at=time.time(),selection_sha256=sha(O/'selection.json'),rows=rows,counts={'input_candidate':0,'completion_candidate':2,'held_window':0},reviewed_frames={'native':50,'projection':122},new_people=0,all_source_and_sheet_hashes_verified=True),ensure_ascii=False,indent=2)+'\n')
(O/'REPORT.md').write_text('# 第九批：两条61槽输入复核\n\n固定名单后查看此前未复核场景：train无数值告警、val有告警各一条，固定路径哈希选择，不能用于估计总体通过率。两人共50个原生帧与122个输入/骨架槽均通过六张接触表查看；尚未实时观看全部动作。\n\n'+ '\n\n'.join('**'+r['split']+'/'+r['scene']+' '+r['review_id']+'**：'+r['notes'] for r in rows)+'\n\n两人身份均保留为补全候选，尚不判整窗可用。局部肢体缺失本身不是排除标准；重点看骨架是否合理、可见投影是否失配以及边界截断。没有启动RGB生成或高斯训练。\n')
print('Two complete temporal contact reviews saved; no quality acceptance')
