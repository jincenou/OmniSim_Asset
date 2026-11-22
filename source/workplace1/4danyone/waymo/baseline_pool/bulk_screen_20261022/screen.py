"""Whole-pool conservative triage using existing full-time evidence, not manual batches."""
import collections,csv,hashlib,html,json,os,time
from pathlib import Path
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.writing');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def longest(values):
    run=maximum=0
    for value in values:
        run=run+1 if value else 0;maximum=max(maximum,run)
    return maximum

def main():
    started=time.time();plan=json.loads((R/'PLAN.json').read_text());policy=plan['new_engineering_review_rules_frozen_before_counts']
    audit={r['package']:r for r in json.loads((P/'research/package_integrity/results.json').read_text())}
    paths=sorted([*(P/'samples').glob('*/*/*/READY.json'),*(P/'retry').glob('*/*/*/*/READY.json')]);rows=[]
    for path in paths:
        d=json.loads(path.read_text());c=d['candidate'];package=path.parent;rel=str(package.relative_to(P))
        evidence=audit.get(rel,{});reasons=[];integrity=evidence.get('state')=='integrity_passed_quality_pending' and evidence.get('ready_sha256')==sha(path)
        bone_path=package/'skeleton_audit.json';bone=json.loads(bone_path.read_text()) if bone_path.exists() else {}
        review_path=package/'review.json';review=json.loads(review_path.read_text()) if review_path.exists() else {}
        flags=bone.get('flags');areas=d.get('mask_risks',{}).get('mask_area_per_frame');empty=[x==0 for x in areas] if areas is not None else []
        empty_fraction=sum(empty)/len(empty) if empty else None;empty_seconds=longest(empty)/25 if empty else None
        native=json.loads((package/'clip0/native.json').read_text())
        with Image.open(native['frames'][0]['raw_source_path']) as source_image:
            source_width,source_height=source_image.size
        touch=[b['bbox_xyxy'][0]<=policy['boundary_distance_pixels'] or b['bbox_xyxy'][1]<=policy['boundary_distance_pixels'] or b['bbox_xyxy'][2]>=source_width-policy['boundary_distance_pixels'] or b['bbox_xyxy'][3]>=source_height-policy['boundary_distance_pixels'] for b in c['observations']]
        if not integrity:reasons.append('missing_changed_or_failed_integrity_evidence')
        if flags is None or bone.get('state') not in ('no_numeric_flag_visual_review_pending','review_priority'):reasons.append('missing_complete_skeleton_diagnostic')
        reasons.extend(flags or [])
        if areas is None or len(areas)!=c['output_slots']:reasons.append('mask_evidence_incomplete')
        elif empty_fraction>=policy['binary_empty_fraction_at_least'] or empty_seconds>=policy['binary_empty_contiguous_seconds_at_least']:reasons.append('persistent_near_empty_binary_foreground_review')
        if longest(touch)>=policy['source_bbox_boundary_native_run_at_least'] or sum(touch)/len(touch)>=policy['source_bbox_boundary_fraction_at_least']:reasons.append('persistent_source_border_contact_review')
        state=review.get('state','unreviewed');passed=review.get('input_review_passed') is True
        if review and not passed:reasons.append('manual_review_does_not_pass_or_is_incomplete')
        if not integrity:group='unsuitable_current_window'
        elif passed:group='priority_generation_candidate'
        elif review and not passed and (state.startswith(('defer','reject','failed')) or 'terminal' in state or 'pose_error' in state or 'no_visible_person' in state):group='unsuitable_current_window'
        else:group='needs_review' if reasons else 'priority_generation_candidate'
        rows.append(dict(package=str(package),split=c['split'],scene=c['scene'],review_id=c['review_id'],camera=c['camera_index'],
            identity_scope=c['identity_scope'],identity_key=c['identity_key'],group=group,review_triggers=reasons,
            frames=c['output_slots'],native_frames=c['native_frame_count'],window_start=c['observations'][0]['frame_index'],window_end=c['observations'][-1]['frame_index'],
            source_width=source_width,source_height=source_height,height_median_px=c.get('height_median_px'),binary_empty_fraction=empty_fraction,longest_binary_empty_seconds=empty_seconds,
            source_boundary_fraction=sum(touch)/len(touch),source_boundary_longest_native=longest(touch),
            pose_reprojection_median=bone.get('reprojection',{}).get('median_detector_reprojection_over_bbox_height'),
            pose_reprojection_p95=bone.get('reprojection',{}).get('p95_detector_reprojection_over_bbox_height'),
            manual_review=state,input_review_passed=passed,quality_accepted=False,generation_allowed=False,
            source=str(package/'clip0/observed_native.mp4'),input=str(package/'clip0/inputs/main.mp4'),pose=str(package/d['input_skeleton_comparison']),
            ready_sha256=sha(path),pose_evidence_sha256=sha(bone_path) if bone_path.exists() else None))
    candidates=[r for r in rows if r['group']=='priority_generation_candidate']
    selected={}
    for row in candidates:
        key=(row['identity_scope'],row['identity_key']);rank=(row['input_review_passed'],-len(row['review_triggers']),row['height_median_px'] or 0,row['frames'],row['package'])
        if key not in selected or rank>selected[key][0]:selected[key]=(rank,row)
    selected_rows=[v[1] for v in selected.values()]
    summary=dict(at=time.time(),seconds=time.time()-started,local_scanned_scenes=json.loads((P/'POOL_STATUS.json').read_text())['scanned_scenes'],
        prepared_windows=len(rows),prepared_scenes=len({(r['split'],r['scene']) for r in rows}),groups=dict(collections.Counter(r['group'] for r in rows)),
        priority_windows_by_split=dict(collections.Counter(r['split'] for r in candidates)),priority_scenes=len({(r['split'],r['scene']) for r in candidates}),
        priority_identity_keys_by_scope=dict(collections.Counter(r['identity_scope'] for r in selected_rows)),selected_windows_one_per_existing_identity_key=len(selected_rows),
        manual_input_pass_windows=sum(r['input_review_passed'] for r in candidates),automatic_only_candidate_windows=sum(not r['input_review_passed'] for r in candidates),
        review_trigger_counts=dict(collections.Counter(t for r in rows for t in r['review_triggers'])),
        whole_pool_verified_unique_people=None,quality_accepted_people=None,plan_sha256=sha(R/'PLAN.json'),script_sha256=sha(Path(__file__)),
        integrity_audit_sha256=sha(P/'research/package_integrity/results.json'),generation_started=False,
        limitations=['Candidate counts are input windows, not proven quality or independent people.',
                    'Unverified camera identities may duplicate each other or official identities; no summed unique-person claim.',
                    'No mask absence or occlusion flag alone rejects an identity. Unsuitable applies only to this observed window.',
                    'Detector-pose consistency and stable skeleton do not prove correct hidden action, carried-object appearance or final multiview quality.',
                    'New generation still requires the accessory-input check; this batch does not launch generation.'])
    write(R/'summary.json',summary);write(R/'results.json',rows)
    for name,values in [('all_windows.csv',rows),('priority_candidates.csv',candidates),('selected_by_identity_scope.csv',selected_rows)]:
        with (R/name).open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader()
            out.writerows({**row,'review_triggers':json.dumps(row['review_triggers'])} for row in values)
    labels={'priority_generation_candidate':'优先试生成候选','needs_review':'需要复核','unsuitable_current_window':'当前窗口不适合'}
    blocks=['<!doctype html><meta charset="utf-8"><title>Waymo全量批筛</title><style>body{font:15px sans-serif;max-width:1400px;margin:24px auto;background:#171717;color:#eee}a{color:#9df}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #444}input{padding:10px;width:50%}</style><h1>Waymo 全量批量筛查</h1>',
        '<p>全部本地1000场景；以下为准备完成窗口的筛查。自动候选不是人工质量通过，也不是独立人数。未新增生成。所有视频引用现有文件，不复制。</p>',
        f'<p>完整窗口 {len(rows)}；优先候选 {len(candidates)}；按既有身份键选一窗 {len(selected_rows)}（含未关联相机轨迹）。</p>',
        '<p><a href="REPORT.md">报告</a> · <a href="priority_candidates.csv">全部优先窗口CSV</a> · <a href="selected_by_identity_scope.csv">按既有身份键选一窗CSV</a> · <a href="summary.json">计数与限制</a></p>',
        '<input id="q" placeholder="搜索场景、编号、问题"><button onclick="filterRows()">筛选</button>']
    for group,label in labels.items():
        subset=[r for r in rows if r['group']==group];blocks.append(f'<details><summary>{label}：{len(subset)}</summary><table><tr><th>场景/窗口</th><th>输入与骨架</th><th>提示</th></tr>')
        for row in subset:
            links=' · '.join(f'<a href="{html.escape(os.path.relpath(row[k],R))}">{lab}</a>' for k,lab in [('source','真实观测'),('input','实际输入'),('pose','骨架对照')])
            text=f"{row['split']}/{row['scene']}/{row['review_id']} cam{row['camera']} {row['window_start']}–{row['window_end']}"
            note=('已人工输入复核；' if row['input_review_passed'] else '自动筛查；')+str(row['review_triggers'])
            blocks.append(f'<tr class="data"><td>{html.escape(text)}</td><td>{links}</td><td>{html.escape(note)}</td></tr>')
        blocks.append('</table></details>')
    blocks.append('<script>function filterRows(){let q=document.getElementById("q").value.toLowerCase();document.querySelectorAll("tr.data").forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(q));if(q)document.querySelectorAll("details").forEach(d=>d.open=true)}document.getElementById("q").addEventListener("input",filterRows)</script>')
    (R/'index.html').write_text('\n'.join(blocks));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
