"""Finalize only after 20 complete deliveries and bounded visual review records exist."""
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
from compact_failures import main as compact

R = Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (R / 'reserve_pending.flag').exists()
    jobs = {}
    for name in ('STATUS.json', 'RESERVE_STATUS.json'):
        state = read(R / name)
        assert state['state'] != 'running'
        jobs.update(state['jobs'])
    assert sum(j['state'] == 'complete' for j in jobs.values()) == 20
    assert all(j['state'] in ('complete', 'failed') for j in jobs.values())
    audit = read(R / 'research/delivery_audit.json')
    assert audit['execution_target_verified'] and audit['completed_people'] == 20
    reviews = read(R / 'research/visual_review.json')['samples']
    native_reviews = read(R / 'research/input_visual_review.json')
    rows = sorted(audit['samples'], key=lambda x: x['scene'])
    assert set(reviews) == {r['sample'] for r in rows}
    for row in rows:
        b = R / row['path']
        assert sha(b / 'manifest.json') == row['manifest_sha256']
        assert sha(b / 'contact.jpg') == reviews[row['sample']]['sha256']
    compact()
    assert not list((R / 'scenes').glob('*/*/.work'))
    assert len(list((R / 'scenes').glob('*/*/manifest.json'))) == 20
    assert len(list((R / 'research/failures').glob('*/failure.json'))) == sum(j['state'] == 'failed' for j in jobs.values())
    state = {'state': 'complete_with_recorded_rejections', 'completed': 20, 'target': 20,
             'attempted': len(jobs), 'failed_candidates': sum(j['state'] == 'failed' for j in jobs.values()),
             'quality_approved': False, 'jobs': jobs}
    (R / 'BATCH_STATUS.json').write_text(json.dumps(state, indent=2))
    with (R / 'inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample', 'scene', 'identity_key', 'frames', 'path', 'generation_seconds', 'gpu', 'bytes'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    parts = ['<!doctype html><meta charset="utf-8"><title>Waymo 20人 / 4DAnyone Turbo</title>',
             '<style>body{background:#181818;color:#eee;font:16px sans-serif;max-width:1200px;margin:24px auto;padding:12px}a{color:#9df}video{max-width:100%;max-height:720px}article{border-top:1px solid #666;padding:20px 0}p{line-height:1.6}</style>',
             '<h1>20个人体轨迹 · 480个多视角视频</h1><p>公开冻结权重推理。每人24视角；本批未进行高斯重建。原始观测、模型输入和生成预览均可查看。</p>',
             '<p>技术验收通过不等于质量合格。下列逐人说明来自六视角首/中/末帧检查；未人工逐帧播放所有视频，不把未检查的闪烁当作不存在。遮挡、携带物和前后朝向仍可能出错。</p>',
             '<p><a href="REPORT.md">交付报告</a> · <a href="inventory.csv">文件清单</a> · <a href="completion_audit.json">验收记录</a></p>']
    table = ['|场景 / 人物|每视角帧数|实际生成分钟|观察与限制|', '|---|---:|---:|---|']
    for row in rows:
        key, rel = row['sample'], row['path']
        note = reviews[key]['observation']
        native = native_reviews.get(key, {}).get('observed', '真实观测行为尚未人工标注。')
        parts.append(f'<article id="{key}"><h2>scene {row["scene"]} / {key} · {row["frames"]}帧 × 24视角</h2><p>真实观测：{html.escape(native)}</p><p>生成检查：{html.escape(note)}</p><p><a href="{rel}/input/observed_native.mp4">真实观测裁剪</a> · <a href="{rel}/input/model_input.mp4">实际模型输入</a> · <a href="{rel}/input/native_contact.jpg">原始观测关键帧</a> · <a href="{rel}/contact.jpg">生成关键帧</a> · <a href="{rel}/manifest.json">来源与文件哈希</a></p><video controls preload="none" src="{rel}/preview.mp4"></video></article>')
        table.append(f'|{row["scene"]} / [{key}]({rel}/preview.mp4)|{row["frames"]}|{row["generation_seconds"]/60:.2f}|{note}|')
    (R / 'index.html').write_text('\n'.join(parts))
    total_mb = sum(r['bytes'] for r in rows) / 2**20
    report = f'''# Waymo 20人：4DAnyone Turbo 多视角视频交付

完成20条不同场景的人体轨迹、480个原始生成视频。共尝试{len(jobs)}条候选，{len(jobs)-20}条在前处理/一致性检查被拒绝；失败没有算入成功人数。按预先排序的候补补足，没有按生成好坏挑人。**全部生成结果保留，包括明显质量失败。** 本批拒绝原因：前景掩码门槛16条、取景半径门槛6条、曝光覆盖1条、人体两坐标系刚性一致性1条，均发生在扩散生成之前；原门槛未放宽，见 [失败分类](research/failure_summary.json)。

[直接查看20人](index.html) · [文件清单](inventory.csv) · [技术验收](completion_audit.json) · [实际视觉检查范围](research/visual_review.json)。完整成果约{total_mb:.1f} MiB，按 `scenes/场景/Pxx/` 保存；失败材料独立放在 `research/failures/`。

流程为真实Waymo轨迹与相机参数 → 等比例跟随裁剪 → RIFE时间插值 → SAM2人物分割 → GVHMR动作估计 → 去水平根运动的骨架条件 → 冻结4DAnyone Turbo生成24视角。原生25/49帧对应61/121帧、25fps、704×1280；没有重复末帧填满。前处理与根运动保留供后续审计，不代表已恢复真实世界尺度轨迹。前端没有训练，也没有为本批新增高斯后端训练。

每人保留：`input/`真实观测及模型输入、`videos/00.mp4`至`23.mp4`、`preview.mp4`六视角同步预览、`contact.jpg`首中末关键帧、`cameras.json`、`root_motion.npz`、`motion/`与`skeletons/`，以及来源和哈希。成功样例的临时PNG/JPEG、重复插值视频、姿态特征缓存均已清理。失败仅保留输入/诊断和日志；原始数据、旧成果和环境未清理。

全部原视频在交付时逐帧解码并检查尺寸、帧数、PTS及条件相机，本轮独立验收重新核对其精确SHA256；骨架视频、输入视频与拼图重新解码，检查相机/根运动/掩码有限数值、身份场景唯一性与真实源帧哈希。技术通过不等于动作、身份或多视角质量通过；没有真实新视角真值，不编造PSNR等质量指标。

已逐人查看六视角首/中/末帧，尚未人工逐帧观看全部视频。P04/P08有行走输入，但生成有相反视角重复正脸；P18/P22的遮挡输入对应异常屈膝姿态；P20伞形残影、P39腿间结构和P40大白环明确保留。P24/P29可看背身行走和携物输入，未观测脸部及背面仍是推断。不要直接把这20套视频都当作可靠重建监督。

''' + '\n'.join(table) + '''

计时来自每人实际生成元数据，含模型加载和视频生成，不含前处理、队列等待与本轮独立验收；共享GPU负载下不作为公平速度基准。后台队列达到20人后结束，原失败退出码和暂停原因保留；P42刚性一致性误差超过原1e-4阈值，未放宽阈值，仅登记拒绝并接续候补。

并行后端任务见 [primitive局部外观读取报告](../../grid4d/evq_primitive_local_20261019/FINDINGS.md)。六组后端已独立完成，局部查询收益未达到主验收条件，不自动追加训练。
'''
    (R / 'REPORT.md').write_text(report)
    runtime = {str(p.relative_to(R)): sha(p) for p in sorted((R / 'scripts').glob('*.py'))}
    (R / 'research/runtime_manifest_final.json').write_text(json.dumps(runtime, indent=2))
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'execution_complete': True, 'people': 20, 'scenes': 20, 'videos': 480,
              'quality_approved': False, 'original_data_modified': False,
              'delivery_audit_sha256': sha(R / 'research/delivery_audit.json'),
              'visual_review_sha256': sha(R / 'research/visual_review.json'),
              'requirements': {'scene_grouped': True, 'frozen_frontend_inference': True,
                               '24_views_each': True, 'unique_scene_tracks': True,
                               'useful_inputs_cameras_motion_retained': True, 'scratch_removed': True,
                               'failures_preserved': True, 'all_results_exposed_in_gallery': True},
              'runtime_manifest': 'research/runtime_manifest_final.json', 'inventory': 'inventory.csv'}
    (R / 'completion_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
