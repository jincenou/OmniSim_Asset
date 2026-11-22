"""Measured costs; distinguish shared-prefix work, subprocess time and polling."""
import csv
import json
from collections import defaultdict
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def main():
    plan = read(NEW/'PLAN.json')
    warmups = {}
    rows = []
    for label in plan['labels']:
        warmups[label] = read(NEW/'runs'/f'{label}_warmup/training.complete.json')
        assert warmups[label]['step'] == 2999
        for arm in plan['arms']:
            key = f'{label}_{arm}'
            run = NEW/'runs'/key
            done = read(run/'training.complete.json')
            snap = read(run/'snapshot_32999.json')
            assert done['step'] == 32999
            rows.append(dict(
                key=key, arm=arm, label=label,
                branch_entry_seconds=done['wall_seconds'],
                shared_prefix_entry_seconds=warmups[label]['wall_seconds'],
                equivalent_one_person_cold_entry_seconds=done['wall_seconds']+warmups[label]['wall_seconds'],
                branch_load_seconds=snap['load_seconds'],
                points=snap['points'], gpu_uuid=snap['device']['uuid'],
                actual_new_branch_optimizer_calls=30000,
                logical_including_prefix_optimizer_calls=32999))
    postgroups = defaultdict(lambda: dict(jobs=0, controller_observed_seconds=0., failed=0))
    for job in read(NEW/'post_history.json'):
        key=job['key']
        group = 'global_or_dense_evaluation' if key.endswith('_metrics') else key.rsplit('_', 1)[-1]
        postgroups[group]['jobs'] += 1
        postgroups[group]['controller_observed_seconds'] += job['seconds']
        postgroups[group]['failed'] += int(job['exit_code'] != 0)
    exact_eval = [dict(key=p.parent.name, measured_main_seconds=read(p)['seconds'])
                  for p in sorted((NEW/'results').glob('*/metrics.json'))]
    prefixes = sum(x['wall_seconds'] for x in warmups.values())
    branches = sum(x['branch_entry_seconds'] for x in rows)
    totals = dict(
        formal_unique_optimizer_calls=3*2999+12*30000,
        terminal_logical_optimizer_calls=12*32999,
        explanation='The three cold prefixes are physically computed once, then each reused by four arms. Per-Gaussian rows can skip updates on native topology steps; see recovery audit.',
        technical_optimizer_calls=read(NEW/'research/preflight.json')['technical_updates'],
        unique_training_main_entry_seconds=prefixes+branches,
        unique_prefix_main_entry_seconds=prefixes,
        unique_branch_main_entry_seconds=branches,
        main_entry_scope='Measured inside training main: loading and checkpoint writes included, interpreter import excluded; concurrent job seconds are summed, not elapsed project time.',
        postprocessing_groups=dict(postgroups),
        postprocessing_time_scope='Controller observations include process launch/import and detection latency, normally up to the polling interval plus controller work. They are not pure GPU time or fair throughput measurements.',
        evaluation_main_seconds=exact_eval,
        evaluation_scope='Inside evaluate.main, including loading, rendering, metrics and writes; excludes interpreter import.',
        shared_GPU_load=True,
        fixed_step_count_not_equal_compute_to_old_single_view_fused=True,
        do_not_rank_algorithms_by_these_wall_times=True,
        unmeasured=dict(pure_GPU_kernel_time=None, per_checkpoint_save_seconds=None,
                        standalone_export_vs_bake_seconds=None,
                        postprocessing_individual_import_seconds=None),
        no_new_timing_training=True)
    with (NEW/'cost_entry_times.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (NEW/'cost_summary.json').write_text(json.dumps(totals, indent=2, allow_nan=False)+'\n')
    lines=['# 本轮成本口径', '',
           '以下为训练入口内部墙钟时间，包含加载和保存，不含解释器导入、前端生成、等待及后续评测。GPU 与其他任务共享负载，不能用此表给方法排公平速度名次。', '',
           '|对象|分支|本分支分钟|共享预热分钟|合计分钟（单独冷启动口径）|终态高斯数|',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['label']}|{r['arm']}|{r['branch_entry_seconds']/60:.2f}|{r['shared_prefix_entry_seconds']/60:.2f}|{r['equivalent_one_person_cold_entry_seconds']/60:.2f}|{r['points']}|")
    lines += ['', f'实际新执行的正式优化调用总数为 **{totals["formal_unique_optimizer_calls"]:,}**，另有 {totals["technical_optimizer_calls"]} 次技术测试更新。每个终态逻辑预算为 32999，不能把四分支共同预热重复计成四份物理训练。',
              '', '新协议每次有两个目标视角，R2/R3 还有同时间支持特征查询；细节见 costs.csv。控制器记录的导出/回放/部位评价耗时含退出检测延迟，不伪称纯 GPU 时间。没有单独测量的阶段记为未测量，未为补计时重训。',
              '', '[训练与参数成本](costs.csv) · [入口时间](cost_entry_times.csv) · [完整口径及后处理开销](cost_summary.json)']
    (NEW/'COSTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
