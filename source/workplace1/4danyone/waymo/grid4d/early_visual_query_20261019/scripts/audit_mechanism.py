"""Terminal R3 mechanism observations, no optimization or quality tuning."""
from assets import *
import argparse


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, choices=['N04', 'N06', 'N11'])
    args = ap.parse_args()
    sys.addaudithook(guard_training)
    device = verify()
    begin = time.monotonic()
    asset = NEW / 'assets' / (args.label + '_R3')
    assert json.loads((asset / 'replay_validation.json').read_text())['passed']
    g, d, q, table, cfg, data, meta = load_portable(asset)
    assert meta['actual_step'] == 32999 and q.arm == 'R3'
    inputs = Inputs(args.label, data, all_frames=False)
    rows = []
    for frame in [0, data['frame_count'] // 2, data['frame_count'] - 1]:
        t = g.get_xyz.new_tensor([[frame / data['time_denominator']]])
        levels = inputs.support(frame)
        z, diagnostics = q.encode(d, t, levels, inputs.projections)
        table_error = float((z - table[frame]).abs().max())
        assert table_error <= 1e-5
        _, features = base_features(d, g.get_xyz.detach(), t)
        dp, dc, decoded = q.decode(g, t, z, features)
        changed = [x.clone() for x in levels]
        # Fixed diagnostic inversion; no model update and no new candidate output.
        for x in changed:
            x[:, :3] = 1 - x[:, :3]
            x[:, 4:6] = -x[:, 4:6]
        zp, _ = q.encode(d, t, changed, inputs.projections)
        pp, cp, _ = q.decode(g, t, zp, features)
        behind = inputs.projections.clone()
        behind[:, :, -1] = 0
        behind[:, 3, -1] = -1
        zn, null = q.encode(d, t, levels, behind)
        assert torch.isfinite(zn).all() and float(null['null_mean']) == 1.
        assert torch.isfinite(dp).all() and torch.isfinite(dc).all()
        rows.append(dict(
            frame=frame, baked_integer_max_abs=table_error,
            support_perturb_latent_max_abs=float((zp-z).abs().max()),
            support_perturb_world_position_max_abs=float((pp-dp).abs().max()),
            support_perturb_RGB_residual_max_abs=float((cp-dc).abs().max()) * C0,
            all_candidates_invalid_null_weight=float(null['null_mean']),
            diagnostics={k: float(v) for k, v in diagnostics.items()},
            position_fraction_max=float(decoded['position_fraction_max']),
            RGB_residual_max=float(decoded['RGB_residual_max']),
            association_rows=len(decoded['association']),
            association_max_index=int(decoded['association'].max())))
    trace = [json.loads(line) for line in (NEW/'runs'/f'{args.label}_R3/trace.jsonl').read_text().splitlines()]
    required = ['query.weight', 'key.weight', 'value.weight', 'offset.weight', 'head.weight']
    gradient = {k: max(r['query_grad_abs_sum'].get(k, 0.) for r in trace) for k in required}
    late = [r['diagnostics'] for r in trace if r['step'] >= 18000 and r['diagnostics']]
    late_ranges = {k: dict(min=min(r[k] for r in late), max=max(r[k] for r in late),
                          mean=sum(r[k] for r in late)/len(late)) for k in late[0]}
    assert torch.isfinite(table).all()
    report = dict(
        label=args.label, terminal=32999, device=device, parameter_updates=0,
        technical_training_budget_consumed=0, source_image_reads=inputs.source_reads,
        fixed_probe_frames=[r['frame'] for r in rows], rows=rows,
        formal_gradient_max=gradient,
        late_training_diagnostic_samples=len(late),
        late_training_diagnostic_ranges=late_ranges,
        baked_Z_all_times=dict(shape=list(table.shape), dtype=str(table.dtype),
                               bytes=table.numel()*table.element_size(),
                               finite=True,
                               temporal_std_mean=float(table.std(dim=0, unbiased=False).mean()),
                               temporal_std_max=float(table.std(dim=0, unbiased=False).max())),
        gradient_observed_for_all_required_parameters=all(v > 0 for v in gradient.values()),
        perturbation_reaches_3d_residual_all_sampled_times=all(
            r['support_perturb_world_position_max_abs'] > 0 or
            r['support_perturb_RGB_residual_max_abs'] > 0 for r in rows),
        no_new_quality_gate=True, unmeasured_times_not_claimed=True,
        attention_not_material_correspondence=True,
        note='Gradient and response show active computation; neither proves useful evidence alignment or quality gain.',
        seconds=time.monotonic()-begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(NEW/'research'/f'{args.label}_terminal_mechanism.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
