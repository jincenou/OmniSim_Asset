"""Read-only native contribution-set evidence; does not change gradient acceptance."""
import argparse
import sys
import time
import numpy as np
import torch
from PIL import Image
from technical_check import captured_output, contributors
from boundary_common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    device = verify()
    sys.addaudithook(protect_inputs)
    out = NEW / 'research' / f'{args.label}_cutoff_audit.json'
    assert not out.exists()
    model, deform, pipe, config, data, state = restore(args.label, regularize=False)
    anchor = next(a for a in json.loads((NEW/'data/anchors.json').read_text()) if a['label'] == args.label)
    paths = json.loads(Path(plan()['identities'][args.label]['data']['training']).read_text())
    row = next(r for r in paths if r['view'] == anchor['view'] and r['frame'] == anchor['frame'])
    with Image.open(row['path']) as image:
        target = torch.tensor(np.asarray(image)[..., 3].copy(), device='cuda', dtype=torch.float32) / 255
    partition = regions(target.cpu().numpy())
    cam = camera(data, anchor['view'], anchor['frame'])
    with torch.no_grad():
        warp = deform.step(model.get_xyz.detach(), cam.fid.unsqueeze(0).expand(len(model.get_xyz), -1))
    def output():
        return render(cam, model, pipe, torch.zeros(3, device='cuda'), warp['d_xyz'], warp['d_rotation'], warp['d_scaling'], override_color=torch.ones_like(model.get_xyz))['render'][0]
    initial = model._opacity.detach().clone()
    epsilon = .001
    alpha, base = captured_output(output)
    loss = extra_alpha(alpha.double(), target.double(), torch.tensor(partition, device='cuda'), 'B')[0]
    analytic = float(torch.autograd.grad(loss, model._opacity)[0].sum())
    with torch.no_grad():
        model._opacity.copy_(initial + epsilon)
        plus, pbuf = captured_output(output)
        model._opacity.copy_(initial - epsilon)
        minus, mbuf = captured_output(output)
        model._opacity.copy_(initial)
    assert torch.equal(model._opacity, initial)
    part = torch.tensor(partition, device='cuda')
    numeric = float((extra_alpha(plus.double(), target.double(), part, 'B')[0] - extra_alpha(minus.double(), target.double(), part, 'B')[0]) / (2 * epsilon))
    relative = abs(analytic - numeric) / max(abs(numeric), 1e-12)
    # Audit every reference boundary pixel, not only pixels selected for a small FD error.
    locations = np.argwhere((partition == 2) | (partition == 3))
    switches = 0
    examples = []
    for y, x in locations:
        sets = [contributors(buffers, len(model.get_xyz), int(y), int(x)) for buffers in (base, pbuf, mbuf)]
        if not (sets[0]['ids'] == sets[1]['ids'] == sets[2]['ids']):
            switches += 1
            if len(examples) < 8:
                examples.append(dict(y=int(y), x=int(x), region=int(partition[y,x]), contributor_ids={key: value['ids'] for key, value in zip(['base','plus','minus'], sets)}, alpha=[float(a[y,x]) for a in (alpha, plus, minus)]))
    save(out, dict(state='completed', label=args.label, device=device, view=anchor['view'], frame=anchor['frame'], epsilon=epsilon, boundary_pixels_checked=len(locations), pixels_with_changed_contributors=switches, examples=examples, full_band_directional=dict(analytic=analytic,numeric=numeric,relative_error=relative,within_original_one_percent=relative < .01), optimizer_updates=0, opacity_restored_exactly=True, render_calls=3, backward_calls=1, seconds=time.monotonic()-begin, interpretation='Exact native forward contributor switches under the original perturbation are recorded. This does not apportion all FD error to switches or replace the original failed full-band check.'))
    print(args.label, 'boundary pixels', len(locations), 'contribution switches', switches, flush=True)


if __name__ == '__main__':
    main()
