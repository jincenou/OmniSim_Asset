"""Remove reproducible CEM intermediates, preserving complete dynamic assets."""
import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['N01', 'N02', 'N03', 'N04', 'N06', 'N07', 'N08']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', nargs='+', choices=LABELS + ['N11'], default=LABELS)
    parser.add_argument('--receipt', choices=['cleanup_completed.json', 'cleanup_N11.json'], default='cleanup_completed.json')
    args = parser.parse_args()
    labels = args.labels
    receipt = ROOT / 'research' / args.receipt
    assert not receipt.exists(), 'One recorded cleanup only; review an existing receipt before further removal'
    keep, targets = [], []
    for label in labels:
        result = json.loads((ROOT / 'results' / label / 'results.json').read_text())
        assert len(result['videos']) == 3
        for stage in (1, 2):
            run = ROOT / 'runs' / label / f'stage{stage}'
            record = json.loads((run / 'complete.json').read_text())
            asset = run / 'asset.pt'
            assert digest(asset) == record['asset_sha256']
            assert max(x['max_error'] for x in result['replay'][f'cem_stage{stage}']) < 1e-5
            keep.append({'path': str(asset.relative_to(ROOT)), 'sha256': record['asset_sha256']})
            targets.extend((p, 'Regenerable native PLY subset; full dynamic asset.pt is retained')
                           for p in (run / 'point_cloud').rglob('*.ply'))
        # Stage1 native checkpoint is retained because the frozen stage2 recipe reads it.
        targets.append((ROOT / 'runs' / label / 'stage2/chkpnt35000.pth',
                        'Intermediate native save; complete stage2 asset and optimizer state retained'))
        raw = ROOT / 'tmp' / f'{label}_hust_raw'
        assert (raw / 'complete.json').exists()
        targets.extend((p, 'Temporary rerender cache; source HUST assets and published comparison retained')
                       for p in raw.iterdir() if p.is_file())
    rows = []
    for path, reason in targets:
        assert path.is_file() and not path.is_symlink()
        assert path.resolve().is_relative_to(ROOT)
        assert not any(parent.is_symlink() for parent in path.parents if parent != ROOT)
        rows.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                     'sha256': digest(path), 'reason': reason, 'removed': False})
    state = {'started_at': datetime.now(timezone.utc).isoformat(), 'state': 'validated',
             'labels': labels,
             'protected': ['all cases outside labels', 'results', 'diagnostics/opacity', 'sources',
                           'logs', 'attempts', 'all original Random8 data/results',
                           'all asset.pt, RNG, sampling, config, stage1 native checkpoints'],
             'retained_assets': keep, 'files': rows}
    receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        path = ROOT / row['path']
        assert path.stat().st_size == row['bytes']
        path.unlink()
        row['removed'] = True
        receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    for label in labels:
        for stage in (1, 2):
            root = ROOT / 'runs' / label / f'stage{stage}' / 'point_cloud'
            for directory in sorted(root.rglob('*'), key=lambda x: len(x.parts), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            if root.exists() and not any(root.iterdir()):
                root.rmdir()
        raw = ROOT / 'tmp' / f'{label}_hust_raw'
        if not any(raw.iterdir()):
            raw.rmdir()
    assert all((ROOT / row['path']).exists() for row in keep)
    state.update(state='complete', ended_at=datetime.now(timezone.utc).isoformat(),
                 removed_bytes=sum(row['bytes'] for row in rows), removed_files=len(rows))
    receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: state[key] for key in ['state', 'removed_bytes', 'removed_files']}))


if __name__ == '__main__':
    main()
