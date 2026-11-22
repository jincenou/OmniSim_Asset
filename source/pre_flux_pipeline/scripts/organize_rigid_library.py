"""Consolidate published rigid assets, then prune verified completed intermediates."""
import argparse
import asyncio
import copy
import csv
import fcntl
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

L = Path('/mnt/ojc/workplace1/asset_library')
B = Path('/mnt/ojc/workplace1/asset_work/vehicle1000_20261025')
R = L / 'rigid'
A = L / 'maintenance/rigid_consolidation'
WORK = Path('/mnt/ojc/workplace1/asset_work')
sys.path.insert(0, '/mnt/wmj/workspace4/pipeline/traj_pipeline')
import numpy as np
from corner_case.assets import AssetLibrary, bind_rigid


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.organizing.tmp')
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def audit(action, **fields):
    with (A / 'operations.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(), action=action, **fields), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def include(source, target):
    """Hardlinks preserve exact evidence without another physical copy."""
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        # Read-only dataset files may belong to another account; copy those inputs.
        if source.stat().st_uid == os.geteuid() and source.stat().st_dev == target.parent.stat().st_dev:
            os.link(source.resolve(), target)
        else:
            shutil.copyfile(source, target)
    if source.stat().st_size != target.stat().st_size or sha(source) != sha(target):
        raise ValueError(f'Content differs: {target}')
    return dict(path=target.name, bytes=target.stat().st_size, sha256=sha(target))


def alias(source, target):
    """Replace an old duplicate with an atomic compatibility link."""
    source, target = Path(source), Path(target)
    if source.is_symlink() and source.resolve() == target.resolve():
        return
    tmp = source.with_name(source.name + '.organizing.link')
    if tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(os.path.relpath(target, source.parent))
    tmp.replace(source)


def asset_dir(row):
    prefix = R if row['status'] == 'keep' else R / '_review'
    batch = row['provenance'].get('batch_id', '')
    tag = {'vehicle1000_20261025': 'batch1000',
           'batch20_20261024:frontend_revision_01': 'pilot',
           'waymo_bus_more_20261025': 'bus_more'}.get(batch, 'archive')
    name = '_'.join([row['route'], tag, row['dataset'], row['provenance']['source_obj_id'],
                     hashlib.sha256(row['uid'].encode()).hexdigest()[:8]])
    if any('/' in p or '..' == p for p in [row['category'], name]):
        raise ValueError('Unsafe asset name')
    return prefix / row['category'] / name


def migrate_name(row, dest):
    old_name = row['dataset'] + '_' + row['provenance']['source_obj_id'] + '_' + hashlib.sha256(row['uid'].encode()).hexdigest()[:8]
    old = dest.parent / old_name
    if not old.exists():
        return
    if not (old / 'package_complete.json').exists():
        raise ValueError(f'Incomplete earlier package: {old}')
    dest.mkdir(parents=True, exist_ok=True)
    for path in old.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Unexpected package member: {path}')
        target = dest / path.name
        if not target.exists():
            os.link(path, target)
        elif not os.path.samefile(path, target) and sha(path) != sha(target):
            raise ValueError('Naming migration content mismatch')
    aliases = [L / row['canonical']['canonical_path']]
    ledger = old / 'cleanup_complete.json'
    if ledger.exists():
        aliases += [Path(r['path']) for r in read(ledger)['files'] if r['geometry_action'] == 'deduplicate']
    for source in aliases:
        if source.is_symlink() and source.resolve() == (old / 'asset.ply').resolve():
            alias(source, dest / 'asset.ply')
    if ledger.exists():
        cleanup = read(ledger)
        for record in cleanup['files']:
            record['replacement'] = str(dest / 'asset.ply')
        write(dest / 'cleanup_complete.json', cleanup)
    audit('rename_verified', uid=row['uid'], old=str(old), new=str(dest))
    shutil.rmtree(old)


def committed(row):
    p = row['provenance']
    return p.get('batch_id') != B.name or (B / 'chunks' / p['batch_chunk'] / 'completed.json').exists()


def package(row):
    dest = asset_dir(row)
    migrate_name(row, dest)
    receipt = dest / 'package_complete.json'
    if receipt.exists():
        old = read(receipt)
        if old['uid'] != row['uid'] or old['canonical_sha256'] != row['canonical']['sha256']:
            raise ValueError('Immutable package identity changed')
        for record in old['files']:
            p = dest / record['path']
            if not p.is_file() or p.stat().st_size != record['bytes']:
                raise ValueError(f'Package incomplete: {p}')
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    original = copy.deepcopy(row)
    mapping, files = {}, []

    def add(source, name, required=True):
        if not source or not Path(source).is_file():
            if required:
                raise FileNotFoundError(str(source))
            return
        files.append(include(source, dest / name))
        mapping[str(source)] = name

    canonical = L / row['canonical']['canonical_path']
    if sha(canonical) != row['canonical']['sha256']:
        raise ValueError('Central payload hash differs')
    add(canonical, 'asset.ply')
    add(row['provenance']['source_image'], 'source.jpg')
    add(row['provenance']['source'].get('source_image'), 'source_frame.jpg')
    add(row['quality']['audit_image'], 'source_and_renders.jpg')
    add(row['quality']['rendering']['evidence'], 'orbit_16.jpg')
    add(row['quality']['rendering'].get('heading_evidence'), 'heading.jpg', False)
    for i, ref in enumerate(row['provenance'].get('real_images', [])):
        add(ref['rgba_path'], f'real_{i:02d}.png')
        add(ref.get('source_image'), f'real_context_{i:02d}.jpg', False)
    raw = Path(row['provenance']['raw_source_path'])
    for p in sorted(raw.parent.glob('flux_*.png')):
        add(p, p.name)
    if row['provenance'].get('batch_id') == 'waymo_bus_more_20261025':
        old = L / 'previews/bus/waymo_bus_more_20261025' / row['provenance']['source_obj_id']
        for name in ['orbit_16.mp4', 'comparison.jpg']:
            add(old / name, name, False)

    def remap(value):
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v) for v in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    local = remap(original)
    local['source_path'] = local['canonical']['canonical_path'] = 'asset.ply'
    local['storage'] = dict(layout='rigid_by_category_v1', original_paths_are_provenance=True,
        canonical_transform_already_applied=True, metric_scale_baked=False)
    write(dest / 'asset.json', local)
    write(dest / 'asset_library_rigid.json', dict(schema_version=2, total=1, assets=[local]))
    write(dest / 'source_camera.json', dict(source=local['provenance']['source'],
        real_images=local['provenance'].get('real_images', []),
        note='Source camera and timestamps, not the camera of a future insertion scene.'))
    write(dest / 'generation.json', dict(route=row['route'], prompts=row['provenance']['prompts'],
        real_count=len(row['provenance'].get('real_images', [])),
        generation_signature=row['provenance'].get('generation_signature'),
        raw_sha256=row['provenance'].get('raw_sha256'),
        canonical_transform=row['canonical'], source_provenance=original['provenance']))
    if row['status'] == 'keep':
        library = AssetLibrary(dest / 'asset_library_rigid.json', dest)
        assert len(library.entries) == 1
        bound = bind_rigid(library.entries[0], np.random.default_rng(42))
        assert bound.asset_path.is_file() and bound.scale > 0 and min(bound.size_lwh_m) > 0
        write(dest / 'binding.json', dict(uid=bound.uid, asset_path='asset.ply', scale=float(bound.scale),
            size_lwh_m=list(bound.size_lwh_m),
            coordinate=dict(up='+Z', ground_z=0, heading=row['canonical'].get('heading'),
                canonical_transform_already_applied=True, metric_scale_baked=False),
            scale_source='Original corner_case.assets.bind_rigid, seed 42, including its +/-6% jitter; visual size estimate, not sensor measurement.',
            insertion_scene=None, insertion_trajectory=None))
    write(dest / 'README.txt', '标准刚体资产。asset.ply 已居中、接地、旋转到规范朝向，不要再应用 canonical.rotation/translation。\n'
          '使用 asset_library_rigid.json 和本目录作为旧 AssetLibrary 的 pipeline_root。\n'
          'binding.json 来自原 bind_rigid，插入时仅施加一次 scale，再由目标场景设置位置、yaw 和轨迹。\n'
          'source_camera.json 是来源相机；尺寸是视觉估计值。历史绝对路径仅作溯源，当前使用本目录文件。\n')
    for p in sorted(dest.iterdir()):
        if p.is_file() and p.name not in {f['path'] for f in files} and p != receipt:
            files.append(dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)))
    write(receipt, dict(uid=row['uid'], status=row['status'], canonical_sha256=row['canonical']['sha256'],
        created=time.time(), files=files, original_canonical_path=str(canonical)))
    audit('package_verified', uid=row['uid'], directory=str(dest), canonical_sha256=row['canonical']['sha256'])
    # The legacy manifest and completion receipts keep working through this link.
    alias(canonical, dest / 'asset.ply')
    return dest


def cleanup_completed(rows):
    """Delete only committed assets' superseded geometry, after package validation."""
    for row in rows:
        if row['status'] != 'keep' or not committed(row):
            continue
        dest = asset_dir(row)
        if not (dest / 'package_complete.json').exists() or (dest / 'cleanup_complete.json').exists():
            continue
        target = dest / 'asset.ply'
        if sha(target) != row['canonical']['sha256']:
            raise ValueError('Cannot prune unverified package')
        raw = Path(row['provenance']['raw_source_path'])
        audit_dir = Path(row['quality']['audit_image']).parent
        candidates = {raw, audit_dir / 'raw.ply', audit_dir / 'axis.ply', audit_dir / 'canonical.ply'}
        p = row['provenance']
        if p.get('batch_id') == B.name:
            chunk = B / 'chunks' / p['batch_chunk']
            for lib in [chunk / 'library', chunk / 'retained_library']:
                candidates.add(lib / row['canonical']['canonical_path'])
        records = []
        for path in sorted(candidates):
            if not path.is_file() or path.is_symlink():
                continue
            if not path.is_relative_to(WORK) or '/human' in str(path) or path.suffix != '.ply':
                raise ValueError(f'Unsafe pruning path: {path}')
            digest = sha(path)
            same = digest == row['canonical']['sha256']
            if path == raw and row['provenance'].get('raw_sha256') and digest != row['provenance']['raw_sha256']:
                raise ValueError('Raw payload changed since publication')
            if path.name == 'canonical.ply' and not same:
                continue
            stat = path.stat()
            record = dict(path=str(path), sha256=digest, bytes=stat.st_size,
                          geometry_action='deduplicate' if same else 'delete_intermediate',
                          replacement=str(target), freed_bytes=stat.st_size if stat.st_nlink == 1 else 0)
            audit('prune_authorized', uid=row['uid'], **record)
            if same:
                alias(path, target)
            else:
                path.unlink()
            audit('pruned', uid=row['uid'], **record)
            records.append(record)
        write(dest / 'cleanup_complete.json', dict(time=time.time(), files=records))


def publish_index(rows):
    kept = [r for r in rows if r['status'] == 'keep' and (asset_dir(r) / 'package_complete.json').exists()]
    aggregate = []
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['uid', 'category', 'dataset', 'scene_id', 'route', 'directory', 'ply', 'metadata'])
    categories = sorted({r['category'] for r in kept})
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>统一刚体资产库</title><style>body{font:16px/1.6 system-ui;background:#111923;color:#eef;margin:28px}a{color:#9cd3ff}img{max-width:330px;max-height:220px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #456;padding:8px}summary{font-size:22px;cursor:pointer}</style>'
    page += f'<h1>统一刚体资产库 · {len(kept)} 个合格资产</h1><p>按类别存放，每个资产包含 PLY、配置、原图与审核图。后台新增资产自动归档。</p><p><a href="rigid/asset_library_rigid.json">统一旧格式清单</a> · <a href="rigid/INDEX.csv">资产表格</a> · <a href="rigid/README.txt">插入说明</a></p>'
    for category in categories:
        group = [r for r in kept if r['category'] == category]
        page += f'<details><summary>{html.escape(category)} · {len(group)} 个</summary><table><tr><th>原图与审核图</th><th>资产</th><th>文件</th></tr>'
        for row in group:
            dest = asset_dir(row)
            relative = dest.relative_to(R).as_posix()
            local = read(dest / 'asset.json')
            names = {p.name for p in dest.iterdir() if p.is_file()}
            # Aggregate paths are relative to rigid/, per-asset paths to that asset directory.
            def rebase(value):
                if isinstance(value, dict):
                    return {k: rebase(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [rebase(v) for v in value]
                if isinstance(value, str) and value in names:
                    return relative + '/' + value
                return value
            aggregate.append(rebase(local))
            writer.writerow([row['uid'], category, row['dataset'], row['provenance']['scene_id'], row['route'], relative, relative + '/asset.ply', relative + '/asset.json'])
            base = 'rigid/' + relative + '/'
            page += '<tr><td><a href="' + base + 'source_and_renders.jpg"><img loading="lazy" src="' + base + 'source_and_renders.jpg"></a></td><td>' + html.escape(row['provenance']['source_obj_id']) + '<br>' + row['dataset'] + ' / ' + row['route'] + '</td><td>'
            for name, label in [('asset.ply','高斯PLY'),('asset.json','完整配置'),('binding.json','插入配置'),('source.jpg','原图'),('orbit_16.jpg','16视角'),('source_camera.json','源相机')]:
                page += f'<a href="{base + name}">{label}</a> · '
            page += '</td></tr>'
        page += '</table></details>'
    write(R / 'asset_library_rigid.json', dict(schema_version=2, total=len(aggregate), assets=aggregate))
    write(R / 'INDEX.csv', table.getvalue())
    write(R / 'README.txt', '统一刚体资产库：类别/路线_批次_数据集_对象ID_唯一后缀/。每个资产目录可单独复制给下游使用。\n'
          '路线 H 或 B0 来自真实生成记录；batch1000 为当前批量任务，pilot 为早期场景试验，bus_more 为额外公交车任务。\n'
          '批量使用旧 AssetLibrary 时，清单选择本目录 asset_library_rigid.json，pipeline_root 设为本目录。\n'
          '每个目录含标准 asset.ply、asset.json、binding.json、source_camera.json、原图、参考图、FLUX 图和16视角审核图。\n'
          'PLY 已规范朝向、居中和接地，未烘焙米制尺度；按 binding.json 的 scale 施加一次尺度。\n'
          '目标场景位置、轨迹、相机和遮挡由场景插入流程提供，本资产库不伪造这些数据。\n'
          '_review 只保留此前尚未通过全部条件的输出，不计入合格库。\n'
          '_archive/identity_variations 保存历史身份差异记录，保留原审核状态，不计入合格库。\n'
          'payloads 下旧路径为兼容链接；asset_work 中仅保留仍需运行、评估或溯源的内容。\n')
    write(L / 'rigid_index.html', page + '</html>')
    return len(kept)


async def main(args):
    A.mkdir(parents=True, exist_ok=True)
    R.mkdir(parents=True, exist_ok=True)
    with (A / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with (L / 'manifests/.publish.lock').open('a') as pub:
                fcntl.flock(pub, fcntl.LOCK_EX)
                data = read(L / 'manifests/asset_library_rigid.json')
            if not (A / 'manifest_before.json').exists():
                write(A / 'manifest_before.json', data)
            rows = [r for r in data['assets'] if r['status'] in {'keep', 'review'} and committed(r)]
            for i, row in enumerate(rows):
                package(row)
                if (i + 1) % 100 == 0:
                    print('Packaged', i + 1, '/', len(rows), flush=True)
            if args.cleanup:
                cleanup_completed(rows)
            count = publish_index(rows)
            states = {ds: read(B / f'{ds}_status.json') for ds in ['nuscenes', 'waymo']}
            done = all(s['status'] == 'target_reached' for s in states.values())
            write(A / 'status.json', dict(status='complete' if done else 'watching' if args.watch else 'snapshot_complete',
                updated=time.time(), kept_packaged=count, review_packaged=sum(r['status']=='review' for r in rows),
                production={k: dict(kept=v['kept'], status=v['status']) for k,v in states.items()}))
            print('Consolidated', count, 'qualified assets', flush=True)
            expected = sum(r['status'] == 'keep' for r in read(L / 'manifests/asset_library_rigid.json')['assets'])
            if not args.watch or (done and count == expected):
                return
            await asyncio.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--watch', action='store_true')
    asyncio.run(main(parser.parse_args()))
